"""
Milestone 10: RAGAS Evaluation
------------------------------------
Goal: put real numbers behind the seven components you've already
built and manually verified - not a new capability, a measurement of
the ones you have.

Uses ragas's current API (checked against current docs, not memory -
this library's interface has moved fast): SingleTurnSample /
EvaluationDataset, metrics as classes taking an explicit evaluator LLM,
not the old datasets.Dataset + lowercase-function pattern from older
tutorials.

Central design decision - resolved before writing the eval set:
  Most RAGAS metrics need a `reference` (ground-truth answer). Your
  pipeline deliberately produces NO answer for some queries (the OAuth
  case). There is no correct reference for a question with no correct
  answer - those need a separate custom check, not RAGAS metrics.

Assumptions this file makes (verify these before trusting the report):

1. The indexed corpus (./chroma_store) is the CSV "AI Data Analyst"
   Streamlit app repo (app.py, README.md, src/data/, src/tools/,
   src/graph/, etc.) - the __main__ smoke-test query in the graph file
   ("How does anomaly is detected") points at anomaly_tool.py in that
   repo, which is the only corpus I've actually seen. If chroma_store
   indexes something else, the happy-path questions below need to be
   swapped for ones grounded in the real corpus.
2. QueryType.query_type is one of "repo_specific" / "general" / "mixed"
   - inferred from route_after_classification's branching (repo_specific
   and mixed both go to refine; everything else goes to web_search), not
   read from grading/correction.py directly. Confirm against the actual
   Literal there.
3. check_correct_refusal() below uses a placeholder REFUSAL_PHRASES list
   since generator.py / guardrail/validator.py weren't available to read
   from. Replace REFUSAL_PHRASES with your actual refusal wording before
   trusting the refusal-accuracy numbers.
"""
import sys
import types # 1. Create a dummy module to satisfy ragas's hardcoded import
dummy_module = types.ModuleType("langchain_community.chat_models.vertexai")
dummy_module.ChatVertexAI = type("ChatVertexAI", (object,), {})
sys.modules["langchain_community.chat_models.vertexai"] = dummy_module
from typing import List, Optional, TypedDict

from ragas import EvaluationDataset, evaluate
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import Faithfulness, ResponseRelevancy, ContextPrecision, LLMContextRecall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from graph.build_graph import build_graph


# ---------------------------------------------------------------- config

# Judge model for RAGAS metrics. Deliberately NOT reusing the pipeline's
# own generation model (gemini-3.7-flash) as the judge - grading a model
# with itself biases scores toward whatever that model considers "good."
# Swap for whatever stronger/independent model you have access to.
JUDGE_MODEL = "gemini-3.7-flash"
EMBEDDING_MODEL = "gemini-embedding-001"

# Pulled verbatim from generator.py's system prompt (rule 2): the model is
# directly instructed to output this exact sentence when context is
# insufficient, so this isn't a guess at how it might phrase a refusal -
# it's the literal string the prompt tells it to produce.
REFUSAL_PHRASES = [
    "The provided context does not contain enough information to answer this question",
]


# ---------------------------------------------------------------- eval set

class EvalCase(TypedDict):
    question: str
    reference: Optional[str]   # ground-truth answer; None for "should correctly refuse" cases
    expected_no_answer: bool   # True for OAuth-style queries with no real answer in the repo


EVAL_SET: List[EvalCase] = [
    # ---- happy path: repo_specific, should grade "correct" on first retrieval ----
    {
        "question": "How does the anomaly detection tool decide whether a value is an outlier?",
        "reference": (
            "It's statistics-first, not LLM-based. By default it flags any value in a "
            "numeric column whose absolute z-score (relative to that column's mean and "
            "population standard deviation) exceeds a threshold of 3.0; an IQR method "
            "(more than 1.5x the interquartile range beyond Q1/Q3) is available as an "
            "alternative. Detection can run globally or within groups via an optional "
            "group column (e.g. per-region outliers instead of dataset-wide ones). A "
            "column/group is only checked if it has at least 5 non-null values, and "
            "results are sorted by how extreme they are and capped at a max count. The "
            "LLM's only role is explaining, in plain language, why an already-flagged "
            "record looks unusual - it never decides what counts as anomalous."
        ),
        "expected_no_answer": False,
    },
    {
        "question": "What SQL statements is the app allowed to run against uploaded data?",
        "reference": (
            "Only read-only SELECT or WITH (CTE) statements are permitted. A regex check "
            "rejects any query containing INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, "
            "REPLACE, ATTACH, DETACH, PRAGMA, or VACUUM before it's ever executed, and a "
            "query is only run if, after that check, it starts with SELECT or WITH. This "
            "is enforced because the LLM's generated SQL is never trusted with write "
            "access - it's an analyst tool, not an admin console. Results are also capped "
            "at a configurable row limit (500 by default) and marked truncated if the "
            "query returned more rows than that."
        ),
        "expected_no_answer": False,
    },
    {
        "question": "How does the app decide which specialized node handles a user's question?",
        "reference": (
            "A router node classifies the question into one of eight intents - sql_qa, "
            "chart, insight, anomaly, quality, forecast, dashboard, or general - using a "
            "Gemini structured-output call (RouterDecision, with an intent and a one-"
            "sentence reasoning field) run at temperature 0. A conditional graph edge then "
            "sends state to the matching node based on that classification. Every "
            "specialized node - regardless of which one ran - flows into a shared "
            "finalize node afterward, which appends the turn's answer to conversation "
            "history for the checkpointer to persist."
        ),
        "expected_no_answer": False,
    },
    {
        "question": "How is a table's overall data quality score calculated?",
        "reference": (
            "It's a heuristic 0-100 score starting at 100: up to 40 points are subtracted "
            "for average missingness across columns (weighted 2x), up to 30 for the "
            "duplicate-row percentage (weighted 3x), and up to 20 based on what fraction "
            "of columns have any flagged issue at all. Per-column checks feeding into "
            "this include missing-value percentage (with severity tiers at >5% and >20%), "
            "whether a column is 100% empty or constant across all rows, unexpected "
            "negative values (skipping columns like lat/latitude), and unparseable values "
            "in any column whose name contains 'date'. The score is clamped to a 0-100 "
            "range."
        ),
        "expected_no_answer": False,
    },
    {
        "question": "Does the app remember earlier questions in the same chat session, and how?",
        "reference": (
            "Yes - conversation memory is handled by LangGraph's MemorySaver checkpointer, "
            "keyed on a per-session thread_id (the Streamlit session's UUID), so follow-up "
            "questions like 'now break that down by product' can resolve against prior "
            "turns. This is an in-process, in-memory implementation, so it resets whenever "
            "the app restarts or the session ends; the project notes that swapping in a "
            "persistent checkpointer (SQLite/Postgres-backed) would be a straightforward "
            "change if durability across restarts were needed."
        ),
        "expected_no_answer": False,
    },

    # ---- designed to plausibly need refine_query's rewrite on a later attempt ----
    # Deliberately colloquial/vague phrasing with little lexical overlap to the
    # source docstrings (which talk about "read-only", "SELECT/WITH", "regex",
    # "forbidden keywords"), on the theory that the first MMR retrieval may miss
    # and refine_query has to rewrite it into something more retrievable. This
    # is a guess, not something verified against your logs - check
    # correction_retry_count in the run_case() output below; if it's 0, swap in
    # a genuinely weak query from your own history instead.
    {
        "question": "If someone tries to sneak a nasty command into the chat box, does the app just let it rip?",
        "reference": (
            "No. Generated SQL is checked before execution and any statement containing "
            "a write/DDL keyword (INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, REPLACE, "
            "ATTACH, DETACH, PRAGMA, VACUUM) is rejected outright; only SELECT/WITH "
            "statements are ever run, because the LLM's SQL is treated as untrusted input "
            "in what is meant to be a read-only analyst tool."
        ),
        "expected_no_answer": False,
    },

    # ---- expected_no_answer: genuinely not in the repo ----
    {
        "question": "How do I set up OAuth so different users can log into the app with their own accounts?",
        "reference": None,
        "expected_no_answer": True,
    },
    {
        "question": "Can I export a session's chat, SQL, and charts as a PDF or HTML report?",
        "reference": None,
        "expected_no_answer": True,
    },

    # ---- general: unrelated to the repo, should route through web_search_fallback ----
    {
        "question": "What is the CAP theorem in distributed systems?",
        "reference": (
            "The CAP theorem states that a distributed data store can only guarantee two "
            "of three properties at once during a network partition: Consistency (every "
            "read gets the latest write or an error), Availability (every request gets a "
            "non-error response, even if not the latest data), and Partition tolerance "
            "(the system keeps working despite network failures between nodes). Since "
            "partitions are unavoidable in real distributed systems, this is usually "
            "framed as a tradeoff between consistency and availability once a partition "
            "occurs, not a free choice among all three."
        ),
        "expected_no_answer": False,
    },

    # ---- mixed: combines a repo-specific fact with general domain knowledge ----
    {
        "question": (
            "This app's forecast tool uses a linear-trend-plus-seasonal-naive baseline "
            "instead of ARIMA - in general terms, what would ARIMA give me that this "
            "simpler approach doesn't?"
        ),
        "reference": (
            "The repo's forecast tool is a deliberately lightweight baseline - "
            "numpy.polyfit for the linear trend, plus an average-by-calendar-month "
            "seasonal adjustment only once at least 24 months of history exist - chosen "
            "to keep the dependency footprint small for a demo app, with a note that "
            "swapping in statsmodels/Prophet is a self-contained change if needed. ARIMA "
            "(AutoRegressive Integrated Moving Average), by contrast, explicitly models "
            "autocorrelation between recent observations and can difference the series to "
            "handle non-stationarity, generally capturing more complex trend/seasonal "
            "structure and irregular patterns than a single fixed linear trend can. That "
            "comes at the cost of needing more historical data, more tuning (choosing p/d/q "
            "orders or using auto-ARIMA), and a heavier dependency than the current "
            "numpy-only implementation."
        ),
        "expected_no_answer": False,
    },
]


# ---------------------------------------------------------------- run + collect

def run_case(graph, case: EvalCase) -> dict:
    """
    Invoke the compiled graph for one eval case and collect what RAGAS
    (or the custom refusal check) needs.
    """
    initial_state = {
        "original_query": case["question"],
        "current_query": case["question"],
        "documents": [],
        "grades": [],
        "overall_grade": None,
        "query_type": None,
        "answer": None,
        "guardrails_passed": None,
        "guardrails_errors": [],
        "retrieval_retry_count": 0,
        "correction_retry_count": 0,
        "guardrails_retry_count": 0,
    }

    result = graph.invoke(initial_state)

    # retrieved_contexts for RAGAS = the page_content of documents AFTER the
    # run completes - that's literally what generate_answer saw, which is
    # what "faithfulness to context" needs to mean here.
    retrieved_contexts = [doc.page_content for doc in result.get("documents", [])]

    query_type = result.get("query_type")

    return {
        "question": case["question"],
        "reference": case.get("reference"),
        "expected_no_answer": case["expected_no_answer"],
        "answer": result.get("answer") or "",
        "retrieved_contexts": retrieved_contexts,
        # path-identifying info for per-path slicing/reporting
        "overall_grade": result.get("overall_grade"),
        "query_type": query_type.query_type if query_type is not None else None,
        "retrieval_retry_count": result.get("retrieval_retry_count", 0),
        "correction_retry_count": result.get("correction_retry_count", 0),
        "guardrails_retry_count": result.get("guardrails_retry_count", 0),
        "guardrails_passed": result.get("guardrails_passed"),
        "guardrails_errors": result.get("guardrails_errors", []),
    }


def check_correct_refusal(answer: str) -> bool:
    """
    Custom check for expected_no_answer cases - did the system correctly
    decline rather than fabricate an answer?

    An empty answer counts as a correct refusal too (guardrails may have
    exhausted retries and returned nothing rather than an unvalidated
    answer). Otherwise this checks REFUSAL_PHRASES as a case-insensitive
    substring rather than requiring an exact-string match, purely as a
    guard against trivial punctuation/whitespace differences - the model
    is directly instructed to emit that exact sentence, so this should
    rarely need the slack. If REFUSAL_PHRASES ever grows beyond that one
    instructed string (e.g. a separate guardrail rejection message gets
    added), keep entries specific enough that a real answer which merely
    hedges ("I don't have exact figures, but revenue was...") can't match.
    """
    if not answer:
        return True
    lowered = answer.lower()
    return any(phrase.lower() in lowered for phrase in REFUSAL_PHRASES)


# ---------------------------------------------------------------- evaluate

def build_evaluator():
    """
    Wrap the judge LLM and embeddings for ragas's evaluator interface.

    Embedding task_type call: ResponseRelevancy works by having the judge
    LLM generate synthetic questions from the answer, then embeds those
    against the original question - both sides of that comparison are
    question-like text, not a document being matched against a query in
    the usual retrieval sense. So RETRIEVAL_QUERY is used for both sides
    here rather than RETRIEVAL_DOCUMENT, since neither text is playing
    the "document" role your pipeline's own retriever uses that type for.
    """
    judge_llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(model=JUDGE_MODEL, temperature=0))
    embeddings = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, task_type="RETRIEVAL_QUERY")
    )
    return judge_llm, embeddings


def run_evaluation():
    """
    Full harness: run every EVAL_SET case through the graph, split results
    into the RAGAS-scored group and the refusal-checked group, run
    ragas.evaluate() on the former, report both - sliced by which graph
    path each case actually took.
    """
    graph = build_graph()

    ragas_records = []
    refusal_records = []

    for case in EVAL_SET:
        record = run_case(graph, case)
        if case["expected_no_answer"]:
            refusal_records.append(record)
        else:
            ragas_records.append(record)

    # ---- RAGAS-scored group ----
    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["retrieved_contexts"] or [""],
            reference=r["reference"],
        )
        for r in ragas_records
    ]
    dataset = EvaluationDataset(samples=samples)

    judge_llm, embeddings = build_evaluator()
    metrics = [
        Faithfulness(llm=judge_llm),
        ResponseRelevancy(llm=judge_llm, embeddings=embeddings),
        ContextPrecision(llm=judge_llm),
        LLMContextRecall(llm=judge_llm),
    ]
    ragas_results = evaluate(dataset=dataset, metrics=metrics)
    results_df = ragas_results.to_pandas()

    # ---- refusal-checked group ----
    refusal_correct = sum(1 for r in refusal_records if check_correct_refusal(r["answer"]))
    refusal_total = len(refusal_records)

    # ---- report ----
    metric_cols = [c for c in ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
                   if c in results_df.columns]

    print("=" * 78)
    print("RAGAS METRIC SCORES (answerable cases)")
    print("=" * 78)
    print(results_df[["user_input", *metric_cols]].to_string(index=False))
    print()
    print("Overall averages:")
    for col in metric_cols:
        print(f"  {col}: {results_df[col].mean():.3f}")

    print()
    print("Per-path breakdown:")
    for r in ragas_records:
        path = "corrected " if r["correction_retry_count"] > 0 else "first-try "
        multi_retrieve = " (multi-retrieval)" if r["retrieval_retry_count"] > 1 else ""
        print(f"  [{path}] grade={str(r['overall_grade']):9s} type={str(r['query_type']):14s}"
              f"{multi_retrieve} -> {r['question'][:60]}")

    print()
    print("=" * 78)
    print("REFUSAL ACCURACY (unanswerable cases)")
    print("=" * 78)
    print(f"  {refusal_correct}/{refusal_total} correctly declined")
    for r in refusal_records:
        verdict = " OK " if check_correct_refusal(r["answer"]) else "FAIL"
        print(f"  [{verdict}] {r['question'][:60]} -> {r['answer'][:80]!r}")

    return {
        "ragas_results_df": results_df,
        "refusal_accuracy": (refusal_correct, refusal_total),
        "ragas_records": ragas_records,
        "refusal_records": refusal_records,
    }


if __name__ == "__main__":
    run_evaluation()