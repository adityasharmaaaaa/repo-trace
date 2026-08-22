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


# -----------------------------------
# Swapped to 1.5-flash to support n > 1 candidate generation for ResponseRelevancy
JUDGE_MODEL = "gemini-3.6-flash"
EMBEDDING_MODEL = "gemini-embedding-001"

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
        "question": "How do I configure Redis to handle distributed rate limiting for the API?",
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

    retrieved_contexts = [doc.page_content for doc in result.get("documents", [])]

    query_type = result.get("query_type")

    return {
        "question": case["question"],
        "reference": case.get("reference"),
        "expected_no_answer": case["expected_no_answer"],
        "answer": result.get("answer") or "",
        "retrieved_contexts": retrieved_contexts,
        "overall_grade": result.get("overall_grade"),
        "query_type": query_type.query_type if query_type is not None else None,
        "retrieval_retry_count": result.get("retrieval_retry_count", 0),
        "correction_retry_count": result.get("correction_retry_count", 0),
        "guardrails_retry_count": result.get("guardrails_retry_count", 0),
        "guardrails_passed": result.get("guardrails_passed"),
        "guardrails_errors": result.get("guardrails_errors", []),
    }


def check_correct_refusal(answer: str) -> bool:
    if not answer:
        return True
    lowered = answer.lower()
    return any(phrase.lower() in lowered for phrase in REFUSAL_PHRASES)


# ---------------------------------------------------------------- evaluate

def build_evaluator():
    judge_llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(model=JUDGE_MODEL, temperature=0))
    embeddings = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, task_type="RETRIEVAL_QUERY")
    )
    return judge_llm, embeddings


def run_evaluation():
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
        #ResponseRelevancy(llm=judge_llm, embeddings=embeddings),
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
        valid_count = results_df[col].count()
        mean_val = results_df[col].mean()
        print(f"  {col}: {mean_val:.3f} (n={valid_count})")

    print()
    print("Per-path breakdown:")
    for r in ragas_records:
        path = "corrected " if r["correction_retry_count"] > 0 else "first-try "
        multi_retrieve = " (multi-retrieval)" if r["retrieval_retry_count"] > 1 else ""
        guard_retry = f" (guard-retries: {r['guardrails_retry_count']})" if r["guardrails_retry_count"] > 0 else ""
        print(f"  [{path}] grade={str(r['overall_grade']):9s} type={str(r['query_type']):14s}"
              f"{multi_retrieve}{guard_retry} -> {r['question'][:60]}")

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