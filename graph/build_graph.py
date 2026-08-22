from typing import List, Literal, Optional, TypedDict, Annotated, Dict
import operator
from pathlib import Path

from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END

from indexing.vector_store import load_vector_store
from retrieval.mmr_retriever import build_mmr_retriever, build_compressed_retriever
from grading.grader import grade_document, aggregate_grades, DocumentGrade
from grading.correction import classify_query, refine_query, web_search_fallback, QueryType
from reranker.reranker import rerank_documents
from generator.generator import generate_answer
from guardrail.validator import build_guarded_generation_guard

# ---------------------------------------------------------------- config

LLM_MODEL = "gemini-3.7-flash"
MAX_CORRECTION_RETRIES = 2
MAX_GUARDRAILS_RETRIES = 2

DEBUG = True

llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL,
    temperature=0,
)
vectorstore = load_vector_store("./chroma_store")
mmr_retriever = build_mmr_retriever(vectorstore)
compressed_retriever = build_compressed_retriever(mmr_retriever, llm)
guard = build_guarded_generation_guard()

# ---------------------------------------------------------------- state


class AgentState(TypedDict):
    original_query: str
    current_query: str
    documents: List[Document]
    grades: List[DocumentGrade]
    overall_grade: Optional[Literal["correct", "ambiguous", "incorrect"]]
    query_type: Optional[QueryType]
    answer: Optional[str]
    guardrails_passed: Optional[bool]
    guardrails_errors: List[str]
    retrieval_retry_count: int
    correction_retry_count: int
    guardrails_retry_count: int


# ---------------------------------------------------------------- helpers


def _normalize_source(source: Optional[str]) -> Optional[str]:
    if not source:
        return source
    path = Path(source)
    if not path.is_absolute():
        return source
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return source


def _is_mixed(state: "AgentState") -> bool:
    query_type = state.get("query_type")
    return query_type is not None and query_type.query_type == "mixed"


# ---------------------------------------------------------------- nodes


def retrieve_node(state: AgentState) -> dict:
    """Run MMR + contextual compression against state["current_query"]."""
    try:
        docs = compressed_retriever.invoke(state["current_query"])
    except ValueError as e:
        if DEBUG:
            print(f"[retrieve_node] Compressor failed: {e}. Falling back to base MMR.")
        docs = mmr_retriever.invoke(state["current_query"])

    return {
        "documents": docs,
        "retrieval_retry_count": state.get("retrieval_retry_count", 0) + 1,
    }


def grade_node(state: AgentState) -> dict:
    """Grade every document in state["documents"], aggregate."""
    grades = []
    for doc in state.get("documents", []):
        grade_result = grade_document(state["current_query"], doc, llm)
        grades.append(grade_result)

    overall = aggregate_grades(grades) if grades else "incorrect"
    if DEBUG:
        print(f"[grade_node] overall_grade={overall} ({len(grades)} docs graded)")
    return {"grades": grades, "overall_grade": overall}


def classify_node(state: AgentState) -> dict:
    query_type = classify_query(state["original_query"], llm)
    if DEBUG:
        print(f"[classify_node] fired -> query_type={query_type.query_type}")
    return {"query_type": query_type}


def refine_node(state: AgentState) -> dict:
    if DEBUG:
        print("[refine_node] fired")
    refined = refine_query(
        original_query=state["original_query"],
        graded_documents=state.get("documents", []),
        llm=llm,
    )
    return {
        "current_query": refined,
        "correction_retry_count": state.get("correction_retry_count", 0) + 1,
    }


def web_search_node(state: AgentState) -> dict:
    if DEBUG:
        print("[web_search_node] fired")
    web_docs = web_search_fallback(state["original_query"])
    merged = state.get("documents", []) + web_docs
    return {"documents": merged}


def rerank_node(state: AgentState) -> dict:
    reranked = rerank_documents(
        state["original_query"],
        state.get("documents", []),
        top_n=5,
    )
    return {"documents": reranked}


def generate_node(state: AgentState) -> dict:
    answer = generate_answer(state["original_query"], state.get("documents", []), llm)
    return {"answer": answer}


def guardrails_node(state: AgentState) -> dict:
    raw_documents = state.get("documents", [])
    valid_sources = {
        _normalize_source(doc.metadata.get("source"))
        for doc in raw_documents
        if doc.metadata.get("source")
    }

    if DEBUG:
        print(f"[guardrails_node] valid_sources={valid_sources}")
        print(f"[guardrails_node] raw sources={[d.metadata.get('source') for d in raw_documents]}")

    result = guard.validate(
        state.get("answer") or "",
        metadata={"valid_sources": valid_sources},
    )

    if result.validation_passed:
        return {"guardrails_passed": True, "guardrails_errors": []}

    errors = [
        summary.failure_reason
        for summary in result.validation_summaries
        if summary.validator_status == "fail"
    ]
    return {
        "guardrails_passed": False,
        "guardrails_errors": errors,
        "guardrails_retry_count": state.get("guardrails_retry_count", 0) + 1,
    }


# ------------------------------------------------------------- routing


def route_after_grading(state: AgentState) -> str:
    if state.get("overall_grade") == "correct":
        if _is_mixed(state):
            return "web_search"
        return "rerank"
    return "classify"


def route_after_classification(state: AgentState) -> str:
    query_type = state.get("query_type")
    qt = query_type.query_type if query_type is not None else None

    if qt in ("repo_specific", "mixed"):
        return "refine"
    return "web_search"


def route_after_refine(state: AgentState) -> str:
    if state.get("correction_retry_count", 0) >= MAX_CORRECTION_RETRIES:
        if _is_mixed(state):
            return "web_search"
        return "rerank"
    return "retrieve"


def route_after_guardrails(state: AgentState) -> str:
    if state.get("guardrails_passed"):
        return "end"
    if state.get("guardrails_retry_count", 0) >= MAX_GUARDRAILS_RETRIES:
        return "end"
    return "generate"


# ------------------------------------------------------------- build


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("classify", classify_node)
    graph.add_node("refine", refine_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("generate", generate_node)
    graph.add_node("guardrails", guardrails_node)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade")

    graph.add_conditional_edges(
        "grade",
        route_after_grading,
        {
            "rerank": "rerank",
            "web_search": "web_search",
            "classify": "classify",
        },
    )

    graph.add_conditional_edges(
        "classify",
        route_after_classification,
        {
            "refine": "refine",
            "web_search": "web_search",
        },
    )

    graph.add_conditional_edges(
        "refine",
        route_after_refine,
        {
            "retrieve": "retrieve",
            "rerank": "rerank",
            "web_search": "web_search",
        },
    )

    graph.add_edge("web_search", "rerank")
    graph.add_edge("rerank", "generate")
    graph.add_edge("generate", "guardrails")

    graph.add_conditional_edges(
        "guardrails",
        route_after_guardrails,
        {
            "generate": "generate",
            "end": END,
        },
    )

    return graph.compile()


if __name__ == "__main__":
    graph = build_graph()
    print(graph.get_graph().print_ascii())

    result = graph.invoke(
        {
            "original_query": "How do I configure Redis to handle distributed rate limiting for the API?",
            "current_query": "How do I configure Redis to handle distributed rate limiting for the API?",
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
    )
    print(result["original_query"])
    print("\n--- FINAL ANSWER ---")
    print(result["answer"])
    print("\n--- GUARDRAILS ---")
    print("passed:", result["guardrails_passed"], "errors:", result["guardrails_errors"])