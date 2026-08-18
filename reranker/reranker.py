from typing import List
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
reranker = CrossEncoder(RERANKER_MODEL)
def rerank_documents(
    query: str,
    documents: List[Document],
    top_n: int = 5,
) -> List[Document]:
    
    if not documents:
        return []
    pairs=[
        (query,doc.page_content) for doc in documents
    ]

    scores = reranker.predict(pairs)
    scored_documents = []

    for doc,score in zip(documents,scores):
        doc.metadata["rerank_score"]=float(score)
        scored_documents.append((doc,score))

    scored_documents.sort(
        key=lambda x: x[1], reverse=True
    )

    return [
        doc for doc,_ in scored_documents[:top_n]
    ]

if __name__ == "__main__":
    # Deliberately adversarial test: relevant doc buried last, on an
    # unrelated topic first - reranking should flip this.
    query = "how does the router decide which node to call?"
    test_docs = [
        Document(
            page_content="This module generates synthetic sample sales data for demos.",
            metadata={"source": "scripts/generate_sample_data.py"},
        ),
        Document(
            page_content="Forecasting uses a linear-trend and seasonal-naive baseline.",
            metadata={"source": "src/tools/forecast_tool.py"},
        ),
        Document(
            page_content=(
                "def router_node(state): classifies the user's question into "
                "one of the intents, using structured output. Every other node "
                "is reached only through this classification."
            ),
            metadata={"source": "src/graph/router.py"},
        ),
    ]

    print(f"Query: {query!r}\n")
    print("BEFORE reranking (original order):")
    for i, doc in enumerate(test_docs, 1):
        print(f"  {i}. {doc.metadata.get('source')}")

    reranked = rerank_documents(query, test_docs, top_n=3)

    print("\nAFTER reranking:")
    for i, doc in enumerate(reranked, 1):
        score = doc.metadata.get("rerank_score", "?")
        print(f"  {i}. {doc.metadata.get('source')} (score: {score})")
