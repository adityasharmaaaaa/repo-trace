from typing import List, Optional
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

EMBEDDING_MODEL = "gemini-embedding-001"


def build_vector_store(
    chunks: List[Document],
    persist_directory: str,
) -> Chroma:
    document_embedder = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL, task_type="retrieval_document"
    )
    vector_store=Chroma.from_documents(
        documents=chunks,
        embedding=document_embedder,
        persist_directory=persist_directory
    )
    return vector_store


def load_vector_store(persist_directory: str) -> Chroma:
    document_embedder= GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL, task_type="retrieval_document"
    )
    vector_store=Chroma(
        persist_directory=persist_directory,
        embedding_function=document_embedder
    )
    return vector_store


def query_store(
    vectorstore: Chroma,
    query: str,
    k: int = 5,
) -> List[Document]:
    query_embedder = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        task_type="retrieval_query",
    )

    query_vector = query_embedder.embed_query(query)

    results = vectorstore.similarity_search_by_vector(
        embedding=query_vector,
        k=k,
    )

    return results

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from ingestion.loader import load_repository
    from ingestion.splitter import split_documents

    repo = sys.argv[1] if len(sys.argv) > 1 else "."
    persist_dir = sys.argv[2] if len(sys.argv) > 2 else "./chroma_store"

    print(f"Loading and splitting {repo}...")
    docs = load_repository(repo)
    chunks = split_documents(docs)
    print(f"{len(docs)} documents -> {len(chunks)} chunks")

    print(f"Embedding and persisting to {persist_dir}...")
    vectorstore = build_vector_store(chunks, persist_dir)
    print("Done.")

    print("\nReloading from disk (simulating a fresh process)...")
    reloaded = load_vector_store(persist_dir)

    test_query = "what does router.py do?"
    print(f"\nTest query: {test_query!r}")
    results = query_store(reloaded, test_query, k=3)
    for i, doc in enumerate(results, 1):
        print(f"\n--- result {i} ({doc.metadata.get('source')}) ---")
        print(doc.page_content[:300])
