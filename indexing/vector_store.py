from typing import List, Optional
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
import hashlib
from pathlib import Path

EMBEDDING_MODEL = "gemini-embedding-001"


def make_chunk_id(doc: Document, chunk_index: int) -> str:
    source = doc.metadata.get("source", "unknown")

    # Make the source path stable
    source = str(Path(source).resolve())

    # Stable identity for the source file
    file_id = hashlib.sha256(
        source.encode("utf-8")
    ).hexdigest()

    return f"{file_id}:{chunk_index}"


def build_vector_store(
    chunks: List[Document],
    persist_directory: str,
) -> Chroma:
    document_embedder = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL, task_type="retrieval_document"
    )
    vector_store = Chroma(
        embedding_function=document_embedder,
        persist_directory=persist_directory
    )

    # Track chunk index PER SOURCE FILE, not globally across the repo
    source_counters = {}
    ids = []
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        current_idx = source_counters.get(source, 0)
        ids.append(make_chunk_id(chunk, current_idx))
        source_counters[source] = current_idx + 1

    vector_store.add_documents(
        documents=chunks,
        ids=ids
    )

    return vector_store


def load_vector_store(persist_directory: str) -> Chroma:
    document_embedder = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL, task_type="retrieval_document"
    )
    vector_store = Chroma(
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
    print(f"{len(docs)} documents -> {len(chunks)} chunks globally")

    print(f"Embedding and persisting to {persist_dir}...")
    vectorstore = build_vector_store(chunks, persist_dir)
    
    db_count = vectorstore._collection.count()
    print(f"Total chunks currently in Chroma DB: {db_count}")
    print("Done.")

    print("\nReloading from disk (simulating a fresh process)...")
    reloaded = load_vector_store(persist_dir)

    test_query = "what does router.py do?"
    print(f"\nTest query: {test_query!r}")
    results = query_store(reloaded, test_query, k=3)
    for i, doc in enumerate(results, 1):
        print(f"\n--- result {i} ({doc.metadata.get('source')}) ---")
        print(doc.page_content[:300])

    router_data = reloaded._collection.get(
        where={"source": "src/graph/router.py"} 
    )
    
    print(f"\n--- ID Stability Check for router.py ---")
    print(f"Total chunks for this file: {len(router_data['ids'])}")
    for chunk_id in sorted(router_data['ids']):
        print(chunk_id)