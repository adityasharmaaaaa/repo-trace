from typing import List
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_classic.retrievers.document_compressors import LLMChainFilter
from langchain_classic.retrievers import ContextualCompressionRetriever

EMBEDDING_MODEL = "gemini-embedding-001"


class AsymmetricMMRRetriever(BaseRetriever):
    vectorstore: Chroma
    query_embedder: GoogleGenerativeAIEmbeddings
    k: int = 5
    fetch_k: int = 15
    lambda_mult: float = 0.3

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:

        query_vector = self.query_embedder.embed_query(query)
        results = self.vectorstore.max_marginal_relevance_search_by_vector(
            embedding=query_vector,
            k=self.k,
            fetch_k=self.fetch_k,
            lambda_mult=self.lambda_mult,
        )
        return results
        


def build_mmr_retriever(
    vectorstore: Chroma,
    k: int = 5,
    fetch_k: int = 15,
    lambda_mult: float = 0.3,
) -> AsymmetricMMRRetriever:
    query_embedder = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        task_type="retrieval_query",
    )
    return AsymmetricMMRRetriever(
        vectorstore=vectorstore,
        query_embedder=query_embedder,
        k=k,
        fetch_k=fetch_k,
        lambda_mult=lambda_mult,
    )


def build_compressed_retriever(
    base_retriever: BaseRetriever,
    llm: ChatGoogleGenerativeAI,
) -> ContextualCompressionRetriever:
    
    _filter=LLMChainFilter.from_llm(llm)
    compressed_retriever = ContextualCompressionRetriever(
        base_compressor=_filter,
        base_retriever=base_retriever,
    )
    return compressed_retriever


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from indexing.vector_store import load_vector_store

    persist_dir = sys.argv[1] if len(sys.argv) > 1 else "./chroma_store"
    query = sys.argv[2] if len(sys.argv) > 2 else "unusual business records statistical deviation thresholds customer segments"

    vectorstore = load_vector_store(persist_dir)
    mmr_retriever = build_mmr_retriever(vectorstore)
    print(f"fetch_k: {mmr_retriever.fetch_k}")
    print(f"lambda_mult: {mmr_retriever.lambda_mult}")
    print(f"Query: {query!r}\n")
    print("=" * 70)
    print("MMR RESULTS (no compression)")
    print("=" * 70)
    mmr_results = mmr_retriever.invoke(query)
    for i, doc in enumerate(mmr_results, 1):
        print(f"\n--- result {i} ({doc.metadata.get('source')}) - {len(doc.page_content)} chars ---")
        print(doc.page_content[:200])

    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)
    compressed_retriever = build_compressed_retriever(mmr_retriever, llm)

    print("\n" + "=" * 70)
    print("COMPRESSED RESULTS")
    print("=" * 70)
    compressed_results = compressed_retriever.invoke(query)
    for i, doc in enumerate(compressed_results, 1):
        print(f"\n--- result {i} ({doc.metadata.get('source')}) - {len(doc.page_content)} chars ---")
        print(doc.page_content)

    print(f"\nMMR returned {len(mmr_results)} documents")

    for i, doc in enumerate(mmr_results, 1):
        print(
            f"\n--- MMR result {i} "
            f"({doc.metadata.get('source')}) "
            f"- {len(doc.page_content)} chars ---"
        )
        print(doc.page_content)

    print(f"\nCompressed retriever returned {len(compressed_results)} documents")

    for i, doc in enumerate(compressed_results, 1):
        print(
            f"\n--- Compressed result {i} "
            f"({doc.metadata.get('source')}) "
            f"- {len(doc.page_content)} chars ---"
        )
        print(doc.page_content)