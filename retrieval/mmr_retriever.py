"""
Milestone 3: MMR + Contextual Compression Retriever
------------------------------------------------------
Goal: replace plain top-k similarity search with a retriever that (a)
actively avoids returning near-duplicate/redundant chunks (MMR), and
(b) trims each returned chunk down to only the part relevant to the
query before it reaches the LLM (Contextual Compression).

Open design question from the conversation - resolve it before you
write build_mmr_retriever():
  Does vectorstore.as_retriever(search_type="mmr") have the same
  query/document embedding asymmetry problem you already found and
  fixed in query_store()? Check what embedding_function the
  vectorstore you get from load_vector_store() is bound to, and check
  what as_retriever() actually calls under the hood. If there's a
  problem, Chroma has a *_by_vector variant of MMR search, the same
  pattern you already used for similarity_search_by_vector - go find
  its exact name and signature rather than guessing.
"""
from typing import List
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.retrievers import ContextualCompressionRetriever
# You'll need to pick a compressor - see the docstring on
# build_compressed_retriever() for the tradeoff between options.
# from langchain.retrievers.document_compressors import ...

EMBEDDING_MODEL = "gemini-embedding-001"


class AsymmetricMMRRetriever(BaseRetriever):
    """
    A retriever that runs MMR search against a Chroma store while
    correctly using a query-type embedder for the incoming query -
    resolving the asymmetry question posed above.

    This subclasses LangChain's BaseRetriever directly rather than
    using vectorstore.as_retriever(), specifically so we control which
    embedder embeds the query.

    Fields (BaseRetriever is a pydantic model, so declare these as
    class-level type-annotated attributes, not __init__ args):
        vectorstore: Chroma
        query_embedder: GoogleGenerativeAIEmbeddings
        k: int              # final number of documents returned
        fetch_k: int        # candidate pool size before diversity filtering
        lambda_mult: float  # 0 = max diversity, 1 = max relevance

    TODO(you): implement _get_relevant_documents.

    Hints:
      - Embed the query yourself with self.query_embedder - do NOT let
        the vectorstore embed it for you (that's the asymmetry bug).
      - Chroma's MMR-by-vector method takes k, fetch_k, and
        lambda_mult - look up its exact name and argument order.
      - fetch_k should be meaningfully larger than k - MMR needs a
        pool of candidates to pick a diverse subset FROM. If
        fetch_k == k, MMR has nothing to select against and behaves
        like plain similarity search. Pick actual numbers and be
        ready to justify them - what pool size makes sense for a
        ~106-chunk repo store versus one with tens of thousands of
        chunks?
      - lambda_mult: for a codebase Q&A tool, do you want to lean
        toward relevance or diversity? Think about what a wrong answer
        looks like in each direction - too much diversity might trade
        away the single best-matching function for variety; too much
        relevance brings back the redundancy problem MMR exists to fix.
        Pick a value and write down your reasoning.
    """

    vectorstore: Chroma
    query_embedder: GoogleGenerativeAIEmbeddings
    k: int = 5
    fetch_k: int = 20
    lambda_mult: float = 0.5

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        # TODO: implement
        raise NotImplementedError


def build_mmr_retriever(
    vectorstore: Chroma,
    k: int = 5,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
) -> AsymmetricMMRRetriever:
    """
    Convenience constructor - builds the query-type embedder and wraps
    it with the vectorstore into an AsymmetricMMRRetriever.
    """
    # TODO: implement (should be a few lines - construct the query
    # embedder with the correct task_type, then construct and return
    # an AsymmetricMMRRetriever)
    raise NotImplementedError


def build_compressed_retriever(
    base_retriever: BaseRetriever,
    llm: ChatGoogleGenerativeAI,
) -> ContextualCompressionRetriever:
    """
    Wrap base_retriever with a compressor that trims each retrieved
    chunk down to the part relevant to the query.

    Decision for you to make: LangChain offers a few compressor
    strategies with real tradeoffs -
      - LLMChainExtractor: uses an LLM call per retrieved document to
        extract only the relevant text. Most precise, costs an LLM
        call per document per query.
      - LLMChainFilter: uses an LLM to decide keep/drop per document
        (binary), doesn't trim within a document. Cheaper than
        Extractor, less granular.
      - EmbeddingsFilter: no LLM call at all - drops documents below a
        similarity threshold to the query. Cheapest and fastest, but
        can't trim partial content out of a chunk it keeps.

    Given that our chunks are already fairly tightly scoped (a single
    function, thanks to Milestone 1's language-aware splitting) - which
    of these actually earns its cost here? Pick one and be ready to
    explain why the others were worse fits for THIS project's chunk
    shape, not just "it's the fanciest option."

    Hints:
      - ContextualCompressionRetriever(base_compressor=..., base_retriever=...)
      - Whichever compressor needs an LLM, use `llm` (ChatGoogleGenerativeAI)
        passed into this function - don't construct a new one here.
    """
    # TODO: implement
    raise NotImplementedError


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from indexing.vector_store import load_vector_store

    persist_dir = sys.argv[1] if len(sys.argv) > 1 else "./chroma_store"
    query = sys.argv[2] if len(sys.argv) > 2 else "how does the router decide which node to call?"

    vectorstore = load_vector_store(persist_dir)
    mmr_retriever = build_mmr_retriever(vectorstore)

    print(f"Query: {query!r}\n")
    print("=" * 70)
    print("MMR RESULTS (no compression)")
    print("=" * 70)
    mmr_results = mmr_retriever.invoke(query)
    for i, doc in enumerate(mmr_results, 1):
        print(f"\n--- result {i} ({doc.metadata.get('source')}) - {len(doc.page_content)} chars ---")
        print(doc.page_content[:200])

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    compressed_retriever = build_compressed_retriever(mmr_retriever, llm)

    print("\n" + "=" * 70)
    print("COMPRESSED RESULTS")
    print("=" * 70)
    compressed_results = compressed_retriever.invoke(query)
    for i, doc in enumerate(compressed_results, 1):
        print(f"\n--- result {i} ({doc.metadata.get('source')}) - {len(doc.page_content)} chars ---")
        print(doc.page_content[:200])
