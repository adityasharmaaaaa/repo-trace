"""Central model configuration.

All chat LLM calls go through Groq. Groq does not offer an embeddings API,
so embeddings default to a local sentence-transformers model (no key needed),
with Gemini embeddings available via EMBEDDING_PROVIDER=google.

Every default can be overridden through environment variables / .env.
"""
import os
from typing import Literal

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_groq import ChatGroq

load_dotenv()

Role = Literal["grading", "generation", "judge"]

# Grading/correction: high-volume, cheap structured-output calls.
# Generation: stronger synthesis. Judge: RAGAS evaluator.
_DEFAULT_MODELS = {
    "grading": "openai/gpt-oss-20b",
    "generation": "openai/gpt-oss-120b",
    "judge": "openai/gpt-oss-120b",
}
_ENV_KEYS = {
    "grading": "GROQ_GRADING_MODEL",
    "generation": "GROQ_GENERATION_MODEL",
    "judge": "GROQ_JUDGE_MODEL",
}

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "huggingface").lower()

# bge models are trained asymmetrically: queries get an instruction prefix,
# documents don't — preserving the query/document split the pipeline relies on.
HF_EMBEDDING_MODEL = os.getenv("HF_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
HF_QUERY_INSTRUCTION = os.getenv(
    "HF_QUERY_INSTRUCTION",
    "Represent this sentence for searching relevant passages: ",
)
GOOGLE_EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL", "gemini-embedding-001")


class GroqChat(ChatGroq):
    """ChatGroq that defaults structured output to native JSON-schema mode.

    The default (forced tool calling) is unreliable on gpt-oss models — they
    sometimes answer in prose instead of calling the tool, which Groq rejects
    with `tool_use_failed`. JSON-schema mode constrains the response directly.
    """

    def with_structured_output(self, schema=None, *, method="json_schema", **kwargs):
        return super().with_structured_output(schema, method=method, **kwargs)


def get_model_name(role: Role) -> str:
    return os.getenv(_ENV_KEYS[role], _DEFAULT_MODELS[role])


def get_llm(role: Role = "generation", temperature: float = 0) -> ChatGroq:
    return GroqChat(model=get_model_name(role), temperature=temperature)


def _hf_embedder(query_side: bool) -> Embeddings:
    from langchain_huggingface import HuggingFaceEmbeddings

    prompt = HF_QUERY_INSTRUCTION if query_side else ""
    return HuggingFaceEmbeddings(
        model_name=HF_EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True, "prompt": prompt},
        query_encode_kwargs={"normalize_embeddings": True, "prompt": prompt},
    )


def _google_embedder(task_type: str) -> Embeddings:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(model=GOOGLE_EMBEDDING_MODEL, task_type=task_type)


def get_document_embedder() -> Embeddings:
    if EMBEDDING_PROVIDER == "google":
        return _google_embedder("retrieval_document")
    return _hf_embedder(query_side=False)


def get_query_embedder() -> Embeddings:
    if EMBEDDING_PROVIDER == "google":
        return _google_embedder("retrieval_query")
    return _hf_embedder(query_side=True)
