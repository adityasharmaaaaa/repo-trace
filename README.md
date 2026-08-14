# Codebase Onboarding & Debugging Copilot (CRAG)

Project 1 of the Advanced RAG portfolio. Architecture: Corrective RAG (CRAG).

## Milestone plan
1. **Ingestion** - `ingestion/loader.py`, `ingestion/splitter.py` *(current)*
2. **Indexing** - embeddings + Chroma vector store
3. **Base retrieval** - MMR + Contextual Compression Retriever
4. **Grading node** - LLM scores retrieved chunks correct/ambiguous/incorrect
5. **Correction path** - query refinement + web search fallback (Tavily)
6. **Reranking** - cross-encoder on merged evidence
7. **Generation** - grounded answer with file citations (LCEL)
8. **Guardrails** - validate output before it reaches the user
9. **Wire the LangGraph StateGraph** - nodes + conditional edges end to end
10. **Eval harness** - RAGAS against a small hand-written Q&A set

## Setup
```
pip install -r requirements.txt
cp .env.example .env   # fill in OPENAI_API_KEY (and TAVILY_API_KEY later, for milestone 5)
```

## Testing Milestone 1
```
cd ingestion
python loader.py /path/to/some/cloned/repo
python splitter.py /path/to/some/cloned/repo
```
Pick a repo you actually know well - your own project, or something
mid-sized and Python-heavy you've contributed to. You'll be asking it
questions in later milestones, so it should be something you can
independently verify the answers to.
