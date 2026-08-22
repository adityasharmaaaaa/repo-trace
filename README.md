# repo-trace

A Corrective RAG (CRAG) assistant for onboarding to and debugging an unfamiliar codebase. Instead of retrieving-then-answering blindly, it grades its own retrieval quality, classifies *why* retrieval fell short, and corrects — refining the query, falling back to the web, or reranking — before it ever generates an answer.

Tested end-to-end against a real, independently built codebase (`ai-data-analyst`, an AI-powered CSV data-analyst app), not a toy example — so every design decision and bug below came from an actual retrieval failure, not a synthetic one.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Key Design Decisions](#key-design-decisions)
- [Real Bugs Found and Fixed](#real-bugs-found-and-fixed)
- [Known Limitations](#known-limitations)
- [Evaluation Results](#evaluation-results)
- [Setup / How to Run](#setup--how-to-run)
- [Project Structure](#project-structure)
- [What's Next](#whats-next--what-id-do-differently)

---

## Problem Statement

Onboarding to an unfamiliar codebase is slow, and most RAG demos make it worse in a subtle way: they answer confidently even when retrieval was weak. A vector search that returns three barely-related chunks still gets stuffed into a prompt, and the LLM will happily synthesize a plausible-sounding — and wrong — answer from them.

repo-trace is built around one idea: **grade retrieval before trusting it.** Every retrieved document is graded `correct` / `ambiguous` / `incorrect` against the specific query, and that three-way grade drives a different corrective action — not just "retry the same search again." A query that's genuinely about general programming knowledge gets routed to the web instead of endlessly re-querying a codebase that will never contain the answer. A query that's ambiguous inside the repo gets its search terms refined using vocabulary pulled from what *was* retrieved. This distinguishes it from plain RAG (embed → retrieve → generate) and from naive corrective loops (retrieve → grade → retry the same query).

---

## Architecture

repo-trace is a [LangGraph](https://langchain-ai.github.io/langgraph/) `StateGraph` with eight nodes:

| Node | Responsibility |
|---|---|
| `retrieve` | Runs the MMR + contextual-compression retriever against the current query |
| `grade` | Grades every retrieved document (`correct` / `ambiguous` / `incorrect`) and aggregates to an overall grade |
| `classify` | Classifies the query as `repo_specific`, `general`, or `mixed` — only reached once grading has already judged retrieval insufficient |
| `refine` | Rewrites the query using vocabulary extracted from what was retrieved (or broadens it if nothing was retrieved) |
| `web_search` | Falls back to a live web search (Tavily) for general-knowledge evidence |
| `rerank` | Cross-encoder reranking of the accumulated document pool down to the top candidates |
| `generate` | Generates the final answer, strictly grounded in the retrieved context, with inline citations |
| `guardrails` | Validates that every citation in the generated answer actually points at a document that was in evidence for that call |

### Two independent retry loops

repo-trace tracks two separate counters, deliberately not shared:

- **`correction_retry_count`** — caps how many times the retrieval/refine loop can run (`retrieve → grade → classify → refine → retrieve → …`).
- **`guardrails_retry_count`** — caps how many times `generate → guardrails` can loop after a citation-validity failure.

They're kept separate because they guard against two unrelated failure modes: bad *retrieval* versus a *generation* step that hallucinates a citation despite having good evidence in front of it. Sharing one counter would let a run burn its entire retry budget on one failure type and have nothing left for the other.

### Routing logic

- **After grading:** `correct` → skip straight to `rerank` (unless the query was classified `mixed`, in which case it still goes to `web_search` first to pick up the general-knowledge half). Anything else (`ambiguous` / `incorrect`) → `classify`.
- **After classification:** `repo_specific` or `mixed` → `refine` (loop back into another retrieval attempt). `general` → straight to `web_search`.
- **After refine:** loop back to `retrieve`, unless `correction_retry_count` has hit its cap — then fall through to `rerank` (or `web_search` first, if `mixed`).
- **After guardrails:** pass → done. Fail and under the retry cap → back to `generate`. Fail and over the cap → done anyway (the answer ships with whatever citation issues remain, rather than looping forever).

```
                 START
                   │
                   ▼
              ┌─────────┐
     ┌───────▶│retrieve │◀────────────┐
     │        └────┬────┘             │
     │             ▼                  │
     │        ┌─────────┐             │
     │        │  grade  │             │
     │        └────┬────┘             │
     │             │                  │
     │     correct │  not-correct     │
     │             │       │          │
     │             ▼       ▼          │
     │        ┌────────┐ ┌──────────┐ │
     │        │ rerank │ │ classify │ │
     │        └───┬────┘ └────┬─────┘ │
     │            │      repo_specific/
     │            │        mixed│  general
     │            │            ▼    │
     │            │       ┌────────┐│
     │            └───────┤ refine ││
     │                    └───┬────┘│
     │                        │     │
     │            retry < cap │     │
     └────────────────────────┘     │
                                     ▼
                              ┌────────────┐
                              │ web_search │
                              └─────┬──────┘
                                    │
                                    ▼
                              ┌──────────┐
                              │ generate │◀─────────┐
                              └────┬─────┘          │
                                   ▼                │
                              ┌────────────┐  fail, retry < cap
                              │ guardrails │─────────┘
                              └─────┬──────┘
                                    │ pass, or retry ≥ cap
                                    ▼
                                   END
```

*(Run `python -m graph.build_graph` for the exact ASCII graph LangGraph itself prints — the diagram above is a redrawn, plain-language version of that output.)*

---

## Tech Stack

| Layer | Choice | Location |
|---|---|---|
| Document Loader | `DirectoryLoader`, Python and Markdown loaded as two separate passes | `ingestion/loader.py` |
| Text Splitter | `RecursiveCharacterTextSplitter.from_language` — `Language.PYTHON` at `chunk_size=1500 / overlap=100`, `Language.MARKDOWN` at `chunk_size=1000 / overlap=100` (sizes chosen from measured function-boundary evidence, not defaults) | `ingestion/splitter.py` |
| Embedding Model | `gemini-embedding-001`, used **asymmetrically** — a separate query-type embedder and document-type embedder | throughout `indexing/`, `retrieval/` |
| Vector Store | Chroma, persisted to disk, idempotent via stable per-file chunk IDs | `indexing/vector_store.py` |
| Advanced Retrieval | MMR via a custom `AsymmetricMMRRetriever`, `k=5, fetch_k=15, lambda_mult=0.3` | `retrieval/mmr_retriever.py` |
| Query Optimization | Contextual Compression via a custom `StructuredRelevanceFilter` (replaced `LLMChainFilter` — see [Real Bugs](#real-bugs-found-and-fixed)) | `retrieval/mmr_retriever.py` |
| SOTA Architecture | Corrective RAG (CRAG) as a LangGraph `StateGraph` | `graph/build_graph.py` |
| Reranking | Local cross-encoder, `cross-encoder/ms-marco-MiniLM-L-6-v2` | `reranker/reranker.py` |
| Guardrails | `guardrails-ai`, custom `ValidCitations` validator checking every citation against the actual per-call evidence set | `guardrail/validator.py` |
| Evaluation | RAGAS — Faithfulness, Context Precision, LLM Context Recall | `evals/ragas_eval.py` |
| LLMs | `gemini-3.5-flash` (grading & correction), `gemini-3.7-flash` (generation), `gemini-3.6-flash` (RAGAS judge) | — |

**Why three different Gemini models?** Grading and correction are high-volume, low-complexity structured-output calls, so the cheaper `gemini-3.5-flash` handles those. Generation needs stronger synthesis, so it runs on `gemini-3.7-flash`. The RAGAS judge runs on `gemini-3.6-flash` instead — `3.7-flash` turned out to be incompatible with the multi-candidate generation requests RAGAS issues internally, so the judge model was pinned to `3.6-flash` specifically to work around that.

---

## Key Design Decisions

- **Asymmetric query/document embeddings.** Queries and documents play different roles in retrieval — a short question and a long code chunk aren't the same kind of text — so they get separate embedder configurations (`retrieval_query` vs `retrieval_document` task types) rather than one embedder used for both.
- **Per-file stable chunk IDs.** Chunk IDs are `sha256(resolved file path) : per-file chunk index`, not a global running index. This makes reindexing idempotent and immune to file-ordering changes — editing one file doesn't shift the IDs of every chunk that comes after it.
- **MMR params tuned toward diversity.** `fetch_k=15` with `lambda_mult=0.3` deliberately leans toward diversity over pure similarity, because codebases are full of many similar-but-distinct chunks (near-duplicate helper functions, repeated boilerplate) that would otherwise crowd out genuinely different, relevant results.
- **Compression compressor: replaced, not just chosen.** The compression step was originally `LLMChainFilter`, and its evolution to a custom filter is its own story — see [Real Bugs](#real-bugs-found-and-fixed).
- **Grade aggregation: "any correct grade wins."** If even one retrieved document is graded `correct`, the overall grade is `correct` — one genuinely relevant document is treated as sufficient evidence to proceed, rather than requiring a majority or unanimous grade.
- **`QueryType` classification with different treatment per type.** `repo_specific` and `mixed` both loop through `refine` (there's still a repository-side query worth improving), while pure `general` queries skip straight to `web_search` — refining a query against the repo when the answer was never going to be in the repo just wastes a retry.
- **File-level citations, not line-level.** Citations point at `[Internal: path/to/file.py]`, not a specific line range. This is a deliberate scope choice, called out explicitly here rather than left as a silent gap — see [What's Next](#whats-next--what-id-do-differently) for the line-level alternative.
- **The all-web-sourced disclaimer fix.** Citation validity alone isn't sufficient for a trustworthy answer — an answer built entirely from web sources, with every citation technically correct, can still misleadingly imply the repository implements something it doesn't. See the last entry in [Real Bugs](#real-bugs-found-and-fixed).

---

## Real Bugs Found and Fixed

**`pathlib.Path.match()` doesn't support recursive `**` wildcards pre-3.13.**
`DirectoryLoader`'s `exclude` parameter silently failed to filter out nested `venv/` and `__pycache__/` directories — files inside them were loaded anyway, with no error. Fixed by dropping the `exclude` glob entirely and doing manual path-parts filtering instead: each loaded document's path is checked against a `SKIP_DIRS` set across every path component, not matched against a glob pattern.

**Chroma inserts weren't idempotent.**
Reindexing the same repo without stable IDs silently duplicated every chunk on each rerun — Chroma happily inserted a fresh set of auto-generated IDs every time. Fixed by switching to `sha256(resolved path) : per-file-index` chunk IDs. Verified by directly inspecting the Chroma collection's IDs before and after an unrelated file edit, and confirming chunk counts for untouched files stayed constant.

**`LLMChainFilter`'s `BooleanOutputParser` crashed / silently failed on Gemini's response shape.**
Gemini returns a list of content blocks rather than a plain string, and one of those blocks was a long, opaque base64 signature string that happened to contain both the substrings `"YES"` and `"NO"` — which broke `BooleanOutputParser`'s parsing in an unpredictable way. Root-caused with a raw, parser-free probe call that printed the actual response shape. Fixed by replacing the entire compressor: `StructuredRelevanceFilter`, a custom `BaseDocumentCompressor` that uses `with_structured_output(RelevanceDecision)` instead of parsing free text for yes/no.

**Guardrails metadata key typo (`"sources"` vs `"source"`).**
An early version of the guardrails wiring built its valid-sources set from the wrong metadata key, which silently produced an always-empty valid set — every citation looked invalid, all the time, with no exception raised. Found by printing and inspecting the actual `valid_sources` dict at runtime rather than assuming the wiring was correct.

**The "all-web-sourced" framing gap.**
Every individual pipeline stage was correct — retrieval, grading, citation, guardrails — but the composed system could still answer confidently about features that don't exist in the repo, using only real, valid citations, because nothing enforced answer-*level* "this isn't implemented here" framing. This wasn't caught by any single unit test; it only showed up during full end-to-end evaluation, when a query with zero repo-relevant evidence produced a fluent, fully-cited answer built entirely from web sources with no disclaimer that the repo itself doesn't do this. Fixed in `generator.py`: when every retrieved document is web-sourced, the system prompt is extended to force the answer to open with the standard "not enough information" disclaimer before answering generally from the web sources.

---

## Known Limitations

Documented on purpose, not hidden:

- **`general`-classified queries never get an internal retry.** They skip the refine/retry path entirely and go straight to web search — found via an anomaly-detection query that should have matched `anomaly_tool.py` in the repo but got classified `general` and routed away from it.
- **Near-empty chunks can still win an MMR diversity slot.** A markdown `---` divider or similarly low-information chunk can be selected purely for its diversity value. Compression usually filters it back out, but that's a cost paid downstream (an extra LLM relevance call) rather than avoided upstream.
- **Query classification isn't perfectly stable on borderline phrasing**, even at `temperature=0` — genuinely ambiguous queries can flip between `repo_specific` and `mixed` across runs.
- **Compression and grading run structurally similar LLM judgments back-to-back.** Both ask "is this document relevant / useful," just phrased differently, on the same documents. Noted as an optimization opportunity, not yet fixed.

---

## Evaluation Results

Evaluation runs against a fixed set of hand-written cases against the `ai-data-analyst` test repo, split into two groups:

- **RAGAS-scored cases** (`n=8`) — questions with a real reference answer. Scored on Faithfulness, Context Precision, and LLM Context Recall, averaged with the sample count reported alongside each mean rather than as a bare number.
- **Refusal-accuracy cases**, scored separately — questions that genuinely have no answer in the repo (e.g. "how do I set up OAuth"), where the correct behavior is a clean refusal. These are scored separately because most RAGAS metrics require a reference answer, and there is no correct reference for a question that has no correct answer.

Results are also broken down **per graph path** — sliced by which route actually fired (first-try vs. corrected, single vs. multi-retrieval, whether guardrails triggered a regeneration) — rather than reported as one aggregate score. A query that needed two retrieval attempts and a guardrails retry is a meaningfully different case from one that graded `correct` on the first pass, and averaging them together would hide that.

**Overall averages (n=8):**

| Metric | Score |
|---|---|
| Faithfulness | 0.884 |
| Context Precision | 0.839 |
| Context Recall | 0.723 |

Faithfulness and Context Precision are both comfortably high, indicating that generated answers stay grounded in what was retrieved, and that what gets retrieved is largely on-topic. Context Recall trailing behind the other two is the more informative number here — it suggests that even when the retrieved evidence is precise and the answer is faithful to it, the retriever isn't always surfacing every piece of relevant context that exists in the repo for a given query. That's consistent with the [Known Limitations](#known-limitations) noted above (e.g. `general`-classified queries never getting an internal retry), rather than a generation-quality problem.

Run `python -m evals.ragas_eval` to reproduce.

---

## Setup / How to Run

**Requirements:** see `requirements.txt`.

**Environment variables** (`.env`):

```
GOOGLE_API_KEY=...
TAVILY_API_KEY=...
```

**1. Build the index** against a target repository:

```bash
python -m indexing.vector_store /path/to/target/repo ./chroma_store
```

**2. Run the full graph** against a query:

```bash
python -m graph.build_graph
```

**3. Run the evaluation harness:**

```bash
python -m evals.ragas_eval
```

---

## Project Structure

```
.
├── ingestion/
│   ├── loader.py          # repo loading + directory filtering
│   └── splitter.py        # language-aware chunking
├── indexing/
│   └── vector_store.py    # Chroma build/load, stable chunk IDs
├── retrieval/
│   └── mmr_retriever.py   # AsymmetricMMRRetriever + StructuredRelevanceFilter
├── grading/
│   ├── grader.py          # per-document grading + aggregation
│   └── correction.py      # classify_query, refine_query, web_search_fallback
├── reranker/
│   └── reranker.py        # cross-encoder reranking
├── generator/
│   └── generator.py       # grounded, cited answer generation
├── guardrail/
│   └── validator.py       # ValidCitations validator
├── graph/
│   └── build_graph.py     # LangGraph StateGraph wiring + routing
└── evals/
    └── ragas_eval.py      # RAGAS + refusal-accuracy evaluation harness
```

---

## What's Next / What I'd Do Differently

- **Line-level citations** instead of file-level, for more precise "go look here" pointers.
- **Give `general` one internal retry** before routing away from the repo entirely, to catch cases like the anomaly-detection query noted in [Known Limitations](#known-limitations).
- **Deduplicate the compression and grading LLM calls** — both are asking a structurally similar relevance question on the same documents.
- **Migrate off the deprecated `guardrails-ai` / `ragas` import paths** flagged in the project's own terminal output during development.
