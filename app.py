"""repo-trace demo UI.

Run with:  streamlit run app.py
"""
import importlib.util
import os
import re
import sys
import time
from typing import Dict, List, Optional

import streamlit as st

_REQUIRED = ["langchain_groq", "langchain_chroma", "langchain_huggingface", "langgraph", "guardrails"]
_missing_pkgs = [m for m in _REQUIRED if importlib.util.find_spec(m) is None]
if _missing_pkgs:
    st.error(
        f"Missing packages in this Python environment ({sys.executable}): "
        f"{', '.join(_missing_pkgs)}.\n\n"
        "Start the app from the project's virtualenv:\n\n"
        "```\nsource .venv/bin/activate && streamlit run app.py\n```\n\n"
        "or install the dependencies: `pip install -r requirements.txt`."
    )
    st.stop()

from utils.models import EMBEDDING_PROVIDER, HF_EMBEDDING_MODEL, GOOGLE_EMBEDDING_MODEL, get_model_name
from workspace.repos import RepoError, RepoRecord, add_repository, delete_repo, get_repo, list_repos

st.set_page_config(page_title="repo-trace", page_icon="🔎", layout="wide")

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
      html, body, .stApp, .stMarkdown, button, input, textarea { font-family: 'Inter', system-ui, sans-serif; }
      [data-testid="stIconMaterial"] { font-family: 'Material Symbols Rounded' !important; }
      code, pre, .chip, .step b { font-family: 'JetBrains Mono', ui-monospace, monospace !important; }
      .block-container { padding-top: 2rem; max-width: 1200px; }
      h1, h2, h3 { letter-spacing: -0.01em; }

      /* ---------- overview ---------- */
      .rt-hero { padding: 2.5rem 0 1.5rem; border-bottom: 1px solid rgba(127,127,127,.18); margin-bottom: 2.25rem; }
      .rt-eyebrow { font-size: .72rem; letter-spacing: .18em; text-transform: uppercase; opacity: .6; font-weight: 600; }
      .rt-title { font-family: 'Instrument Serif', Georgia, serif; font-weight: 400; font-size: clamp(2.3rem, 5vw, 3.6rem);
                  line-height: 1.05; margin: .9rem 0 1.1rem; letter-spacing: -0.015em; }
      .rt-title em { color: #4F5BD5; font-style: italic; }
      .rt-lede { font-size: 1.08rem; line-height: 1.65; opacity: .78; max-width: 62ch; }
      .rt-stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 0;
                  margin-top: 2rem; border-top: 1px solid rgba(127,127,127,.18); }
      .rt-stat { padding: 1.1rem 1.2rem 0 0; }
      .rt-stat .v { font-family: 'Instrument Serif', Georgia, serif; font-size: 2.3rem; line-height: 1; }
      .rt-stat .l { font-size: .78rem; opacity: .6; margin-top: .4rem; letter-spacing: .02em; }
      .rt-section { margin: 2.75rem 0 1rem; }
      .rt-kicker { font-family: 'JetBrains Mono', monospace; font-size: .72rem; opacity: .5; letter-spacing: .08em; }
      .rt-h2 { font-family: 'Instrument Serif', Georgia, serif; font-weight: 400; font-size: 2rem; margin: .2rem 0 1.2rem; }
      .rt-grid { display: grid; gap: 1rem; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); }
      .rt-grid.four { grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); }
      .rt-card { border: 1px solid rgba(127,127,127,.2); border-radius: 10px; padding: 1.2rem 1.3rem;
                 background: rgba(127,127,127,.03); }
      .rt-card h4 { margin: 0 0 .45rem; font-size: 1rem; font-weight: 600; padding: 0; }
      .rt-card p { margin: 0; font-size: .92rem; line-height: 1.6; opacity: .78; }
      .rt-card .n { font-family: 'JetBrains Mono', monospace; font-size: .75rem; color: #4F5BD5; margin-bottom: .6rem; }
      .rt-card.muted { background: transparent; }
      .rt-card.accent { border-color: rgba(79,91,213,.45); background: rgba(79,91,213,.06); }
      .rt-tags { display: flex; flex-wrap: wrap; gap: .45rem; }
      .rt-tag { border: 1px solid rgba(127,127,127,.25); border-radius: 999px; padding: .25rem .75rem; font-size: .82rem; }
      .rt-tag span { opacity: .55; margin-right: .35rem; }
      .rt-foot { margin-top: 3rem; padding-top: 1.2rem; border-top: 1px solid rgba(127,127,127,.18);
                 font-size: .85rem; opacity: .6; }
      .rt-table { width: 100%; border-collapse: collapse; font-size: .92rem; }
      .rt-table td { padding: .7rem .4rem; border-bottom: 1px solid rgba(127,127,127,.15); vertical-align: top; }
      .rt-table td:first-child { font-family: 'JetBrains Mono', monospace; font-size: .85rem; white-space: nowrap; width: 1%; padding-right: 1.5rem; }
      .rt-table td:last-child { opacity: .8; }

      /* ---------- trace ---------- */
      .step { padding: .55rem .8rem; border-left: 3px solid var(--accent, #6b7280);
              margin: .35rem 0; border-radius: 4px; background: rgba(127,127,127,.06); }
      .step b { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
      .step .detail { opacity: .8; font-size: .9rem; }
      .chip { display: inline-block; padding: .1rem .55rem; border-radius: 999px;
              font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .8rem;
              background: rgba(127,127,127,.14); margin: .1rem 0; }
      .arrow { opacity: .5; margin: 0 .25rem; }
      .grade-correct { color: #16a34a; } .grade-ambiguous { color: #d97706; }
      .grade-incorrect { color: #dc2626; }
    </style>
    """,
    unsafe_allow_html=True,
)

NODE_COLORS = {
    "retrieve": "#2563eb", "grade": "#7c3aed", "classify": "#d97706", "refine": "#d97706",
    "web_search": "#0891b2", "rerank": "#4f46e5", "generate": "#16a34a", "guardrails": "#dc2626",
}

EXAMPLE_QUESTIONS = [
    "What does this project do, and how is the code organised?",
    "Where is the main entry point and what happens when it runs?",
    "How do I configure Redis for distributed rate limiting?",  # out of scope → web fallback
]


# ------------------------------------------------------------------ resources


@st.cache_resource(show_spinner="Loading pipeline (models, index)…")
def load_graph(index_dir: str):
    from graph.build_graph import build_graph  # heavy: loads reranker, guardrails

    return build_graph(index_dir)


def missing_keys() -> List[str]:
    required = ["GROQ_API_KEY", "TAVILY_API_KEY"]
    if EMBEDDING_PROVIDER == "google":
        required.append("GOOGLE_API_KEY")
    return [k for k in required if not os.getenv(k)]


# ------------------------------------------------------------------ helpers


def linkify_citations(answer: str, repo: RepoRecord) -> str:
    """Turn [Internal: path] / [Web: url] citations into clickable links."""

    def internal(m: re.Match) -> str:
        path = m.group(1).strip()
        return f"[`{path}`]({repo.file_url(path)})"

    def web(m: re.Match) -> str:
        url = m.group(1).strip()
        host = re.sub(r"^https?://(www\.)?", "", url).split("/")[0]
        return f"[🌐 {host}]({url})"

    answer = re.sub(r"\[Internal:\s*([^\]]+)\]", internal, answer)
    return re.sub(r"\[Web:\s*([^\]]+)\]", web, answer)


def source_label(doc) -> str:
    src = doc.metadata.get("source", "unknown")
    return f"🌐 {src}" if doc.metadata.get("file_type") == "web" else f"📄 {src}"


def step_html(node: str, title: str, detail: str = "") -> str:
    color = NODE_COLORS.get(node, "#6b7280")
    detail_html = f"<div class='detail'>{detail}</div>" if detail else ""
    return f"<div class='step' style='--accent:{color}'><b>{node}</b> · {title}{detail_html}</div>"


def describe_step(node: str, update: Dict, state: Dict, prev_docs: List) -> str:
    """Human-readable summary of one node's output, shown live during a run."""
    if node == "retrieve":
        docs = update.get("documents", [])
        files = sorted({d.metadata.get("source") for d in docs})
        detail = "<br>".join(f"<code>{f}</code>" for f in files) or "nothing relevant survived compression"
        return step_html(
            node,
            f"attempt {update.get('retrieval_retry_count')} — {len(docs)} chunk(s) kept "
            f"for “{state['current_query']}”",
            detail,
        )
    if node == "grade":
        overall = update.get("overall_grade")
        rows = [
            f"<span class='grade-{g.grade}'>{g.grade}</span> — <code>{d.metadata.get('source')}</code>"
            for d, g in zip(prev_docs, update.get("grades", []))
        ]
        return step_html(node, f"overall: <span class='grade-{overall}'><b>{overall}</b></span>", "<br>".join(rows))
    if node == "classify":
        qt = update["query_type"].query_type
        nxt = "refine query and retry the repo" if qt in ("repo_specific", "mixed") else "go to web search"
        return step_html(node, f"query type: <b>{qt}</b> → {nxt}")
    if node == "refine":
        return step_html(node, f"retry {update.get('correction_retry_count')}: rewrote query",
                         f"“{update.get('current_query')}”")
    if node == "web_search":
        web = [d for d in update.get("documents", []) if d.metadata.get("file_type") == "web"]
        return step_html(node, f"{len(web)} web result(s)",
                         "<br>".join(f"<code>{d.metadata.get('source')}</code>" for d in web))
    if node == "rerank":
        docs = update.get("documents", [])
        rows = [f"{d.metadata.get('rerank_score', 0):+.2f} — <code>{d.metadata.get('source')}</code>" for d in docs]
        return step_html(node, f"cross-encoder kept top {len(docs)}", "<br>".join(rows))
    if node == "generate":
        return step_html(node, f"answer drafted ({len(update.get('answer') or '')} chars)")
    if node == "guardrails":
        if update.get("guardrails_passed"):
            return step_html(node, "✅ every citation points at evidence from this run")
        return step_html(node, "❌ invalid citation(s) — " + "; ".join(update.get("guardrails_errors", [])))
    return step_html(node, "done")


def run_pipeline(graph, question: str, trace_box) -> Dict:
    from graph.build_graph import initial_state

    state = initial_state(question)
    path: List[str] = []
    prev_docs: List = []
    started = time.perf_counter()

    for chunk in graph.stream(state, stream_mode="updates"):
        for node, update in chunk.items():
            update = update or {}
            state.update(update)
            path.append(node)
            trace_box.markdown(describe_step(node, update, state, prev_docs), unsafe_allow_html=True)
            if "documents" in update:
                prev_docs = update["documents"]

    return {"question": question, "state": state, "path": path,
            "seconds": time.perf_counter() - started}


# ------------------------------------------------------------------ rendering


def render_path(path: List[str]) -> None:
    chips = "<span class='arrow'>→</span>".join(f"<span class='chip'>{n}</span>" for n in path)
    st.markdown(chips, unsafe_allow_html=True)


def render_result(run: Dict, repo: RepoRecord, show_trace: bool = True) -> None:
    s = run["state"]
    qt = s["query_type"].query_type if s.get("query_type") else "—"
    cols = st.columns(6)
    cols[0].metric("Retrieval grade", s.get("overall_grade") or "—")
    cols[1].metric("Query type", qt)
    cols[2].metric("Retrievals", s.get("retrieval_retry_count", 0))
    cols[3].metric("Corrections", s.get("correction_retry_count", 0))
    cols[4].metric("Citations", "valid" if s.get("guardrails_passed") else "flagged")
    cols[5].metric("Time", f"{run['seconds']:.1f}s")

    st.caption("Route taken through the graph")
    render_path(run["path"])

    st.markdown("#### Answer")
    with st.container(border=True):
        st.markdown(linkify_citations(s.get("answer") or "_No answer produced._", repo))
    if not s.get("guardrails_passed") and s.get("guardrails_errors"):
        st.warning("Guardrails flagged: " + "; ".join(s["guardrails_errors"]))

    docs = s.get("documents", [])
    st.markdown(f"#### Evidence used ({len(docs)})")
    for i, doc in enumerate(docs, 1):
        score = doc.metadata.get("rerank_score")
        label = f"{i}. {source_label(doc)}" + (f"  ·  rerank {score:+.2f}" if score is not None else "")
        with st.expander(label):
            if doc.metadata.get("file_type") == "web":
                st.markdown(f"[Open source]({doc.metadata.get('source')})")
                st.write(doc.page_content)
            else:
                st.markdown(f"[View on GitHub]({repo.file_url(doc.metadata.get('source', ''))})")
                lang = "python" if doc.metadata.get("file_type") == "code" else "markdown"
                st.code(doc.page_content, language=lang)


# ------------------------------------------------------------------ sidebar


def sidebar() -> Optional[RepoRecord]:
    st.sidebar.title("🔎 repo-trace")
    st.sidebar.caption("Corrective RAG for understanding unfamiliar codebases")

    st.sidebar.subheader("Add a repository")
    with st.sidebar.form("add_repo", clear_on_submit=True):
        url = st.text_input("GitHub URL", placeholder="https://github.com/owner/repo")
        submitted = st.form_submit_button("Clone & index", width="stretch", type="primary")
    if submitted and url:
        with st.sidebar.status("Indexing repository…", expanded=True) as status:
            try:
                record = add_repository(url, progress=status.write)
                load_graph.clear()
                st.session_state["active_repo"] = record.repo_id
                status.update(label=f"Indexed {record.display_name}", state="complete", expanded=False)
            except RepoError as e:
                status.update(label="Could not add repository", state="error")
                st.sidebar.error(str(e))
            except Exception as e:  # noqa: BLE001 — surface anything unexpected in the UI
                status.update(label="Indexing failed", state="error")
                st.sidebar.exception(e)
    st.sidebar.caption("Public repos only · indexes `.py` and `.md` files")

    repos = list_repos()
    if not repos:
        return None

    st.sidebar.subheader("Indexed repositories")
    ids = [r.repo_id for r in repos]
    active = st.session_state.get("active_repo")
    choice = st.sidebar.radio(
        "Active repository",
        ids,
        index=ids.index(active) if active in ids else 0,
        format_func=lambda rid: get_repo(rid).display_name,
        label_visibility="collapsed",
    )
    st.session_state["active_repo"] = choice
    repo = get_repo(choice)

    with st.sidebar.container(border=True):
        st.markdown(f"**[{repo.display_name}]({repo.url})**")
        st.caption(
            f"{repo.num_files} files · {repo.num_chunks} chunks · "
            f"commit `{repo.short_commit}`" + (f" · branch `{repo.branch}`" if repo.branch else "")
        )
        c1, c2 = st.columns(2)
        if c1.button("Re-index", width="stretch"):
            with st.spinner("Re-indexing…"):
                try:
                    add_repository(repo.url)
                    load_graph.clear()
                except RepoError as e:
                    st.error(str(e))
            st.rerun()
        if c2.button("Remove", width="stretch"):
            load_graph.clear()
            delete_repo(repo.repo_id)
            st.session_state.pop("active_repo", None)
            st.rerun()

    with st.sidebar.expander("Models"):
        emb = HF_EMBEDDING_MODEL if EMBEDDING_PROVIDER != "google" else GOOGLE_EMBEDDING_MODEL
        st.markdown(
            f"- Generation & grading: `{get_model_name('generation')}` (Groq)\n"
            f"- Embeddings: `{emb}`\n"
            f"- Reranker: `ms-marco-MiniLM-L-6-v2`\n"
            f"- Web search: Tavily"
        )
    return repo


# ------------------------------------------------------------------ pages


def ask_tab(repo: Optional[RepoRecord]) -> None:
    if repo is None:
        st.info("Add a public GitHub repository in the sidebar to get started.")
        return

    st.markdown(f"Ask anything about **{repo.display_name}**.")
    runs = st.session_state.setdefault("runs", {}).setdefault(repo.repo_id, [])

    cols = st.columns(len(EXAMPLE_QUESTIONS))
    clicked = None
    for col, q in zip(cols, EXAMPLE_QUESTIONS):
        if col.button(q, width="stretch"):
            clicked = q

    with st.form("ask", clear_on_submit=False):
        question = st.text_area("Question", value=clicked or "", height=80,
                                placeholder="e.g. How does authentication work?")
        go = st.form_submit_button("Run pipeline", type="primary")

    question = clicked or (question.strip() if go else "")
    if question:
        graph = load_graph(repo.index_dir)
        with st.status("Running corrective RAG pipeline…", expanded=True) as status:
            trace = st.container()
            try:
                run = run_pipeline(graph, question, trace)
                status.update(label=f"Pipeline finished · {len(run['path'])} steps · {run['seconds']:.1f}s",
                              state="complete", expanded=False)
                runs.insert(0, run)
            except Exception as e:  # noqa: BLE001
                status.update(label="Pipeline failed", state="error")
                st.exception(e)

    for i, run in enumerate(runs):
        if i == 0:
            st.divider()
            st.markdown(f"### {run['question']}")
            render_result(run, repo)
        else:
            with st.expander(f"Earlier: {run['question']}"):
                render_result(run, repo)


GRAPH_DOT = """
digraph G {
  rankdir=TB; bgcolor="transparent";
  node [shape=box, style="rounded,filled", fillcolor="#eef2ff", color="#6366f1", fontname="Helvetica"];
  edge [fontname="Helvetica", fontsize=10, color="#64748b", fontcolor="#475569"];
  start [shape=circle, label="", width=.2, fillcolor="#334155"];
  end [shape=doublecircle, label="", width=.2, fillcolor="#334155"];
  start -> retrieve -> grade;
  grade -> rerank [label="correct"];
  grade -> web_search [label="correct + mixed"];
  grade -> classify [label="ambiguous / incorrect"];
  classify -> refine [label="repo_specific / mixed"];
  classify -> web_search [label="general"];
  refine -> retrieve [label="retries left"];
  refine -> rerank [label="cap hit"];
  refine -> web_search [label="cap hit + mixed"];
  web_search -> rerank -> generate -> guardrails;
  guardrails -> generate [label="bad citation, retries left"];
  guardrails -> end [label="pass / cap hit"];
}
"""


def html(markup: str) -> None:
    """Render raw HTML; lines are stripped so Markdown doesn't treat indentation as code."""
    st.markdown("\n".join(line.strip() for line in markup.splitlines() if line.strip()),
                unsafe_allow_html=True)


def section(kicker: str, title: str) -> None:
    html(f"<div class='rt-section'><div class='rt-kicker'>{kicker}</div><div class='rt-h2'>{title}</div></div>")


def cards(items, numbered: bool = False, cls: str = "", grid: str = "") -> None:
    body = "".join(
        f"<div class='rt-card {cls}'>"
        + (f"<div class='n'>{i:02d}</div>" if numbered else "")
        + f"<h4>{title}</h4><p>{text}</p></div>"
        for i, (title, text) in enumerate(items, 1)
    )
    html(f"<div class='rt-grid {grid}'>{body}</div>")


def overview_tab() -> None:
    html(
        """
        <div class="rt-hero">
          <div class="rt-eyebrow">Corrective RAG &nbsp;·&nbsp; LangGraph &nbsp;·&nbsp; Codebase intelligence</div>
          <div class="rt-title">Answers about code that<br><em>know when they don't know.</em></div>
          <div class="rt-lede">
            repo-trace is a question-answering assistant for unfamiliar GitHub repositories.
            Unlike plain retrieval-augmented generation, it grades the evidence it retrieves,
            corrects course when that evidence is weak, and verifies every citation before an
            answer is shown.
          </div>
          <div class="rt-stats">
            <div class="rt-stat"><div class="v">0.884</div><div class="l">Faithfulness · RAGAS</div></div>
            <div class="rt-stat"><div class="v">0.839</div><div class="l">Context precision · RAGAS</div></div>
            <div class="rt-stat"><div class="v">0.723</div><div class="l">Context recall · RAGAS</div></div>
            <div class="rt-stat"><div class="v">8</div><div class="l">Graph nodes · 2 retry loops</div></div>
          </div>
        </div>
        """
    )

    section("01 — MOTIVATION", "Why it exists")
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        cards([("The problem",
                "Onboarding to a new codebase is slow, and most RAG demos make it subtly worse: "
                "they answer confidently even when retrieval was weak. Three barely-related chunks "
                "still get stuffed into a prompt, and the model synthesises a plausible, wrong answer.")],
              cls="muted")
    with c2:
        cards([("The approach",
                "<b>Grade retrieval before trusting it.</b> Every retrieved chunk is judged "
                "<i>correct</i>, <i>ambiguous</i> or <i>incorrect</i>, and that verdict drives a "
                "specific corrective action — refine the query, fall back to the web, or proceed — "
                "rather than blindly retrying the same search.")],
              cls="accent")

    section("02 — PIPELINE", "The life of a question")
    cards([
        ("Retrieve",
         "Language-aware chunks, asymmetric query/document embeddings, diversity-tuned MMR, "
         "then an LLM relevance filter that drops off-topic chunks."),
        ("Grade",
         "Each surviving chunk is graded against the question. One genuinely relevant chunk "
         "is enough to proceed; otherwise the pipeline corrects itself."),
        ("Correct",
         "The question is classified as repo-specific, general or mixed. Repo questions are "
         "rewritten with vocabulary from the code; general ones go to live web search."),
        ("Answer & verify",
         "A cross-encoder reranks the evidence, the answer is generated with inline citations, "
         "and a guardrail rejects any citation not backed by this run's evidence."),
    ], numbered=True, grid="four")

    section("03 — ENGINEERING", "Decisions that matter")
    cards([
        ("Idempotent indexing",
         "Chunk IDs are <code>sha256(path):index</code> per file, so re-indexing never "
         "duplicates data and editing one file never shifts another's IDs."),
        ("Two independent retry budgets",
         "Retrieval corrections and citation regenerations are capped separately, so one "
         "failure mode can't exhaust the budget meant for the other."),
        ("Structured relevance filtering",
         "A custom compressor with typed structured output replaced a free-text YES/NO parser "
         "that silently broke on real model responses."),
        ("Honest out-of-scope answers",
         "When every piece of evidence comes from the web, the answer must open by saying "
         "the repository itself doesn't cover the topic."),
        ("Evaluated per route",
         "RAGAS scores are sliced by the path each query took through the graph, with "
         "refusal accuracy scored separately for unanswerable questions."),
        ("Live, inspectable runs",
         "Every graph step streams to the UI — grades, rewrites, web results, rerank scores "
         "and the guardrail verdict — with citations linked to the exact commit."),
    ])

    section("04 — STACK", "Built with")
    stack = [
        ("Orchestration", "LangGraph"), ("LLM", "Groq · gpt-oss-120b"), ("Embeddings", "BGE-small-en-v1.5"),
        ("Vector store", "Chroma"), ("Retrieval", "MMR + compression"), ("Reranker", "ms-marco MiniLM cross-encoder"),
        ("Web fallback", "Tavily"), ("Guardrails", "guardrails-ai"), ("Evaluation", "RAGAS"), ("Interface", "Streamlit"),
    ]
    html("<div class='rt-tags'>" + "".join(f"<div class='rt-tag'><span>{k}</span>{v}</div>" for k, v in stack) + "</div>")

    html(
        """
        <div class="rt-foot">
          Open the <b>Ask</b> tab, add any public GitHub repository from the sidebar, and watch
          the pipeline reason about it step by step.
        </div>
        """
    )


NODES = [
    ("retrieve", "MMR + contextual-compression retrieval against the current query."),
    ("grade", "Grades every chunk correct / ambiguous / incorrect and aggregates an overall grade."),
    ("classify", "Labels the query repo_specific, general or mixed — reached only when retrieval fell short."),
    ("refine", "Rewrites the query using vocabulary from what was retrieved, or broadens it if nothing was."),
    ("web_search", "Live Tavily search for general-knowledge evidence."),
    ("rerank", "Cross-encoder reranking of the accumulated evidence pool."),
    ("generate", "Grounded answer with inline [Internal: …] / [Web: …] citations."),
    ("guardrails", "Rejects any citation not present in this run's evidence; triggers regeneration."),
]


def how_tab() -> None:
    section("ARCHITECTURE", "A self-correcting graph")
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.graphviz_chart(GRAPH_DOT, width="stretch")
    with right:
        st.markdown(
            """
**Grade retrieval before trusting it.** Plain RAG stuffs whatever the vector search
returns into the prompt. repo-trace grades every retrieved chunk as `correct`,
`ambiguous` or `incorrect` against the question, and the grade decides what happens next:

- **correct** → rerank and answer.
- **not correct** → classify the question:
  - `repo_specific` / `mixed` → **refine** the query using vocabulary from what was
    retrieved, then retrieve again (max 2 corrections).
  - `general` → fall back to **web search**.
- **guardrails** check that every `[Internal: …]` / `[Web: …]` citation in the answer
  points at evidence actually retrieved in this run; otherwise regenerate (max 2).

If all evidence is from the web, the answer must open by saying the repository itself
doesn't cover the topic.

**Retrieval stack:** language-aware chunking → asymmetric embeddings → MMR
(`k=5, fetch_k=15, λ=0.3`) → LLM relevance filter → cross-encoder rerank.
"""
        )

    section("NODES", "What each step does")
    html("<table class='rt-table'>" + "".join(f"<tr><td>{n}</td><td>{d}</td></tr>" for n, d in NODES) + "</table>")


def eval_tab() -> None:
    section("EVALUATION", "Measured, not assumed")
    cols = st.columns(3)
    cols[0].metric("Faithfulness", "0.884", help="Are answer claims supported by the retrieved context?")
    cols[1].metric("Context precision", "0.839", help="Is the retrieved context on-topic?")
    cols[2].metric("Context recall", "0.723", help="Was all the relevant context retrieved?")
    st.caption("RAGAS, n = 8 hand-written cases with reference answers, run against an independently built "
               "codebase (an AI-powered CSV data-analyst app). Reproduce with `python -m evals.ragas_eval`.")

    section("METHOD", "How it was evaluated")
    cards([
        ("Reference-answer cases",
         "Scored on faithfulness, context precision and LLM context recall, reported with the "
         "sample count alongside every mean."),
        ("Refusal accuracy",
         "Questions with no answer in the repo (e.g. OAuth setup) are scored separately — the "
         "correct behaviour is a clean refusal, and there is no reference answer to compare."),
        ("Per-route breakdown",
         "Results are sliced by the path taken: first-try vs. corrected, single vs. multi-retrieval, "
         "and whether guardrails forced a regeneration."),
    ])

    section("READING THE NUMBERS", "What they say")
    st.markdown(
        "High faithfulness and precision mean answers stay grounded and retrieved context is on-topic. "
        "Recall trailing behind is the more informative signal: even when evidence is precise, the "
        "retriever doesn't always surface *every* relevant chunk — consistent with the limitations below "
        "rather than a generation-quality problem."
    )

    section("LIMITATIONS & NEXT STEPS", "Documented on purpose")
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        cards([
            ("General queries skip internal retries",
             "A query misclassified as general is routed to the web without another repo attempt."),
            ("Low-information chunks can win MMR slots",
             "A markdown divider may be picked for diversity; compression removes it downstream, at a cost."),
            ("Borderline classification can flip",
             "Even at temperature 0, ambiguous phrasing can alternate between repo_specific and mixed."),
        ], cls="muted")
    with c2:
        cards([
            ("Line-level citations",
             "Point at exact line ranges instead of whole files."),
            ("One internal retry for general queries",
             "Try the repo once more before routing away from it entirely."),
            ("Merge compression and grading",
             "Both ask a similar relevance question of the same chunks — one LLM pass could do both."),
        ], cls="accent")


# ------------------------------------------------------------------ main


def main() -> None:
    repo = sidebar()

    tab_overview, tab_ask, tab_how, tab_eval = st.tabs(["Overview", "Ask", "How it works", "Evaluation"])
    with tab_overview:
        overview_tab()
    with tab_ask:
        missing = missing_keys()
        if missing:
            st.error(f"Missing environment variables: {', '.join(missing)}. Add them to `.env` and restart.")
        else:
            ask_tab(repo)
    with tab_how:
        how_tab()
    with tab_eval:
        eval_tab()


main()
