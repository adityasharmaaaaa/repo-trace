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
      .block-container { padding-top: 2rem; max-width: 1200px; }
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


def how_tab() -> None:
    left, right = st.columns([3, 2])
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


# ------------------------------------------------------------------ main


def main() -> None:
    repo = sidebar()
    st.title("repo-trace")
    st.caption("Ask questions about any public GitHub repository and see how the "
               "corrective RAG pipeline grades, corrects and verifies its own retrieval.")

    missing = missing_keys()
    if missing:
        st.error(f"Missing environment variables: {', '.join(missing)}. Add them to `.env` and restart.")
        st.stop()

    tab_ask, tab_how = st.tabs(["Ask", "How it works"])
    with tab_ask:
        ask_tab(repo)
    with tab_how:
        how_tab()


main()
