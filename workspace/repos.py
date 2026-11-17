"""Add GitHub repositories to repo-trace: clone, index, and keep a registry.

Layout under DATA_DIR (default ./data, git-ignored):
    repos/<owner>__<name>/      shallow clone of the repository
    indexes/<owner>__<name>/    Chroma index built from that clone
    repos.json                  registry of indexed repositories
"""
import json
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

from ingestion.loader import load_repository
from ingestion.splitter import split_documents

DATA_DIR = Path(os.getenv("REPO_TRACE_DATA_DIR", "./data"))
REPOS_DIR = DATA_DIR / "repos"
INDEXES_DIR = DATA_DIR / "indexes"
REGISTRY_PATH = DATA_DIR / "repos.json"

# Keeps indexing time reasonable for a live demo; override via env if needed.
MAX_FILES = int(os.getenv("REPO_TRACE_MAX_FILES", "400"))
CLONE_TIMEOUT_S = 180

# owner/name rules follow GitHub's: alphanumerics, '-', '_', '.'.
_GITHUB_URL = re.compile(
    r"^(?:https?://)?(?:www\.)?github\.com/"
    r"(?P<owner>[A-Za-z0-9](?:[A-Za-z0-9-]{0,38}))/"
    r"(?P<name>[A-Za-z0-9._-]{1,100}?)"
    r"(?:\.git)?"
    r"(?:/tree/(?P<branch>[A-Za-z0-9._/-]{1,200}))?"
    r"/?$"
)


class RepoError(Exception):
    """User-facing error while adding or indexing a repository."""


@dataclass
class RepoRecord:
    repo_id: str
    url: str
    owner: str
    name: str
    branch: Optional[str]
    commit: str
    num_files: int
    num_chunks: int
    indexed_at: str
    index_dir: str

    @property
    def display_name(self) -> str:
        return f"{self.owner}/{self.name}"

    @property
    def short_commit(self) -> str:
        return self.commit[:7]

    def file_url(self, path: str) -> str:
        return f"https://github.com/{self.owner}/{self.name}/blob/{self.commit}/{path}"

    @property
    def clone_dir(self) -> str:
        return str(REPOS_DIR / self.repo_id)


def parse_github_url(url: str) -> Dict[str, Optional[str]]:
    match = _GITHUB_URL.match(url.strip())
    if not match:
        raise RepoError(
            "Not a valid GitHub repository URL. "
            "Expected something like https://github.com/owner/repo"
        )
    parts = match.groupdict()
    if parts["name"] in {".", ".."}:
        raise RepoError("Invalid repository name.")
    return parts


def make_repo_id(owner: str, name: str) -> str:
    return f"{owner}__{name}".lower()


# ------------------------------------------------------------- registry


def list_repos() -> List[RepoRecord]:
    if not REGISTRY_PATH.exists():
        return []
    raw = json.loads(REGISTRY_PATH.read_text())
    records = [RepoRecord(**r) for r in raw]
    return sorted(records, key=lambda r: r.indexed_at, reverse=True)


def get_repo(repo_id: str) -> Optional[RepoRecord]:
    return next((r for r in list_repos() if r.repo_id == repo_id), None)


def _save_registry(records: List[RepoRecord]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = REGISTRY_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps([asdict(r) for r in records], indent=2))
    tmp.replace(REGISTRY_PATH)


def _upsert(record: RepoRecord) -> None:
    records = [r for r in list_repos() if r.repo_id != record.repo_id]
    _save_registry(records + [record])


def delete_repo(repo_id: str) -> None:
    shutil.rmtree(REPOS_DIR / repo_id, ignore_errors=True)
    shutil.rmtree(INDEXES_DIR / repo_id, ignore_errors=True)
    _save_registry([r for r in list_repos() if r.repo_id != repo_id])


# ------------------------------------------------------------- clone + index


def _git(*args: str, cwd: Optional[Path] = None) -> str:
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}  # never hang on auth prompts
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=CLONE_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        raise RepoError("Timed out cloning the repository.")
    if result.returncode != 0:
        stderr = result.stderr.strip()
        if "not found" in stderr.lower() or "could not read username" in stderr.lower():
            raise RepoError("Repository not found or private (only public repos are supported).")
        if "remote branch" in stderr.lower():
            raise RepoError("Branch not found in that repository.")
        raise RepoError(f"git failed: {stderr.splitlines()[-1] if stderr else 'unknown error'}")
    return result.stdout.strip()


def add_repository(
    url: str,
    progress: Callable[[str], None] = lambda msg: None,
) -> RepoRecord:
    """Clone (or re-clone) a public GitHub repo and build its index."""
    from indexing.vector_store import build_vector_store  # heavy import, load lazily

    parts = parse_github_url(url)
    owner, name, branch = parts["owner"], parts["name"], parts["branch"]
    repo_id = make_repo_id(owner, name)
    clone_dir = REPOS_DIR / repo_id
    # A fresh, versioned index dir per build: no stale chunks from deleted files,
    # and no clash with a Chroma client a running app still holds on the old dir.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    index_dir = INDEXES_DIR / repo_id / stamp
    canonical_url = f"https://github.com/{owner}/{name}"
    previous = get_repo(repo_id)

    shutil.rmtree(clone_dir, ignore_errors=True)
    REPOS_DIR.mkdir(parents=True, exist_ok=True)
    index_dir.parent.mkdir(parents=True, exist_ok=True)

    try:
        progress(f"Cloning {owner}/{name}" + (f" @ {branch}" if branch else "") + "…")
        clone_args = ["clone", "--depth", "1", "--single-branch"]
        if branch:
            clone_args += ["--branch", branch]
        _git(*clone_args, "--", f"{canonical_url}.git", str(clone_dir))
        commit = _git("rev-parse", "HEAD", cwd=clone_dir)

        progress("Loading Python and Markdown files…")
        docs = load_repository(str(clone_dir))
        if not docs:
            raise RepoError(
                "No Python (.py) or Markdown (.md) files found — "
                "repo-trace currently indexes only those file types."
            )
        if len(docs) > MAX_FILES:
            raise RepoError(
                f"Repository has {len(docs)} indexable files; the limit is {MAX_FILES} "
                "(set REPO_TRACE_MAX_FILES to raise it)."
            )

        progress(f"Splitting {len(docs)} files into chunks…")
        chunks = split_documents(docs)

        progress(f"Embedding {len(chunks)} chunks…")
        vectorstore = build_vector_store(chunks, str(index_dir))
        num_chunks = vectorstore._collection.count()
    except Exception:
        shutil.rmtree(clone_dir, ignore_errors=True)
        shutil.rmtree(index_dir, ignore_errors=True)
        raise

    record = RepoRecord(
        repo_id=repo_id,
        url=canonical_url + (f"/tree/{branch}" if branch else ""),
        owner=owner,
        name=name,
        branch=branch,
        commit=commit,
        num_files=len(docs),
        num_chunks=num_chunks,
        indexed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        index_dir=str(index_dir),
    )
    _upsert(record)
    if previous and previous.index_dir != record.index_dir:
        shutil.rmtree(previous.index_dir, ignore_errors=True)
    progress("Done.")
    return record


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("usage: python -m workspace.repos <github-url>")
        sys.exit(1)
    rec = add_repository(sys.argv[1], progress=print)
    print(rec)
