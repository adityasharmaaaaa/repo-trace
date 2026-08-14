from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, TextLoader


SKIP_DIRS = {".git", "__pycache__", "venv", ".venv", "node_modules", "build", "dist"}


def load_repository(repo_path: str) -> List[Document]:
    repo_path = Path(repo_path).resolve()
    directory_loader_py=DirectoryLoader(
        repo_path,
        glob="**/*.py",
        loader_cls=TextLoader,
        recursive=True,
    )

    directory_loader_md=DirectoryLoader(
        repo_path,
        glob="**/*.md",
        loader_cls=TextLoader,
        recursive=True,
    )

    docs_py = [
        doc for doc in directory_loader_py.load() if doc.page_content.strip()
    ]
    docs_md = [
        doc for doc in directory_loader_md.load() if doc.page_content.strip()
    ]


    filtered_docs = []
    for doc in docs_py + docs_md:
        source_path = Path(doc.metadata["source"]).resolve()
        
        try:
            rel_path = source_path.relative_to(repo_path)
            
            if any(part in SKIP_DIRS for part in rel_path.parts[:-1]):
                continue

            doc.metadata["file_type"] = "code" if source_path.suffix == ".py" else "doc"
            doc.metadata["source"] = str(rel_path)
            filtered_docs.append(doc)
        except ValueError:
            continue

    return filtered_docs

if __name__ == "__main__":
    import sys
    repo = sys.argv[1] if len(sys.argv) > 1 else "."
    docs = load_repository(repo)
    print(f"Loaded {len(docs)} documents from {repo}")
    for d in docs[:5]:
        print(f"  - {d.metadata.get('source')} ({d.metadata.get('file_type')}, {len(d.page_content)} chars)")
