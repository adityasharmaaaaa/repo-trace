from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language


def split_documents(docs: List[Document]) -> List[Document]:
    python_splitter=RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=1500,  # Chosen to capture most small functions in one chunk while allowing for overlap in larger ones.
        chunk_overlap=100
    )

    markdown_splitter=RecursiveCharacterTextSplitter.from_language(
        language=Language.MARKDOWN,
        chunk_size=1000,  # Chosen to capture meaningful sections of documentation while allowing for overlap.
        chunk_overlap=100
    )

    chunks=[]
    for doc in docs:
        if doc.metadata.get("file_type")=="code":
            chunks.extend(python_splitter.split_documents([doc]))
        elif doc.metadata.get("file_type")=="doc":
            chunks.extend(markdown_splitter.split_documents([doc]))
        else:
            raise ValueError(f"Unknown file_type: {doc.metadata.get('file_type')}")

    return chunks


if __name__ == "__main__":
    import sys
    from loader import load_repository

    repo = sys.argv[1] if len(sys.argv) > 1 else "."
    docs = load_repository(repo)
    chunks = split_documents(docs)
    print(f"{len(docs)} documents -> {len(chunks)} chunks")
    for i, chunk in enumerate(chunks[:8]):
        print(f"\n--- chunk {i + 1} ---")
        print(chunk.page_content)
        print(f"\nmetadata: {chunk.metadata}")

        assert "source" in chunk.metadata
        assert "file_type" in chunk.metadata

        print("Metadata check: OK")