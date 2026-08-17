from typing import List, Literal
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate


class DocumentGrade(BaseModel):
    grade: Literal["correct", "ambiguous", "incorrect"] = Field(...)


def grade_document(
    query: str,
    document: Document,
    llm: ChatGoogleGenerativeAI,
) -> "DocumentGrade":  
    grader_llm=llm.with_structured_output(DocumentGrade)
    grading_prompt = ChatPromptTemplate.from_messages([
      (
          "system",
          """
    You are a retrieval evaluator in a RAG system.

    Your task is to evaluate whether a single retrieved document is useful
    for answering the user's query.

    Do NOT judge relevance based merely on shared keywords or general topic
    similarity. The key question is:

    "Would this specific document provide useful evidence or information
    needed to answer the query?"

    Grade the document using exactly one of these labels:

    - correct:
    The document is directly relevant and contains useful information that
    can materially help answer the query.

    - ambiguous:
    The document is related to the query and may provide some useful context,
    but the information is incomplete, indirect, unclear, or insufficient
    to confidently answer the query.

    - incorrect:
    The document does not provide useful information for answering the query,
    even if it contains related keywords or discusses a similar topic.

    Guidelines:
    1. Judge the document against the specific query.
    2. Do not give "correct" merely because the document contains matching
      keywords.
    3. Prefer "ambiguous" when the document provides partial but potentially
      useful information.
    4. Use "incorrect" when the document would not materially help construct
      an answer.
    5. Base your judgment only on the provided query and document. Do not
      assume information that is not present in the document.

    Return only the structured output matching the provided schema.
          """,
      ),
      (
          "human",
          """
    Query:
    {query}

    Retrieved document:
    {document}
          """,
      ),
    ])

    chain=grading_prompt | grader_llm
    return chain.invoke({"query": query, "document": document.page_content})
    


def aggregate_grades(
    grades: List["DocumentGrade"],
) -> Literal["correct", "ambiguous", "incorrect"]:
    correct=sum(g.grade=="correct" for g in grades)
    ambiguous=sum(g.grade=="ambiguous" for g in grades)

    if correct>=1:
        return "correct"
    elif ambiguous>=1:
        return "ambiguous"
    else:
        return "incorrect"

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from indexing.vector_store import load_vector_store
    from retrieval.mmr_retriever import build_mmr_retriever, build_compressed_retriever

    persist_dir = sys.argv[1] if len(sys.argv) > 1 else "./chroma_store"
    query = sys.argv[2] if len(sys.argv) > 2 else "How does the application generate a dashboard and decide which charts to include?"

    vectorstore = load_vector_store(persist_dir)
    mmr_retriever = build_mmr_retriever(vectorstore)
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)
    compressed_retriever = build_compressed_retriever(mmr_retriever, llm)

    docs = compressed_retriever.invoke(query)
    print(f"Grading {len(docs)} documents for query: {query!r}\n")

    grades = []
    for doc in docs:
        grade = grade_document(query, doc, llm)
        grades.append(grade)
        print(f"--- {doc.metadata.get('source')} ---")
        print(f"  {grade}")
        print(f"  content preview: {doc.page_content[:100]!r}\n")

    overall = aggregate_grades(grades)
    print(f"OVERALL ASSESSMENT: {overall}")
