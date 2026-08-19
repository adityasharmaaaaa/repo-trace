from typing import List, Literal
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_tavily import TavilySearch
from utils.llm_helpers import extract_text

class QueryType(BaseModel):
    query_type: Literal["repo_specific", "general", "mixed"] = Field(
        ...,
        description=(
            "Whether answering the query requires knowledge specific to "
            "this repository's own code, can be answered using general/"
            "public knowledge (e.g. about a library or language feature "
            "this code uses), or genuinely needs both."
        ),
    )


def classify_query(query: str, llm: ChatGoogleGenerativeAI) -> QueryType:
    query_type_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You classify a user query according to what KIND of knowledge is
    needed to answer it after repository retrieval has already been judged
    insufficient.

    Choose exactly one:

    - repo_specific:
    The answer depends on facts, behavior, implementation details, conventions,
    configuration, or APIs that are specific to this repository/codebase.
    The query cannot be answered reliably from general/public knowledge alone.

    - general:
    The query can be answered using general or public knowledge, without needing
    facts about this repository. This includes programming-language behavior,
    well-known libraries/frameworks, standard APIs, algorithms, concepts, or
    publicly documented technology.

    - mixed:
    The query genuinely requires BOTH repository-specific information AND
    general/public knowledge to answer correctly. Use this only when both kinds
    of information are necessary, not when the query is merely ambiguous.

    Important signals:

    1. Repository-specific names are strong evidence for repo_specific.
    Examples:
    - "What does our `AuthManager.refresh_token()` method do?"
    - "Why does the `PaymentService` in this repo retry failed requests?"
    - "Where is `UserRepository` instantiated?"
    These require inspecting this repository's code.

    2. Well-known language/library/framework concepts are strong evidence for
    general.
    Examples:
    - "What does Python's `asyncio.gather()` do?"
    - "How does FastAPI dependency injection work?"
    - "What is the difference between a Python list and tuple?"
    These can be answered from public/general knowledge unless the query also
    asks how this repository specifically uses them.

    3. A query mentioning a general technology is NOT automatically general.
    If it asks how this repository uses that technology, it is repo_specific.
    Example:
    - "How does this repo use FastAPI dependency injection?"
    -> repo_specific

    4. Use mixed only when the answer requires understanding BOTH the repository's
    implementation and the external/general concept.
    Example:
    - "How does our error handling compare to Python's recommended exception
        handling patterns?"
    -> mixed
    The repository is needed to understand "our error handling", while general
    Python knowledge is needed for the comparison.

    5. Do not use mixed as a fallback for uncertainty.
    If the query is clearly about this repository, choose repo_specific.
    If it is clearly answerable without this repository, choose general.

    Your task is classification only. Do not answer the user's query and do not
    explain your reasoning.""",
            ),
            (
                "human",
                """Classify this query:

    {query}""",
            ),
        ]
    )

    correction_llm=llm.with_structured_output(QueryType)
    chain=query_type_prompt | correction_llm
    return chain.invoke({"query": query})


def refine_query(
    original_query: str,
    graded_documents: List[Document],
    llm: ChatGoogleGenerativeAI,
) -> str:

    if not graded_documents:
        broaden_query_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are broadening a repository search query.

The original query retrieved ZERO documents from the repository.

Your job is NOT to answer the query.
Your job is to rewrite the query so that repository retrieval has a
better chance of finding relevant documents.

Because no repository documents were retrieved, you have NO evidence
about repository-specific terminology.

Therefore:

- Preserve the user's original intent.
- Make the query broader and less restrictive.
- Remove unnecessary specificity if present.
- Use only concepts and terminology already present in the original query.
- Do NOT invent repository-specific identifiers.
- Do NOT invent filenames, functions, classes, variables, modules,
  implementation strategies, or domain terminology.
- Do NOT answer the query.
- Return only the broadened search query as plain text.

The goal is to increase recall so that the next repository retrieval
attempt can find some relevant documents.""",
                ),
                (
                    "human",
                    """Original query:
{original_query}

Broaden this query for another repository retrieval attempt.""",
                ),
            ]
        )

        chain = broaden_query_prompt | llm

        response = chain.invoke(
            {"original_query": original_query}
        )

    else:
        refine_query_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are refining a repository-specific search query for
retrieval against the same codebase.

The original query was not answered well by the first retrieval attempt.
Your job is NOT to answer the query. Your job is to rewrite it so that
repository retrieval has a better chance of finding the relevant code.

Use the retrieved documents as evidence of the repository's actual
vocabulary.

When rewriting the query:

- Preserve the original user's intent.
- Extract useful repository-specific terminology from the retrieved
  documents.
- Prefer concrete identifiers when relevant.
- Incorporate repository terminology that is semantically related to
  the original query.
- If the retrieved documents reveal a more precise implementation term,
  prefer that term.
- Do NOT invent identifiers, filenames, functions, classes, or
  terminology that do not appear in the provided documents.
- Do NOT change the question into a different question.
- Do NOT answer the query.
- Return only the rewritten search query as plain text.""",
                ),
                (
                    "human",
                    """Original query:
{original_query}

Retrieved documents from the first attempt:
{graded_documents}

Rewrite the original query using relevant repository-specific
vocabulary found in these documents.""",
                ),
            ]
        )

        formatted_documents = "\n\n".join(
            f"Source: {doc.metadata.get('source', 'unknown')}\n"
            f"{doc.page_content}"
            for doc in graded_documents
        )

        chain = refine_query_prompt | llm

        response = chain.invoke(
            {
                "original_query": original_query,
                "graded_documents": formatted_documents,
            }
        )

    return extract_text(response.content).strip()
    


def web_search_fallback(query: str) -> List[Document]:
    search = TavilySearch(max_results=5)

    response = search.invoke({"query": query})
    results = response.get("results", [])

    documents = []

    for result in results:
        if not isinstance(result, dict):
            continue

        content = result.get("content")
        url = result.get("url")

        if not content:
            continue

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": url or "unknown",
                    "file_type": "web",
                },
            )
        )

    return documents

if __name__ == "__main__":
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.documents import Document

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        temperature=0,
    )

    # ============================================================
    # TEST 1: classify_query()
    # ============================================================

    print("\n" + "=" * 70)
    print("TEST 1: CLASSIFY_QUERY")
    print("=" * 70)

    classification_queries = [
        "How does the application generate a dashboard and decide which charts to include?",
        "How does FastAPI dependency injection work?",
        "How does this repo use FastAPI dependency injection?",
    ]

    for query in classification_queries:
        result = classify_query(query, llm)

        print("\nQuery:")
        print(query)

        print("\nClassification:")
        print(result.query_type)

    # ============================================================
    # TEST 2: refine_query() with actual repository documents
    # ============================================================

    print("\n\n" + "=" * 70)
    print("TEST 2: REFINE_QUERY WITH DOCUMENTS")
    print("=" * 70)

    dashboard_query = (
        "How does the application generate a dashboard "
        "and decide which charts to include?"
    )

    dashboard_documents = [
        Document(
            page_content="""Router -->|sql_qa| SQLQA[SQL Q&A Node]
Router -->|chart| Chart[Chart Node]
Router -->|insight| Insight[Insight Node]
Router -->|anomaly| Anomaly[Anomaly Node]
Router -->|quality| Quality[Quality Node]
Router -->|forecast| Forecast[Forecast Node]
Router -->|dashboard| Dashboard[Dashboard Node]
Router -->|general| General[General Node]

SQLQA <--> Store
Chart <--> Store
Insight <--> Store
Anomaly <--> Store
Quality <--> Store
Forecast <--> Store
Dashboard <--> Store
General --> Finalize[Finalize Node]""",
            metadata={"source": "README.md"},
        ),
        Document(
            page_content="""app.py
src/
  config.py
  logging_config.py
  data/
    loader.py
    sql_store.py
  tools/
    sql_tool.py
    chart_tool.py
    anomaly_tool.py
    quality_tool.py
    forecast_tool.py
  llm/
    gemini.py
    schemas.py
  graph/
    state.py
    router.py
    nodes.py
    build_graph.py""",
            metadata={"source": "README.md"},
        ),
        Document(
            page_content="""class AnomalyPlan(BaseModel):
    table: str = Field(description="Table to analyze for anomalies.")
    numeric_columns: list[str] = Field(description="Numeric columns to check for outliers.")
    group_column: Optional[str] = Field(
        default=None,
        description="Optional column to check outliers within-group."
    )
    id_column: Optional[str] = Field(
        default=None,
        description="Optional identifier column to include for context."
    )


class QualityPlan(BaseModel):
    tables: list[str] = Field(
        description="Which table(s) to run data quality checks on."
    )


class DashboardPlan(BaseModel):
    charts: list[ChartPlan] = Field(
        description="3-4 charts giving a rounded overview of the dataset."
    )""",
            metadata={"source": "src/llm/schemas.py"},
        ),
    ]

    print("\nOriginal query:")
    print(dashboard_query)

    print("\nDocuments passed to refine_query:")

    for i, doc in enumerate(dashboard_documents, 1):
        print(f"\n--- Document {i} ---")
        print(f"Source: {doc.metadata.get('source', 'unknown')}")
        print(doc.page_content)

    refined_dashboard = refine_query(
        original_query=dashboard_query,
        graded_documents=dashboard_documents,
        llm=llm,
    )

    print("\n" + "-" * 70)
    print("REFINED QUERY")
    print("-" * 70)
    print(refined_dashboard)

    # ============================================================
    # TEST 3: refine_query() with ZERO documents
    # ============================================================

    print("\n\n" + "=" * 70)
    print("TEST 3: REFINE_QUERY WITH ZERO DOCUMENTS")
    print("=" * 70)

    forecast_query = (
        "What is the implementation strategy for identifying unusual business records using standardized statistical deviation thresholds within customer-segment-specific cohorts?"
    )

    refined_forecast = refine_query(
        original_query=forecast_query,
        graded_documents=[],
        llm=llm,
    )

    print("\nOriginal query:")
    print(forecast_query)

    print("\nDocuments passed to refine_query: 0")

    print("\n" + "-" * 70)
    print("REFINED QUERY")
    print("-" * 70)
    print(refined_forecast)

    # ============================================================
    # TEST 4: web_search_fallback()
    # ============================================================

    print("\n\n" + "=" * 70)
    print("TEST 4: WEB_SEARCH_FALLBACK")
    print("=" * 70)

    web_query = "How does FastAPI dependency injection work?"

    print("\nQuery:")
    print(web_query)

    web_documents = web_search_fallback(web_query)

    print(f"\nNumber of web documents returned: {len(web_documents)}")

    for i, doc in enumerate(web_documents, 1):
        print(f"\n--- Web Result {i} ---")
        print("Source:", doc.metadata.get("source"))
        print("File type:", doc.metadata.get("file_type"))
        print("Content:")
        print(doc.page_content[:1000])

    # ============================================================
    # TEST 5: END-TO-END CORRECTION DECISION
    # ============================================================

    print("\n\n" + "=" * 70)
    print("TEST 5: END-TO-END CORRECTION DECISION")
    print("=" * 70)

    test_queries = [
        "How does FastAPI dependency injection work?",
        "How does this repo use FastAPI dependency injection?",
        "How does the application generate a dashboard?",
    ]

    for query in test_queries:
        print("\n" + "-" * 70)
        print("Query:")
        print(query)

        classification = classify_query(query, llm)

        print("\nClassification:")
        print(classification.query_type)

        if classification.query_type == "repo_specific":
            print("Action: refine internal repository query")

            refined = refine_query(
                original_query=query,
                graded_documents=dashboard_documents,
                llm=llm,
            )

            print("\nRefined query:")
            print(refined)

        elif classification.query_type == "general":
            print("Action: perform web search")

            web_documents = web_search_fallback(query)

            print(
                f"\nWeb documents returned: "
                f"{len(web_documents)}"
            )

            for i, doc in enumerate(web_documents[:3], 1):
                print(f"\nWeb result {i}:")
                print("Source:", doc.metadata.get("source"))
                print("Content:", doc.page_content[:500])

        else:
            print(
                "Action: mixed query — "
                "requires repository + general knowledge"
            )

