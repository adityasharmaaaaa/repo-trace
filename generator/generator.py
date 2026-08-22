from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from utils.llm_helpers import extract_text

LLM_MODEL = "gemini-3.7-flash"


def format_context(documents: List[Document]) -> str:
    final_context = []
    for doc in documents:
        if doc.metadata.get("file_type") == "web":
            final_context.append(
                f"Web source ({doc.metadata.get('source')}):\n{doc.page_content}"
            )
        else:
            final_context.append(
                f"Internal source ({doc.metadata.get('source')}):\n{doc.page_content}"
            )
    return "\n\n".join(final_context)


def generate_answer(
    query: str,
    documents: List[Document],
    llm: ChatGoogleGenerativeAI,
) -> str:
    system_prompt = """You are an expert developer assistant answering a question about a codebase.
    
Your task is to answer the user's question STRICTLY and ONLY using the provided context.

CRITICAL RULES:
1. NO OUTSIDE KNOWLEDGE: You must not use your pre-trained knowledge to answer the question. If the answer is not explicitly supported by the context, do not include it.
2. MISSING INFORMATION: If the provided context does not contain enough information to fully and confidently answer the query, you must explicitly state: "The provided context does not contain enough information to answer this question." Do not attempt to guess or provide a plausible-sounding answer.
3. STRICT CITATIONS: Every factual claim in your answer must be backed by an inline citation to the specific source document.
4. DISTINGUISHING SOURCES: The context contains two types of documents: "Internal source" (this repository's code/docs) and "Web source" (external internet documentation). 
   - When citing the repository's code, use the format `[Internal: <source_path>]`.
   - When citing a web result, use the format `[Web: <url>]`.
   - Make it clear in your prose whether the information describes the repository's actual implementation or external/general knowledge."""

    all_web_sourced = bool(documents) and all(
        doc.metadata.get("file_type") == "web" for doc in documents
    )

    if all_web_sourced:
        system_prompt += """

5. NO REPOSITORY EVIDENCE: None of the retrieved evidence comes from this repository; it is entirely from the web. You MUST begin your answer by stating EXACTLY: "The provided context does not contain enough information to answer this question." Immediately following that, state plainly that this repository does not appear to implement or document the requested topic. Only after providing this clear disclaimer may you answer the question generally using the web sources."""

    system_prompt += """

Context for answering the question:
{context}"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "{query}"),
    ])

    formatted_context = format_context(documents)
    messages = prompt.invoke({
        "context": formatted_context,
        "query": query,
    })

    response = llm.invoke(messages)

    return extract_text(response.content).strip()


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")

    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0)

    # Test 1: strong internal evidence
    query = "how does the router decide which node to call?"
    good_docs = [
        Document(
            page_content=(
                "def router_node(state: AgentState) -> dict:\n"
                "    llm = get_llm(temperature=0)\n"
                "    structured_llm = llm.with_structured_output(RouterDecision)\n"
                "    # ... classifies the question into one of the intents"
            ),
            metadata={"source": "src/graph/router.py", "file_type": "code"},
        ),
    ]
    print("=" * 70)
    print("TEST 1: sufficient evidence")
    print("=" * 70)
    print(generate_answer(query, good_docs, llm))

    # Test 2: no evidence at all - should NOT confidently answer
    print("\n" + "=" * 70)
    print("TEST 2: empty evidence (should admit insufficient info)")
    print("=" * 70)
    print(generate_answer(query, [], llm))

    # Test 3: mixed internal + web - citation style should differ
    mixed_docs = good_docs + [
        Document(
            page_content=(
                "LangGraph's StateGraph supports conditional edges, which route "
                "to different nodes based on the return value of a routing function."
            ),
            metadata={"source": "https://langchain-ai.github.io/langgraph/", "file_type": "web"},
        ),
    ]
    print("\n" + "=" * 70)
    print("TEST 3: mixed internal + web sources")
    print("=" * 70)
    print(generate_answer(
        "how does the router work, and how does that relate to LangGraph's conditional edges generally?",
        mixed_docs,
        llm,
    ))