
from typing import TypedDict

from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import StateGraph, START, END


# Configuration
DATABASE_PATH = "./chroma_db_multi"
EMBEDDING_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2:3b"
RETRIEVAL_COUNT = 3


# Graph state
class ResearchState(TypedDict):
    question: str
    context: str
    answer: str
    citations: list[str]


# Local embedding model
embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL
)


# Existing local vector database
vector_store = Chroma(
    persist_directory=DATABASE_PATH,
    embedding_function=embeddings
)


# Local language model
llm = ChatOllama(
    model=CHAT_MODEL,
    temperature=0.2
)


def retrieve_documents(state: ResearchState) -> dict:
    """Retrieve relevant passages and preserve their source metadata."""

    documents = vector_store.similarity_search(
        state["question"],
        k=RETRIEVAL_COUNT
    )

    passages = []
    citations = []

    for doc in documents:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "Unknown")

        citation = f"[Source: {source}, Page: {page}]"

        passages.append(
            f"{citation}\n"
            f"Content:\n{doc.page_content}"
        )

        if citation not in citations:
            citations.append(citation)

    return {
        "context": "\n\n---\n\n".join(passages),
        "citations": citations
    }


def generate_answer(state: ResearchState) -> dict:
    """Generate an answer using retrieved PDF passages."""

    if not state["context"].strip():
        return {
            "answer": "I couldn't find relevant text in the indexed PDF."
        }

    prompt = f"""
You are ResearchPilot AI, a research assistant.

RULES:
1. Answer using the supplied PDF passages.
2. Do not invent facts, sources, or page numbers.
3. If the passages do not answer the question, say so clearly.
4. Cite supporting claims using the source and page metadata
   shown in the passages.
5. Distinguish information from the paper from your own explanation.

PDF PASSAGES:
{state["context"]}

QUESTION:
{state["question"]}

Write a clear, useful answer with citations.
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }


def attach_verified_sources(state: ResearchState) -> dict:
    """Append source references taken directly from retrieved metadata."""

    citations = state.get("citations", [])

    if not citations:
        return {"answer": state["answer"]}

    sources_section = "\n\nRetrieved sources:\n" + "\n".join(
        f"- {citation}" for citation in citations
    )

    return {
        "answer": state["answer"] + sources_section
    }


# Build the LangGraph workflow
builder = StateGraph(ResearchState)

builder.add_node("retrieve", retrieve_documents)
builder.add_node("generate", generate_answer)
builder.add_node("attach_sources", attach_verified_sources)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", "attach_sources")
builder.add_edge("attach_sources", END)

research_graph = builder.compile()


def main():
    print("\n=== ResearchPilot AI ===")
    print("PDF Research | LangChain + LangGraph + Ollama")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() == "exit":
            print("ResearchPilot closed.")
            break

        if not question:
            continue

        try:
            result = research_graph.invoke({
                "question": question,
                "context": "",
                "answer": "",
                "citations": []
            })

            print("\nResearchPilot:\n")
            print(result["answer"])
            print()

        except Exception as error:
            print(f"\nError: {error}\n")


if __name__ == "__main__":
    main()
