
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate

DATABASE_PATH = "./chroma_db"

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

vector_store = Chroma(
    persist_directory=DATABASE_PATH,
    embedding_function=embeddings
)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0.2
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are ResearchPilot AI, a research assistant.
        Answer using the supplied document context.
        If the context does not contain the answer, say so.
        Never invent facts or citations.

        Document context:
        {context}"""
    ),
    ("human", "{question}")
])


def answer_question(question: str) -> str:
    documents = retriever.invoke(question)

    if not documents:
        return "I couldn't find relevant information in the indexed PDF."

    context = "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'Unknown')}, "
        f"Chunk: {doc.metadata.get('chunk', 'Unknown')}]\n"
        f"{doc.page_content}"
        for doc in documents
    )

    chain = prompt | llm

    response = chain.invoke({
        "context": context,
        "question": question
    })

    return response.content


if __name__ == "__main__":
    print("\n=== ResearchPilot AI | PDF Q&A ===")
    print("Ask questions about your indexed PDF.")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not question:
            continue

        try:
            print("\nResearchPilot:\n")
            print(answer_question(question))
            print()
        except Exception as error:
            print(f"Error: {error}")
