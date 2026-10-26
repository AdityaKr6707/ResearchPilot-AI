
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


DATABASE_PATH = "./chroma_db_pages"
EMBEDDING_MODEL = "nomic-embed-text"


def test_retrieval():
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    store = Chroma(
        persist_directory=DATABASE_PATH,
        embedding_function=embeddings
    )

    retriever = store.as_retriever(
        search_kwargs={"k": 3}
    )

    question = input("Ask a question about your PDF: ").strip()

    if not question:
        print("Please enter a question.")
        return

    results = retriever.invoke(question)

    if not results:
        print("No relevant passages found.")
        return

    print("\nRelevant passages:\n")

    for i, doc in enumerate(results, start=1):
        print(f"--- Result {i} ---")
        print("Source:", doc.metadata.get("source", "Unknown"))
        print("Page:", doc.metadata.get("page", "Unknown"))
        print(doc.page_content[:700])
        print()


if __name__ == "__main__":
    try:
        test_retrieval()
    except Exception as error:
        print("Error:", error)
