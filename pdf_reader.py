
from pypdf import PdfReader
from langchain_core.documents import Document


def read_pdf_documents(file_path: str) -> list[Document]:
    reader = PdfReader(file_path)
    documents = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if text and text.strip():
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": file_path,
                        "page": page_number
                    }
                )
            )

    return documents


def read_pdf(file_path: str) -> str:
    documents = read_pdf_documents(file_path)
    return "\n\n".join(
        f"--- Page {doc.metadata['page']} ---\n{doc.page_content}"
        for doc in documents
    )


if __name__ == "__main__":
    path = input("Enter PDF file path: ").strip().strip('"')

    try:
        docs = read_pdf_documents(path)
        print(f"Pages with extractable text: {len(docs)}")

        for doc in docs[:2]:
            print(f"\nPage {doc.metadata['page']}:")
            print(doc.page_content[:500])

    except Exception as error:
        print(f"Could not read PDF: {error}")
