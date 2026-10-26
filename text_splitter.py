
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pdf_reader import read_pdf_documents


def split_pdf_into_documents(pdf_path: str):
    pages = read_pdf_documents(pdf_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(pages)
    return chunks


if __name__ == "__main__":
    path = input("Enter PDF file path: ").strip().strip('"')

    try:
        chunks = split_pdf_into_documents(path)

        print(f"\nTotal chunks created: {len(chunks)}")

        for i, chunk in enumerate(chunks[:3], start=1):
            print(f"\n--- Chunk {i} ---")
            print("Original page:", chunk.metadata["page"])
            print("Source:", chunk.metadata["source"])
            print(chunk.page_content[:300])

    except Exception as error:
        print(f"Error: {error}")
