
from pathlib import Path
import hashlib

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from text_splitter import split_pdf_into_documents


PAPERS_FOLDER = Path("./papers")
DATABASE_PATH = "./chroma_db_multi"
EMBEDDING_MODEL = "nomic-embed-text"


def make_chunk_id(source: str, page: int, content: str) -> str:
    raw = f"{source}|{page}|{content}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def create_vector_store():
    pdf_files = sorted(PAPERS_FOLDER.glob("*.pdf"))

    if not pdf_files:
        raise ValueError(
            "No PDF files found. Add PDFs to the papers folder."
        )

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    store = Chroma(
        collection_name="research_papers",
        embedding_function=embeddings,
        persist_directory=DATABASE_PATH
    )

    total_added = 0
    total_removed = 0

    # Get all currently indexed chunk IDs and their source metadata.
    existing = store.get(include=["metadatas"])

    existing_ids = existing["ids"]
    existing_metadatas = existing["metadatas"] or []

    indexed_by_source = {}

    for chunk_id, metadata in zip(existing_ids, existing_metadatas):
        source = metadata.get("source", "")
        indexed_by_source.setdefault(source, set()).add(chunk_id)

    current_sources = {pdf_path.name for pdf_path in pdf_files}

    # Remove records for PDFs that no longer exist in the folder.
    for source, chunk_ids in indexed_by_source.items():
        if source not in current_sources:
            store.delete(ids=list(chunk_ids))
            total_removed += len(chunk_ids)
            print(f"Removed old PDF from index: {source}")

    for pdf_path in pdf_files:
        source = pdf_path.name
        print(f"\nProcessing: {source}")

        chunks = split_pdf_into_documents(str(pdf_path))

        if not chunks:
            print("No extractable text found. Skipping.")
            continue

        new_ids = []
        new_chunks = []

        for chunk in chunks:
            page = int(chunk.metadata.get("page", 0))
            chunk.metadata["source"] = source

            chunk_id = make_chunk_id(
                source,
                page,
                chunk.page_content
            )

            new_ids.append(chunk_id)
            new_chunks.append(chunk)

        old_ids = indexed_by_source.get(source, set())
        new_id_set = set(new_ids)

        # Remove stale chunks when the PDF content changes.
        stale_ids = list(old_ids - new_id_set)

        if stale_ids:
            store.delete(ids=stale_ids)
            total_removed += len(stale_ids)
            print(f"Old chunks removed: {len(stale_ids)}")

        # Add only chunks that are not already indexed.
        existing_ids_for_source = old_ids & new_id_set

        chunks_to_add = []
        ids_to_add = []

        for chunk, chunk_id in zip(new_chunks, new_ids):
            if chunk_id not in existing_ids_for_source:
                chunks_to_add.append(chunk)
                ids_to_add.append(chunk_id)

        if chunks_to_add:
            store.add_documents(
                documents=chunks_to_add,
                ids=ids_to_add
            )

        total_added += len(chunks_to_add)

        print(f"Chunks found: {len(chunks)}")
        print(f"New chunks added: {len(chunks_to_add)}")
        print(f"Existing chunks kept: {len(existing_ids_for_source)}")

    print("\nIndexing complete!")
    print(f"PDF files processed: {len(pdf_files)}")
    print(f"New chunks added: {total_added}")
    print(f"Old chunks removed: {total_removed}")
    print(f"Database: {DATABASE_PATH}")

    return store


if __name__ == "__main__":
    try:
        create_vector_store()
    except Exception as error:
        print(f"\nError: {error}")
