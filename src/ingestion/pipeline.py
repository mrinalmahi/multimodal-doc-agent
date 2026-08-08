from .document_id import generate_document_id
from .duplicate_detection import find_existing_document, register_document
from .text_extraction import extract_and_clean_text
from .caption_mapping import build_and_save_caption_maps
from .figure_rendering import render_pages
from .table_cropping import crop_all_tables
from .vector_store import build_vector_store


def ingest_document(pdf_path, processed_root="data/processed", db_path="data/chroma_db"):
    existing_doc_id = find_existing_document(pdf_path)
    if existing_doc_id:
        print(f"This file was already ingested as '{existing_doc_id}' -- skipping reprocessing.")
        return existing_doc_id

    doc_id = generate_document_id(pdf_path, processed_dir=processed_root)
    processed_dir = f"{processed_root}/{doc_id}"

    print(f"Ingesting document '{doc_id}' from {pdf_path}")

    text_path = extract_and_clean_text(pdf_path, processed_dir)
    figure_map, table_map = build_and_save_caption_maps(text_path, processed_dir)

    pages_to_render = sorted({p for pages in figure_map.values() for p in pages})
    render_pages(pdf_path, f"{processed_dir}/figures", pages_to_render)

    crop_all_tables(pdf_path, table_map, f"{processed_dir}/tables")
    build_vector_store(text_path, document_id=doc_id, db_path=db_path)

    register_document(pdf_path, doc_id)

    print(f"Ingestion complete for document_id: {doc_id}")
    return doc_id