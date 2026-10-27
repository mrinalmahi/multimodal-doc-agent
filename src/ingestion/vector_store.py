import json
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


SHARED_COLLECTION_NAME = "all_documents"


def load_pages(text_json_path):
    with open(text_json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def chunk_pages(pages_data, document_id, chunk_size=512, chunk_overlap=50):
    # e5 expects "passage: " on documents and "query: " on queries.
    # The unprefixed text is kept in metadata so answers don't show the prefix.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )

    all_documents = []
    for page in pages_data:
        page_num = page["page_number"]
        text = page["text"]

        for chunk_text in splitter.split_text(text):
            prefixed_text = f"passage: {chunk_text}"
            doc = Document(
                page_content=prefixed_text,
                metadata={
                    "document_id": document_id,
                    "page_number": page_num,
                    "original_text": chunk_text
                }
            )
            all_documents.append(doc)

    return all_documents


def build_vector_store(text_json_path, document_id, db_path="data/chroma_db"):
    # all documents share one collection; search filters by document_id
    pages_data = load_pages(text_json_path)
    documents = chunk_pages(pages_data, document_id)

    embedding_model = HuggingFaceEmbeddings(model_name="intfloat/e5-base-v2")
    vector_store = Chroma(
        collection_name=SHARED_COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=db_path
    )
    vector_store.add_documents(documents)

    print(f"Added {len(documents)} chunks for document_id '{document_id}' "
          f"to shared collection '{SHARED_COLLECTION_NAME}'")
    return vector_store