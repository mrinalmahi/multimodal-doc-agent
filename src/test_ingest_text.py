# src/test_ingest_full.py
from ingestion.pipeline import ingest_document

if __name__ == "__main__":
    doc_id = ingest_document("data/pdfs/explainability-methods-from-machine-learning-detect-important-drugs-atoms-in-drug-target-interactions.pdf")
    print(f"\nFinal document_id: {doc_id}")