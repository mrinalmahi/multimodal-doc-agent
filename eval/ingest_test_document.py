import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.pipeline import ingest_document

TEST_PDF_PATH = "eval/fixtures/test_paper.pdf"
DOCUMENT_ID_OUTPUT_PATH = "eval/test_document_id.json"


def main():
    print(f"Ingesting test document: {TEST_PDF_PATH}")
    doc_id = ingest_document(TEST_PDF_PATH)

    with open(DOCUMENT_ID_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"document_id": doc_id}, f, indent=2)

    print(f"Test document ingested with document_id: {doc_id}")
    print(f"Saved to {DOCUMENT_ID_OUTPUT_PATH}")


if __name__ == "__main__":
    main()