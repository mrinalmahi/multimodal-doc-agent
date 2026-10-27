import re
from pathlib import Path


def generate_document_id(pdf_path, processed_dir="data/processed"):
    stem = Path(pdf_path).stem
    doc_id = re.sub(r"[^a-zA-Z0-9_-]", "_", stem).lower()

    processed_dir = Path(processed_dir)
    candidate = doc_id
    counter = 1
    while (processed_dir / candidate).exists():
        candidate = f"{doc_id}_{counter}"
        counter += 1

    return candidate