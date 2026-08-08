import hashlib
import json
from pathlib import Path


def compute_file_hash(pdf_path):
    """
    Compute a SHA-256 hash of the file's actual bytes, so we can
    detect true duplicates regardless of filename.
    """
    hasher = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_hash_registry(registry_path="data/processed/_hash_registry.json"):
    registry_path = Path(registry_path)
    if registry_path.exists():
        with open(registry_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_hash_registry(registry, registry_path="data/processed/_hash_registry.json"):
    registry_path = Path(registry_path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)


def find_existing_document(pdf_path, registry_path="data/processed/_hash_registry.json"):
    """
    Check if this exact file (by content) has already been ingested.
    Returns the existing document_id if found, otherwise None.
    """
    file_hash = compute_file_hash(pdf_path)
    registry = load_hash_registry(registry_path)
    return registry.get(file_hash)


def register_document(pdf_path, document_id, registry_path="data/processed/_hash_registry.json"):
    """
    Record that this file's content hash maps to the given document_id.
    """
    file_hash = compute_file_hash(pdf_path)
    registry = load_hash_registry(registry_path)
    registry[file_hash] = document_id
    save_hash_registry(registry, registry_path)