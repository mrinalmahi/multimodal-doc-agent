import fitz
import json
from collections import defaultdict
from pathlib import Path


def extract_page_text(page, column_gap_threshold=0.5):
    """
    Extract text from a single PDF page, handling two-column layout.
    """
    page_width = page.rect.width
    midpoint = page_width * column_gap_threshold

    blocks = page.get_text("blocks")

    full_width_blocks = []
    left_column = []
    right_column = []

    for block in blocks:
        x0, y0, x1, y1, text, block_no, block_type = block

        if block_type != 0:
            continue

        block_width = x1 - x0

        if block_width > page_width * 0.7:
            full_width_blocks.append((y0, text))
        elif (x0 + x1) / 2 < midpoint:
            left_column.append((y0, text))
        else:
            right_column.append((y0, text))

    full_width_blocks.sort(key=lambda b: b[0])
    left_column.sort(key=lambda b: b[0])
    right_column.sort(key=lambda b: b[0])

    ordered_text = []
    for _, text in full_width_blocks:
        ordered_text.append(text)
    for _, text in left_column:
        ordered_text.append(text)
    for _, text in right_column:
        ordered_text.append(text)

    return "\n".join(ordered_text)


def extract_raw_pages(pdf_path):
    """
    Extract raw (uncleaned) text from every page of the PDF.
    """
    doc = fitz.open(pdf_path)
    pages_data = []

    for page_num, page in enumerate(doc):
        text = extract_page_text(page)
        pages_data.append({
            "page_number": page_num + 1,
            "text": text
        })

    doc.close()
    return pages_data


def split_into_lines(page_text):
    """
    Split a page's text into non-empty, whitespace-stripped lines.
    """
    return [line.strip() for line in page_text.split("\n") if line.strip()]


def find_repeating_lines(pages_data, min_page_fraction=0.5):
    """
    Find lines of text that repeat across many pages -- these are
    likely running headers/footers, regardless of what they say.
    """
    total_pages = len(pages_data)
    line_page_count = defaultdict(set)

    for page in pages_data:
        for line in split_into_lines(page["text"]):
            if len(line) < 100:
                line_page_count[line].add(page["page_number"])

    threshold = total_pages * min_page_fraction
    repeating_lines = {
        line for line, pages in line_page_count.items()
        if len(pages) >= threshold
    }

    return repeating_lines


def clean_pages(pages_data, repeating_lines):
    """
    Remove repeating header/footer lines and collapse extra blank
    lines left behind afterward.
    """
    cleaned_pages = []
    for page in pages_data:
        lines = page["text"].split("\n")
        kept_lines = [
            line for line in lines
            if line.strip() not in repeating_lines
        ]
        cleaned_text = "\n".join(kept_lines)
        while "\n\n\n" in cleaned_text:
            cleaned_text = cleaned_text.replace("\n\n\n", "\n\n")

        cleaned_pages.append({
            "page_number": page["page_number"],
            "text": cleaned_text.strip()
        })
    return cleaned_pages


def extract_and_clean_text(pdf_path, output_dir):
    """
    Full text extraction + cleanup pipeline for one document.
    Writes the result to output_dir/text_chunks_clean.json and
    returns the path to that file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pages_data = extract_raw_pages(pdf_path)
    repeating_lines = find_repeating_lines(pages_data)
    cleaned_pages = clean_pages(pages_data, repeating_lines)

    output_path = output_dir / "text_chunks_clean.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_pages, f, indent=2, ensure_ascii=False)

    print(f"Extracted and cleaned {len(cleaned_pages)} pages -> {output_path}")
    return output_path