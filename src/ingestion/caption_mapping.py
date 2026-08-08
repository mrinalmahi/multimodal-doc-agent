import json
import re
from pathlib import Path


def build_caption_page_map(text_json_path, label):
    """
    Scan cleaned page text for captions like "Figure N." or "Table N.",
    and record every page each number appears on (in case a table or
    figure spans multiple pages, e.g. "Table 3 (Continued)" on a
    following page).

    label should be "Figure" or "Table". Matches common journal
    variations: full word or abbreviation ("Fig." for figures), a
    period or whitespace after the number, and any letter casing
    (e.g. "FIGURE 1 |", "Table 2 ", "fig. 3.").

    Returns a dict mapping each number to a sorted list of page
    numbers it appears on -- always a list, even if only one page.
    """
    with open(text_json_path, "r", encoding="utf-8") as f:
        pages_data = json.load(f)

    if label == "Figure":
        word_pattern = r"(?:Figure|Fig\.)"
    else:
        word_pattern = "Table"

    pattern = re.compile(
        rf"^{word_pattern}\s+(\d+)[.\s]",
        re.MULTILINE | re.IGNORECASE
    )

    all_occurrences = {}
    for page in pages_data:
        page_num = page["page_number"]
        text = page["text"]

        for num_str in pattern.findall(text):
            num = int(num_str)
            all_occurrences.setdefault(num, []).append(page_num)

    page_map = {
        num: sorted(set(pages))
        for num, pages in all_occurrences.items()
    }

    return page_map


def build_and_save_caption_maps(text_json_path, output_dir):
    """
    Build both the figure-page map and table-page map, saving each
    as its own JSON file in output_dir. Each entry maps a number to
    a list of page numbers (usually length 1, longer if the item
    spans multiple pages). Returns both maps directly, since the
    pipeline needs the actual page numbers to decide which pages to
    render/crop.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figure_map = build_caption_page_map(text_json_path, "Figure")
    table_map = build_caption_page_map(text_json_path, "Table")

    figure_map_path = output_dir / "figure_page_map.json"
    table_map_path = output_dir / "table_page_map.json"

    with open(figure_map_path, "w", encoding="utf-8") as f:
        json.dump(figure_map, f, indent=2)
    with open(table_map_path, "w", encoding="utf-8") as f:
        json.dump(table_map, f, indent=2)

    print(f"Figure map ({len(figure_map)} figures) -> {figure_map_path}")
    print(f"Table map ({len(table_map)} tables) -> {table_map_path}")

    multi_page_tables = {k: v for k, v in table_map.items() if len(v) > 1}
    multi_page_figures = {k: v for k, v in figure_map.items() if len(v) > 1}
    if multi_page_tables:
        print(f"Multi-page tables detected: {multi_page_tables}")
    if multi_page_figures:
        print(f"Multi-page figures detected: {multi_page_figures}")

    return figure_map, table_map