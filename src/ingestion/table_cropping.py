import fitz
from pathlib import Path
from io import BytesIO
from PIL import Image


def get_page_blocks(page):
    blocks = page.get_text("blocks")
    text_blocks = [b for b in blocks if b[6] == 0]
    text_blocks.sort(key=lambda b: b[1])
    return text_blocks


def find_caption_block(blocks, table_num):
    # "table n " also matches continuation captions like "Table 3 | (Continued)"
    prefix_dot = f"table {table_num}."
    prefix_space = f"table {table_num} "

    for block in blocks:
        text = block[4].strip().lower()
        if text.startswith(prefix_dot) or text.startswith(prefix_space):
            return block
    return None


def collect_table_blocks(blocks, caption_block, page_width, gap_threshold=15):
    # Walk down from the caption through blocks in the same column and
    # stop at the first big vertical gap, which is where the table ends.
    cap_x0, cap_y0, cap_x1, cap_y1 = caption_block[:4]
    midpoint = page_width * 0.5
    caption_in_left = cap_x0 < midpoint

    candidates = []
    for b in blocks:
        if b[1] <= cap_y1:
            continue
        b_x0, b_x1 = b[0], b[2]
        b_straddles = b_x0 < midpoint and b_x1 > midpoint
        b_in_left = b_x0 < midpoint
        if b_straddles or b_in_left == caption_in_left:
            candidates.append(b)

    candidates.sort(key=lambda b: b[1])

    table_blocks = [caption_block]
    prev_y1 = cap_y1
    boundary_y = None

    for block in candidates:
        gap = block[1] - prev_y1
        if gap > gap_threshold:
            boundary_y = block[1]
            break
        table_blocks.append(block)
        prev_y1 = block[3]

    if boundary_y is None:
        boundary_y = candidates[-1][3] if candidates else cap_y1

    return table_blocks, boundary_y


def compute_crop_bounds(table_blocks, boundary_y, page_width, padding=5):
    midpoint = page_width * 0.5

    min_x0 = min(b[0] for b in table_blocks)
    max_x1 = max(b[2] for b in table_blocks)
    min_y0 = min(b[1] for b in table_blocks)

    straddles = min_x0 < midpoint and max_x1 > midpoint

    if straddles:
        crop_left, crop_right = 0, page_width
    elif min_x0 < midpoint:
        crop_left, crop_right = 0, midpoint
    else:
        crop_left, crop_right = midpoint, page_width

    crop_top = min_y0 - padding
    crop_bottom = boundary_y - padding

    return crop_left, crop_top, crop_right, crop_bottom


def render_crop(page, clip_rect, zoom=2.0):
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, clip=clip_rect)
    img_bytes = pix.tobytes("png")
    return Image.open(BytesIO(img_bytes)).convert("RGB")


def crop_single_page(pdf_path, doc, page_num, table_num, zoom=2.0, gap_threshold=15):
    page =doc[page_num - 1]
    page_width = page.rect.width

    blocks = get_page_blocks(page)
    caption_block = find_caption_block(blocks, table_num)
    if caption_block is None:
        return None

    table_blocks, boundary_y = collect_table_blocks(
        blocks, caption_block, page_width, gap_threshold=gap_threshold
    )
    crop_left, crop_top, crop_right, crop_bottom = compute_crop_bounds(
        table_blocks, boundary_y, page_width
    )
    clip_rect = fitz.Rect(crop_left, crop_top, crop_right, crop_bottom)
    return render_crop(page, clip_rect, zoom=zoom)


def crop_table_from_pages(pdf_path, page_nums, table_num, output_path,
                            zoom=2.0, gap_threshold=15):
    # multi-page tables get cropped page by page and stacked vertically
    doc =fitz.open(pdf_path)

    images = []
    for page_num in page_nums:
        img = crop_single_page(pdf_path, doc, page_num, table_num,
                                 zoom=zoom, gap_threshold=gap_threshold)
        if img is not None:
            images.append(img)

    doc.close()

    if not images:
        return False

    if len(images) == 1:
        images[0].save(output_path)
    else:
        combined_width = max(img.width for img in images)
        combined_height = sum(img.height for img in images)
        combined = Image.new("RGB", (combined_width, combined_height), "white")
        y_offset = 0
        for img in images:
            combined.paste(img, (0, y_offset))
            y_offset += img.height
        combined.save(output_path)

    return True


def crop_all_tables(pdf_path, table_map, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    for table_num, page_nums in table_map.items():
        output_path = output_dir / f"table_{table_num}.png"

        success = crop_table_from_pages(
            pdf_path=pdf_path,
            page_nums=page_nums,
            table_num=table_num,
            output_path=str(output_path)
        )
        if success:
            success_count += 1
        else:
            print(f"  Warning: could not crop Table {table_num} on page(s) {page_nums}")

    print(f"Cropped {success_count}/{len(table_map)} tables -> {output_dir}")