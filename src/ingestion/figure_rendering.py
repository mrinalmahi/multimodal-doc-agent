import fitz
from pathlib import Path


def render_pages(pdf_path, output_dir, page_numbers, zoom=2.0):
    """
    Render specific pages of the PDF as PNG images.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    mat = fitz.Matrix(zoom, zoom)

    for page_num in page_numbers:
        page = doc[page_num - 1]
        pix = page.get_pixmap(matrix=mat)
        output_path = output_dir / f"page_{page_num}.png"
        pix.save(str(output_path))

    doc.close()
    print(f"Rendered {len(page_numbers)} pages -> {output_dir}")