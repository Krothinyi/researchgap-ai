
import json
from pathlib import Path

import pymupdf as fitz

# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

MANIFEST_PATH = BASE_DIR / "data" / "downloaded_papers.json"

OUTPUT_DIR = BASE_DIR / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. PARSE ONE PDF
# ==========================================

def parse_pdf(pdf_path, output_path):

    print(f"\nProcessing: {pdf_path.name}")

    if not pdf_path.exists():
        print("PDF not found:", pdf_path)
        return

    with fitz.open(pdf_path) as document:

        total_pages = len(document)
        all_pages = []
        total_characters = 0

        for page_number, page in enumerate(document, start=1):

            text = page.get_text(
                "text",
                sort=False
            )

            if not text.strip():
                print(
                    f"Warning: Page {page_number} "
                    "has no extractable text."
                )

            total_characters += len(text)

            page_text = (
                f"\n=== PAGE {page_number} ===\n"
                f"{text}\n"
            )

            all_pages.append(page_text)

    full_text = "\n".join(all_pages)

    output_path.write_text(
        full_text,
        encoding="utf-8"
    )

    print("Total pages:", total_pages)
    print("Characters extracted:", total_characters)
    print("Saved to:", output_path.name)

# ==========================================
# 3. PROCESS ALL DOWNLOADED PAPERS
# ==========================================

def main():

    manifest = json.loads(
        MANIFEST_PATH.read_text(encoding="utf-8")
    )

    print("=" * 60)
    print("RESEARCHGAP AI - AUTOMATIC PDF PARSING")
    print("=" * 60)

    for paper_id, paper in manifest.items():

        local_pdf = paper.get("local_pdf")

        if not local_pdf:
            continue

        pdf_path = BASE_DIR / local_pdf

        short_id = paper_id.split("/")[-1]

        output_path = (
            OUTPUT_DIR / f"{short_id}.txt"
        )

        # Avoid repeated processing
        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"\nAlready processed: {short_id}")
            continue

        try:
            parse_pdf(
                pdf_path,
                output_path
            )

        except Exception as e:
            print(f"Failed to process {short_id}: {e}")

    print("\n" + "=" * 60)
    print("PDF PARSING COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
