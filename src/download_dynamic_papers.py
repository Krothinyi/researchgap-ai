
import json
import re
from pathlib import Path

import requests

if __package__:
    from .search_papers import search_papers
else:
    from search_papers import search_papers

# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
PDF_DIR = BASE_DIR / "data" / "pdfs"
MANIFEST_PATH = BASE_DIR / "data" / "downloaded_papers.json"

PDF_DIR.mkdir(parents=True, exist_ok=True)

MAX_PAPERS = 2
MAX_PDF_SIZE = 20 * 1024 * 1024  # 20 MB

# ==========================================
# 2. DOWNLOAD ONE PDF
# ==========================================

def download_pdf(paper):
    paper_id = paper["paper_id"].split("/")[-1]

    if not re.fullmatch(r"W\d+", paper_id):
        raise ValueError("Invalid OpenAlex paper ID")

    output_path = PDF_DIR / f"{paper_id}.pdf"

    # Reuse previously downloaded PDFs
    if output_path.exists():
        with output_path.open("rb") as f:
            if f.read(5) == b"%PDF-":
                print("Already downloaded:", output_path.name)
                return output_path

    temp_path = output_path.with_suffix(".pdf.part")

    try:
        with requests.get(
            paper["pdf_url"],
            stream=True,
            timeout=(15, 60),
            headers={"User-Agent": "ResearchGapAI/0.1"},
        ) as response:

            response.raise_for_status()

            total_bytes = 0

            with temp_path.open("wb") as f:
                for chunk in response.iter_content(65536):
                    if not chunk:
                        continue

                    total_bytes += len(chunk)

                    if total_bytes > MAX_PDF_SIZE:
                        raise ValueError("PDF exceeds 20 MB limit")

                    f.write(chunk)

        with temp_path.open("rb") as f:
            if f.read(5) != b"%PDF-":
                raise ValueError("Downloaded content is not a PDF")

        temp_path.replace(output_path)

        print("Downloaded:", output_path.name)
        return output_path

    finally:
        if temp_path.exists():
            temp_path.unlink()


# ==========================================
# 3. SEARCH AND DOWNLOAD
# ==========================================

def main():
    topic = input("Enter research topic: ").strip()

    if not topic:
        print("Please enter a research topic.")
        return

    papers = search_papers(topic, limit=10)

    if MANIFEST_PATH.exists():
        manifest = json.loads(
            MANIFEST_PATH.read_text(encoding="utf-8")
        )
    else:
        manifest = {}

    selected = []
    seen_titles = set()

    for paper in papers:
        if not paper.get("pdf_url"):
            continue

        normalized_title = " ".join(
            (paper.get("title") or "").lower().split()
        )

        if normalized_title in seen_titles:
            continue

        seen_titles.add(normalized_title)
        selected.append(paper)

        if len(selected) >= MAX_PAPERS:
            break

    print("\nPapers selected:", len(selected))

    for paper in selected:
        print("\nTitle:", paper["title"])

        try:
            pdf_path = download_pdf(paper)

            manifest[paper["paper_id"]] = {
                **paper,
                "local_pdf": str(
                    pdf_path.relative_to(BASE_DIR)
                ),
            }

        except (requests.RequestException, OSError, ValueError) as e:
            print("Download failed:", e)

    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("\n" + "=" * 50)
    print("DOWNLOAD PROCESS COMPLETED")
    print("=" * 50)
    print("Total papers in manifest:", len(manifest))
    print("Metadata saved to:", MANIFEST_PATH)


if __name__ == "__main__":
    main()
