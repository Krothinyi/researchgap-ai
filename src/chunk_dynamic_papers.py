
import json
import re
from pathlib import Path

# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

MANIFEST_PATH = BASE_DIR / "data" / "downloaded_papers.json"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

CHUNK_SIZE = 250
CHUNK_OVERLAP = 40

# ==========================================
# 2. CLEAN EXTRACTED TEXT
# ==========================================

def clean_text(text):

    # Remove soft hyphens
    text = text.replace("\u00ad", "")

    # Repair words broken across lines
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ==========================================
# 3. SPLIT TEXT BY PDF PAGE
# ==========================================

def extract_pages(full_text):

    parts = re.split(
        r"===\s*PAGE\s+(\d+)\s*===",
        full_text
    )

    pages = []

    for i in range(1, len(parts), 2):

        page_number = int(parts[i])
        page_text = clean_text(parts[i + 1])

        if page_text:
            pages.append({
                "page": page_number,
                "text": page_text
            })

    return pages


# ==========================================
# 4. CREATE CHUNKS
# ==========================================

def create_chunks(paper_id, pages):

    chunks = []

    short_id = paper_id.split("/")[-1]

    step = CHUNK_SIZE - CHUNK_OVERLAP

    for page_data in pages:

        page_number = page_data["page"]
        words = page_data["text"].split()

        chunk_number = 0

        for start in range(0, len(words), step):

            chunk_words = words[start:start + CHUNK_SIZE]

            if not chunk_words:
                break

            chunk_number += 1

            chunk = {
                "chunk_id": (
                    f"{short_id}_p{page_number}_c{chunk_number}"
                ),
                "paper_id": paper_id,
                "page": page_number,
                "text": " ".join(chunk_words),
                "word_count": len(chunk_words)
            }

            chunks.append(chunk)

            if start + CHUNK_SIZE >= len(words):
                break

    return chunks


# ==========================================
# 5. PROCESS ALL PAPERS
# ==========================================

def main():

    manifest = json.loads(
        MANIFEST_PATH.read_text(encoding="utf-8")
    )

    print("=" * 60)
    print("RESEARCHGAP AI - DYNAMIC CHUNKING")
    print("=" * 60)

    total_chunks = 0

    for paper_id, paper in manifest.items():

        short_id = paper_id.split("/")[-1]

        text_path = PROCESSED_DIR / f"{short_id}.txt"

        if not text_path.exists():
            print(f"Text file missing: {short_id}")
            continue

        print(f"\nProcessing: {paper['title']}")

        full_text = text_path.read_text(
            encoding="utf-8"
        )

        pages = extract_pages(full_text)

        if not pages:
            print("No usable text found.")
            continue

        chunks = create_chunks(paper_id, pages)

        output_path = (
            PROCESSED_DIR / f"{short_id}_chunks.json"
        )

        output_path.write_text(
            json.dumps(
                chunks,
                indent=2,
                ensure_ascii=False
            ),
            encoding="utf-8"
        )

        total_chunks += len(chunks)

        print("Pages processed:", len(pages))
        print("Chunks created:", len(chunks))
        print("Saved to:", output_path.name)

    print("\n" + "=" * 60)
    print("CHUNKING COMPLETED")
    print("=" * 60)

    print("Total chunks created:", total_chunks)


if __name__ == "__main__":
    main()
