from src.demo_config import require_write_permission
import os
import json
from pathlib import Path

import requests
from dotenv import load_dotenv
from qdrant_client import models

from src.download_dynamic_papers import download_pdf
from src.parse_dynamic_papers import parse_pdf
from src.chunk_dynamic_papers import extract_pages, create_chunks
from src.index_single_paper import index_paper


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

MANIFEST_PATH = (
    BASE_DIR / "data" / "downloaded_papers.json"
)

PROCESSED_DIR = (
    BASE_DIR / "data" / "processed"
)

PDF_DIR = (
    BASE_DIR / "data" / "pdfs"
)

COLLECTION_NAME = "research_papers"

MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PDF_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================
# 2. CHECK WHETHER PAPER IS INDEXED
# ==========================================

def is_paper_indexed(paper_id, qdrant):

    paper_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="paper_id",
                match=models.MatchValue(
                    value=paper_id
                )
            )
        ]
    )

    count = qdrant.count(
        collection_name=COLLECTION_NAME,
        count_filter=paper_filter,
        exact=True
    ).count

    return count > 0


# ==========================================
# 3. VALIDATE PDF FILE
# ==========================================

def validate_pdf(pdf_path, is_upload=False):

    pdf_path = Path(pdf_path).resolve()

    if not pdf_path.is_file():
        raise FileNotFoundError(
            f"PDF file not found: {pdf_path}"
        )

    file_size = pdf_path.stat().st_size

    if file_size < 5:
        raise ValueError(
            "PDF file is missing or empty."
        )

    if is_upload:

        # Uploaded files must be inside
        # the application's PDF directory.

        allowed_directory = PDF_DIR.resolve()

        if not pdf_path.is_relative_to(
            allowed_directory
        ):
            raise ValueError(
                "Invalid uploaded PDF location."
            )

        if file_size > MAX_UPLOAD_SIZE:
            raise ValueError(
                "Uploaded PDF exceeds the 50 MB limit."
            )

    # Basic PDF format verification
    with open(pdf_path, "rb") as file:
        header = file.read(5)

    if header != b"%PDF-":
        raise ValueError(
            "File does not have a valid PDF header."
        )

    return pdf_path


# ==========================================
# 4. DOWNLOAD PDF WITH OPEN-ACCESS FALLBACK
# ==========================================

def download_with_fallback(paper):

    paper_id = paper["paper_id"]
    short_id = paper_id.rstrip("/").split("/")[-1]

    load_dotenv(
        BASE_DIR / ".env",
        override=True
    )

    urls = []

    def add_url(url):

        if (
            isinstance(url, str)
            and url.startswith(
                ("https://", "http://")
            )
            and url not in urls
        ):
            urls.append(url)

    # --------------------------------------
    # STEP 1: FIND OPEN-ACCESS SOURCES
    # --------------------------------------

    try:

        params = {}

        api_key = os.getenv(
            "OPENALEX_API_KEY"
        )

        if api_key:
            params["api_key"] = api_key

        response = requests.get(
            f"https://api.openalex.org/works/{short_id}",
            params=params,
            timeout=15
        )

        response.raise_for_status()

        work = response.json()

        locations = [
            work.get("best_oa_location")
        ] + (
            work.get("locations") or []
        )

        for location in locations:

            if not isinstance(location, dict):
                continue

            if not location.get("is_oa"):
                continue

            # Direct open-access PDF
            add_url(
                location.get("pdf_url")
            )

            landing_url = (
                location.get("landing_page_url")
                or ""
            )

            # Handle arXiv landing pages
            if "arxiv.org/abs/" in landing_url:

                arxiv_id = landing_url.split(
                    "/abs/", 1
                )[1].split("?")[0].split("#")[0]

                add_url(
                    f"https://arxiv.org/pdf/{arxiv_id}"
                )

    except requests.RequestException as e:

        print(
            "OpenAlex lookup failed:",
            e
        )

    # --------------------------------------
    # STEP 2: INCLUDE ORIGINAL PDF URL
    # --------------------------------------

    add_url(
        paper.get("pdf_url")
    )

    if not urls:
        raise RuntimeError(
            "No accessible PDF URLs were found. "
            "Please upload the paper manually."
        )

    errors = []

    # --------------------------------------
    # STEP 3: TRY AVAILABLE SOURCES
    # --------------------------------------

    for url in urls:

        print(
            "\nTrying PDF source:",
            url,
            flush=True
        )

        try:

            alternative_paper = {
                **paper,
                "pdf_url": url
            }

            downloaded_path = download_pdf(
                alternative_paper
            )

            pdf_path = validate_pdf(
                downloaded_path
            )

            print(
                "PDF download successful!",
                flush=True
            )

            return pdf_path, url

        except Exception as e:

            print(
                "Download failed:",
                e,
                flush=True
            )

            errors.append(
                f"{url}: {e}"
            )

    raise RuntimeError(
        "All PDF download sources failed. "
        "You can upload the paper manually.\n"
        + "\n".join(errors)
    )


# ==========================================
# 5. PROCESS AND INDEX ONE PAPER
# ==========================================

def process_and_index_paper(
    paper,
    embedding_model,
    qdrant,
    uploaded_pdf_path=None
):
    require_write_permission("Downloading and indexing research papers")

    paper_id = paper["paper_id"]

    short_id = (
        paper_id.rstrip("/").split("/")[-1]
    )

    print("\n" + "=" * 60)
    print("PROCESSING RESEARCH PAPER")
    print("=" * 60)

    print("Title:", paper["title"])
    print("Paper ID:", paper_id)

    # ======================================
    # STEP 1: CHECK EXISTING QDRANT INDEX
    # ======================================

    if is_paper_indexed(
        paper_id,
        qdrant
    ):

        print(
            "Paper already indexed. Skipping."
        )

        return {
            "title": paper["title"],
            "status": "already_indexed",
            "passages": 0
        }

    # ======================================
    # STEP 2: GET PDF
    # ======================================

    if uploaded_pdf_path is not None:

        # ----------------------------------
        # OPTION A: MANUALLY UPLOADED PDF
        # ----------------------------------

        print(
            "\nUsing manually uploaded PDF..."
        )

        pdf_path = validate_pdf(
            uploaded_pdf_path,
            is_upload=True
        )

        paper = {
            **paper,
            "source_type": "user_upload"
        }

        print(
            "Uploaded PDF verified:",
            pdf_path.name
        )

    else:

        # ----------------------------------
        # OPTION B: AUTOMATIC DOWNLOAD
        # ----------------------------------

        print(
            "\nDownloading research PDF..."
        )

        pdf_path, downloaded_url = (
            download_with_fallback(paper)
        )

        paper = {
            **paper,
            "pdf_url": downloaded_url,
            "source_type": "downloaded_pdf"
        }

        print(
            "Downloaded from:",
            downloaded_url
        )

    # ======================================
    # STEP 3: EXTRACT TEXT
    # ======================================

    print(
        "\nExtracting research paper text..."
    )

    text_path = (
        PROCESSED_DIR / f"{short_id}.txt"
    )

    chunks_path = (
        PROCESSED_DIR
        / f"{short_id}_chunks.json"
    )

    # Uploaded PDFs must be reprocessed
    # instead of using previously cached text.

    if uploaded_pdf_path is not None:

        text_path.unlink(
            missing_ok=True
        )

        chunks_path.unlink(
            missing_ok=True
        )

    if (
        not text_path.exists()
        or text_path.stat().st_size == 0
    ):

        parse_pdf(
            pdf_path,
            text_path
        )

    if (
        not text_path.exists()
        or text_path.stat().st_size == 0
    ):

        raise RuntimeError(
            "PDF text extraction failed."
        )

    full_text = text_path.read_text(
        encoding="utf-8"
    )

    if not full_text.strip():
        raise RuntimeError(
            "Extracted PDF text is empty."
        )

    pages = extract_pages(
        full_text
    )

    if not pages:
        raise RuntimeError(
            "No extractable text found in the PDF."
        )

    print(
        "Pages extracted:",
        len(pages)
    )

    # ======================================
    # STEP 4: CREATE RESEARCH CHUNKS
    # ======================================

    print(
        "\nCreating research chunks..."
    )

    chunks = create_chunks(
        paper_id,
        pages
    )

    if not chunks:
        raise RuntimeError(
            "No research chunks were created."
        )

    chunks_path.write_text(
        json.dumps(
            chunks,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print(
        "Research chunks created:",
        len(chunks)
    )

    # ======================================
    # STEP 5: UPDATE PAPER MANIFEST
    # ======================================

    print(
        "\nUpdating paper manifest..."
    )

    if MANIFEST_PATH.exists():

        manifest = json.loads(
            MANIFEST_PATH.read_text(
                encoding="utf-8"
            )
        )

    else:

        manifest = {}

    # Save metadata and actual local PDF path
    manifest[paper_id] = {
        **paper,
        "local_pdf": str(
            pdf_path.relative_to(BASE_DIR)
        )
    }

    MANIFEST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    MANIFEST_PATH.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print(
        "Manifest updated successfully."
    )

    # ======================================
    # STEP 6: GENERATE EMBEDDINGS AND INDEX
    # ======================================

    print(
        "\nGenerating embeddings "
        "and indexing in Qdrant..."
    )

    result = index_paper(
        short_id,
        embedding_model,
        qdrant
    )

    # ======================================
    # STEP 7: VERIFY QDRANT INDEX
    # ======================================

    if not is_paper_indexed(
        paper_id,
        qdrant
    ):

        raise RuntimeError(
            "Indexing completed but no vectors "
            "were found for this paper in Qdrant."
        )

    # ======================================
    # STEP 8: FINAL SUMMARY
    # ======================================

    print("\n" + "=" * 60)
    print("RESEARCH PAPER INDEXING COMPLETED")
    print("=" * 60)

    print(
        "Title:",
        paper["title"]
    )

    print(
        "Source:",
        paper["source_type"]
    )

    print(
        "Pages:",
        len(pages)
    )

    print(
        "Chunks:",
        len(chunks)
    )

    print(
        "Indexed passages:",
        result["passages"]
    )

    return {
        "title": paper["title"],
        "status": "indexed",
        "source_type": paper["source_type"],
        "pages": len(pages),
        "chunks": len(chunks),
        "passages": result["passages"],
        "vectors_after": result["vectors_after"],
        "local_pdf": str(
            pdf_path.relative_to(BASE_DIR)
        )
    }
