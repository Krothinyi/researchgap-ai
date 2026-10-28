from src.demo_config import (
    MANIFEST_PATH,
    require_write_permission
)
import json
import re
from pathlib import Path

from qdrant_client import models


# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent


PDF_DIR = BASE_DIR / "data" / "pdfs"

PROCESSED_DIR = BASE_DIR / "data" / "processed"

COLLECTION_NAME = "research_papers"


# ==========================================
# 2. LOAD PAPER MANIFEST
# ==========================================

def load_manifest():

    if not MANIFEST_PATH.exists():
        return {}

    return json.loads(
        MANIFEST_PATH.read_text(encoding="utf-8")
    )


# ==========================================
# 3. DELETE ONE INDEXED PAPER
# ==========================================

def delete_indexed_paper(paper_id, qdrant):

    require_write_permission("Deleting indexed papers")

    manifest = load_manifest()

    if paper_id not in manifest:
        raise ValueError(
            "Paper not found in the local manifest."
        )

    short_id = paper_id.rstrip("/").split("/")[-1]

    # Prevent unsafe file paths.
    if not re.fullmatch(r"W[0-9]+", short_id):
        raise ValueError("Unsupported paper ID format.")

    paper = manifest[paper_id]

    # ======================================
    # STEP 1: IDENTIFY QDRANT VECTORS
    # ======================================

    paper_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="paper_id",
                match=models.MatchValue(value=paper_id)
            )
        ]
    )

    vectors_before = qdrant.count(
        collection_name=COLLECTION_NAME,
        count_filter=paper_filter,
        exact=True
    ).count

    # ======================================
    # STEP 2: DELETE ONLY THIS PAPER'S VECTORS
    # ======================================

    if vectors_before > 0:

        qdrant.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=paper_filter
            ),
            wait=True
        )

    vectors_remaining = qdrant.count(
        collection_name=COLLECTION_NAME,
        count_filter=paper_filter,
        exact=True
    ).count

    if vectors_remaining != 0:
        raise RuntimeError(
            "Some vectors remain. Deletion was incomplete."
        )

    # ======================================
    # STEP 3: REMOVE MANIFEST ENTRY
    # ======================================

    manifest.pop(paper_id)

    temporary_manifest = (
        MANIFEST_PATH.parent /
        "downloaded_papers.json.tmp"
    )

    temporary_manifest.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    temporary_manifest.replace(MANIFEST_PATH)

    # ======================================
    # STEP 4: DELETE PROCESSED FILES
    # ======================================

    deleted_files = []
    warnings = []

    processed_files = [
        PROCESSED_DIR / f"{short_id}.txt",
        PROCESSED_DIR / f"{short_id}_chunks.json"
    ]

    for file_path in processed_files:

        try:
            if file_path.is_file():
                file_path.unlink()
                deleted_files.append(str(file_path))

        except OSError as e:
            warnings.append(
                f"Could not delete {file_path.name}: {e}"
            )

    # ======================================
    # STEP 5: DELETE LOCAL PDF SAFELY
    # ======================================

    local_pdf = paper.get("local_pdf")

    if local_pdf:

        pdf_path = Path(local_pdf)

        if not pdf_path.is_absolute():
            pdf_path = BASE_DIR / pdf_path

        pdf_path = pdf_path.resolve()

        # Do not delete a PDF still referenced
        # by another paper in the manifest.
        referenced_elsewhere = False

        for other_paper in manifest.values():

            other_pdf = other_paper.get("local_pdf")

            if not other_pdf:
                continue

            other_path = Path(other_pdf)

            if not other_path.is_absolute():
                other_path = BASE_DIR / other_path

            if other_path.resolve() == pdf_path:
                referenced_elsewhere = True
                break

        # Delete only PDFs stored in the
        # application's managed PDF directory.
        if (
            pdf_path.is_relative_to(PDF_DIR.resolve())
            and not referenced_elsewhere
        ):

            try:
                if pdf_path.is_file():
                    pdf_path.unlink()
                    deleted_files.append(str(pdf_path))

            except OSError as e:
                warnings.append(
                    f"Could not delete PDF: {e}"
                )

        else:
            warnings.append(
                "PDF retained because it is shared "
                "or outside the managed PDF directory."
            )

    # ======================================
    # STEP 6: RETURN SUMMARY
    # ======================================

    return {
        "title": paper["title"],
        "paper_id": paper_id,
        "deleted_vectors": vectors_before,
        "deleted_files": deleted_files,
        "warnings": warnings,
        "status": "deleted"
    }
