from src.demo_config import require_write_permission
import json
import re
import tempfile
import uuid
from pathlib import Path

from qdrant_client import models

from src.batch_paper_pipeline import validate_pdf
from src.chunk_dynamic_papers import (
    create_chunks,
    extract_pages
)
from src.index_single_paper import index_paper
from src.parse_dynamic_papers import parse_pdf


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"
PROCESSED_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

MANIFEST_PATH = (
    DATA_DIR / "downloaded_papers.json"
)

COLLECTION_NAME = "research_papers"


# ==========================================
# 2. ATOMIC FILE WRITING
# ==========================================

def _write_bytes_atomically(path, contents):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    temp_path = path.with_name(
        f".{path.name}.{uuid.uuid4().hex}.tmp"
    )

    try:

        temp_path.write_bytes(contents)
        temp_path.replace(path)

    finally:

        temp_path.unlink(missing_ok=True)


# ==========================================
# 3. CREATE PAPER-SPECIFIC QDRANT FILTER
# ==========================================

def _paper_filter(paper_id):

    return models.Filter(
        must=[
            models.FieldCondition(
                key="paper_id",
                match=models.MatchValue(
                    value=paper_id
                )
            )
        ]
    )


# ==========================================
# 4. BACK UP EXISTING QDRANT VECTORS
# ==========================================

def _get_paper_points(qdrant, paper_id):

    found = []
    offset = None

    while True:

        page, offset = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=_paper_filter(paper_id),
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=True
        )

        for record in page:

            if record.vector is None:
                raise RuntimeError(
                    "Unable to back up old Qdrant vectors."
                )

            found.append(
                models.PointStruct(
                    id=record.id,
                    vector=record.vector,
                    payload=record.payload or {}
                )
            )

        if offset is None:
            break

    return found


# ==========================================
# 5. DELETE VECTORS FOR ONE PAPER
# ==========================================

def _delete_paper_vectors(qdrant, paper_id):

    qdrant.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.FilterSelector(
            filter=_paper_filter(paper_id)
        ),
        wait=True
    )


# ==========================================
# 6. RESTORE BACKED-UP FILES
# ==========================================

def _restore_files(backups):

    for path, content in backups.items():

        if content is None:

            path.unlink(missing_ok=True)

        else:

            _write_bytes_atomically(
                path,
                content
            )


# ==========================================
# 7. REPLACE AN INDEXED RESEARCH PAPER
# ==========================================

def replace_indexed_paper(
    paper_id,
    uploaded_pdf_path,
    embedding_model,
    qdrant
):

    require_write_permission("Replacing indexed papers")

    # ======================================
    # STEP 1: VERIFY EXISTING PAPER
    # ======================================

    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            "Paper manifest does not exist."
        )

    manifest_bytes = MANIFEST_PATH.read_bytes()

    manifest = json.loads(manifest_bytes)

    if paper_id not in manifest:
        raise ValueError(
            "Selected paper not found in manifest."
        )

    short_id = (
        paper_id.rstrip("/").split("/")[-1]
    )

    if not re.fullmatch(r"W\d+", short_id):
        raise ValueError(
            "Expected a valid OpenAlex paper ID."
        )

    old_paper = manifest[paper_id]

    # ======================================
    # STEP 2: VALIDATE REPLACEMENT PDF
    # ======================================

    pdf_path = validate_pdf(
        uploaded_pdf_path,
        is_upload=True
    )

    print("\n" + "=" * 60)
    print("REPLACING RESEARCH PAPER")
    print("=" * 60)

    print("Paper:", old_paper["title"])
    print("Replacement PDF:", pdf_path.name)

    # ======================================
    # STEP 3: PREPARE DIRECTORIES
    # ======================================

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    EMBEDDINGS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    text_path = (
        PROCESSED_DIR / f"{short_id}.txt"
    )

    chunks_path = (
        PROCESSED_DIR / f"{short_id}_chunks.json"
    )

    vector_path = (
        EMBEDDINGS_DIR /
        f"{short_id}_embeddings.npy"
    )

    metadata_path = (
        EMBEDDINGS_DIR /
        f"{short_id}_embedding_chunks.json"
    )

    managed_files = (
        text_path,
        chunks_path,
        vector_path,
        metadata_path
    )

    # ======================================
    # STEP 4: VALIDATE AND PREPARE NEW CONTENT
    # ======================================

    # Extract the replacement text before
    # modifying the existing paper.

    with tempfile.TemporaryDirectory(
        prefix=f"replace_{short_id}_",
        dir=DATA_DIR
    ) as tmp:

        stage = Path(tmp)

        staged_text = (
            stage / "replacement.txt"
        )

        print(
            "\nExtracting replacement PDF..."
        )

        parse_pdf(
            pdf_path,
            staged_text
        )

        if not staged_text.is_file():
            raise ValueError(
                "Replacement PDF extraction failed."
            )

        full_text = staged_text.read_text(
            encoding="utf-8"
        )

        if not full_text.strip():
            raise ValueError(
                "Replacement PDF contains no extractable text."
            )

        pages = extract_pages(full_text)

        if not pages:
            raise ValueError(
                "Could not extract pages from replacement PDF."
            )

        chunks = create_chunks(
            paper_id,
            pages
        )

        if not chunks:
            raise ValueError(
                "Could not create research chunks."
            )

        staged_chunks = json.dumps(
            chunks,
            indent=2,
            ensure_ascii=False
        ).encode("utf-8")

        print("Pages:", len(pages))
        print("Chunks:", len(chunks))

        # ==================================
        # STEP 5: BACK UP EXISTING FILES
        # ==================================

        print(
            "\nBacking up existing paper data..."
        )

        file_backups = {
            path: (
                path.read_bytes()
                if path.is_file()
                else None
            )
            for path in managed_files
        }

        # ==================================
        # STEP 6: BACK UP OLD QDRANT VECTORS
        # ==================================

        old_points = _get_paper_points(
            qdrant,
            paper_id
        )

        old_count = len(old_points)

        print(
            "Existing Qdrant vectors:",
            old_count
        )

        # ==================================
        # STEP 7: PREPARE UPDATED MANIFEST
        # ==================================

        new_manifest = dict(manifest)

        new_manifest[paper_id] = {
            **old_paper,
            "source_type": "user_upload",
            "local_pdf": str(
                pdf_path.relative_to(BASE_DIR)
            )
        }

        new_manifest_bytes = json.dumps(
            new_manifest,
            indent=2,
            ensure_ascii=False
        ).encode("utf-8")

        # ==================================
        # STEP 8: PERFORM REPLACEMENT
        # ==================================

        try:

            print(
                "\nWriting replacement text and chunks..."
            )

            _write_bytes_atomically(
                text_path,
                staged_text.read_bytes()
            )

            _write_bytes_atomically(
                chunks_path,
                staged_chunks
            )

            _write_bytes_atomically(
                MANIFEST_PATH,
                new_manifest_bytes
            )

            # ------------------------------
            # DELETE OLD PAPER VECTORS
            # ------------------------------

            if old_count > 0:

                print(
                    "\nRemoving old Qdrant vectors..."
                )

                _delete_paper_vectors(
                    qdrant,
                    paper_id
                )

            # ------------------------------
            # GENERATE NEW EMBEDDINGS
            # ------------------------------

            print(
                "\nGenerating replacement embeddings..."
            )

            result = index_paper(
                short_id,
                embedding_model,
                qdrant
            )

            # ------------------------------
            # VERIFY NEW QDRANT VECTORS
            # ------------------------------

            new_count = qdrant.count(
                collection_name=COLLECTION_NAME,
                count_filter=_paper_filter(paper_id),
                exact=True
            ).count

            if (
                new_count == 0
                or new_count != result["passages"]
            ):

                raise RuntimeError(
                    "Qdrant verification failed. "
                    f"Expected {result['passages']} "
                    f"passages, found {new_count}."
                )

            print(
                "\nReplacement indexing successful!"
            )

        # ==================================
        # STEP 9: ROLLBACK ON FAILURE
        # ==================================

        except Exception as original_error:

            print(
                "\nReplacement failed. "
                "Attempting to restore original data..."
            )

            rollback_errors = []

            # Restore Qdrant vectors
            try:

                _delete_paper_vectors(
                    qdrant,
                    paper_id
                )

                for pos in range(
                    0,
                    len(old_points),
                    64
                ):

                    qdrant.upsert(
                        collection_name=COLLECTION_NAME,
                        points=old_points[
                            pos:pos + 64
                        ],
                        wait=True
                    )

            except Exception as exc:

                rollback_errors.append(
                    f"Qdrant restore failed: {exc}"
                )

            # Restore files and manifest
            try:

                _restore_files(
                    file_backups
                )

                _write_bytes_atomically(
                    MANIFEST_PATH,
                    manifest_bytes
                )

            except Exception as exc:

                rollback_errors.append(
                    f"File restore failed: {exc}"
                )

            if rollback_errors:

                raise RuntimeError(
                    f"PDF replacement failed: "
                    f"{original_error}. "
                    "Manual recovery required: "
                    + "; ".join(rollback_errors)
                ) from original_error

            raise RuntimeError(
                "PDF replacement failed. "
                "The original paper data was restored. "
                f"Cause: {original_error}"
            ) from original_error

    # ======================================
    # STEP 10: CLEAN UP PREVIOUS PDF
    # ======================================

    warnings = []

    old_location = old_paper.get(
        "local_pdf"
    )

    if old_location:

        old_path = Path(old_location)

        if not old_path.is_absolute():
            old_path = BASE_DIR / old_path

        old_path = old_path.resolve()

        # Check if another paper references
        # the original PDF.
        referenced_elsewhere = False

        for other_id, other in new_manifest.items():

            if other_id == paper_id:
                continue

            other_location = other.get(
                "local_pdf"
            )

            if not other_location:
                continue

            other_path = Path(
                other_location
            )

            if not other_path.is_absolute():
                other_path = BASE_DIR / other_path

            if other_path.resolve() == old_path:
                referenced_elsewhere = True
                break

        # Only delete an old application-managed
        # PDF that is no longer referenced.
        if (
            old_path != pdf_path
            and old_path.is_relative_to(
                PDF_DIR.resolve()
            )
            and not referenced_elsewhere
        ):

            try:

                old_path.unlink(
                    missing_ok=True
                )

            except OSError as exc:

                warnings.append(
                    f"Old PDF retained: {exc}"
                )

    # ======================================
    # STEP 11: RETURN REPLACEMENT SUMMARY
    # ======================================

    print("\n" + "=" * 60)
    print("RESEARCH PDF REPLACEMENT COMPLETED")
    print("=" * 60)

    print("Paper:", old_paper["title"])
    print("Old vectors:", old_count)
    print("New vectors:", new_count)

    return {
        "status": "replaced",
        "paper_id": paper_id,
        "title": old_paper["title"],
        "old_vectors": old_count,
        "new_vectors": new_count,
        "pages": len(pages),
        "chunks": len(chunks),
        "warnings": warnings
    }
