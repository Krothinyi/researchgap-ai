
import json
from pathlib import Path

from qdrant_client import models

from src.demo_config import (
    BASE_DIR,
    DATA_DIR,
    MANIFEST_PATH,
    DEMO_MODE,
)


SEED_DIR = BASE_DIR / "demo_seed"

EXPECTED_PAPERS = {
    "https://openalex.org/W4411120331",
    "https://openalex.org/W4410761148",
}


def initialize_demo_library(qdrant, collection_name):
    """
    Restore approved example papers in demo mode.

    Uses the existing Qdrant client and does not
    modify the normal research database.
    """

    if not DEMO_MODE:
        return None

    manifest_file = SEED_DIR / "demo_manifest.json"
    vectors_file = SEED_DIR / "demo_vectors.json"

    if not manifest_file.is_file():
        raise FileNotFoundError(manifest_file)

    if not vectors_file.is_file():
        raise FileNotFoundError(vectors_file)

    # -------------------------------------
    # 1. LOAD DEMO DATA
    # -------------------------------------

    manifest = json.loads(
        manifest_file.read_text(encoding="utf-8")
    )

    vectors = json.loads(
        vectors_file.read_text(encoding="utf-8")
    )

    papers = (
        manifest
        if isinstance(manifest, list)
        else list(manifest.values())
    )

    paper_ids = {
        paper["paper_id"]
        for paper in papers
    }

    if paper_ids != EXPECTED_PAPERS:
        raise ValueError(
            "Demo manifest contains unexpected papers."
        )

    if len(vectors) != 160:
        raise ValueError(
            "Unexpected number of demo vectors."
        )

    # -------------------------------------
    # 2. VALIDATE EXPORTED VECTORS
    # -------------------------------------

    point_ids = []

    for point in vectors:

        if point["payload"]["paper_id"] not in EXPECTED_PAPERS:
            raise ValueError("Unexpected paper in demo vectors.")

        if len(point["vector"]) != 384:
            raise ValueError("Invalid embedding dimensions.")

        point_ids.append(point["id"])

    if len(set(point_ids)) != len(point_ids):
        raise ValueError("Duplicate demo vector IDs.")

    # -------------------------------------
    # 3. RESTORE MISSING VECTORS
    # -------------------------------------

    existing = qdrant.retrieve(
        collection_name=collection_name,
        ids=point_ids,
        with_payload=False,
        with_vectors=False,
    )

    existing_ids = {
        str(point.id)
        for point in existing
    }

    missing = [
        point
        for point in vectors
        if str(point["id"]) not in existing_ids
    ]

    for start in range(0, len(missing), 32):

        batch = missing[start:start + 32]

        qdrant.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=point["id"],
                    vector=point["vector"],
                    payload=point["payload"],
                )
                for point in batch
            ],
            wait=True,
        )

    # -------------------------------------
    # 4. RESTORE PAPER METADATA
    # -------------------------------------

    # Remove machine-specific PDF paths.
    if isinstance(manifest, list):
        clean_manifest = [
            {
                key: value
                for key, value in paper.items()
                if key != "local_pdf"
            }
            for paper in manifest
        ]
    else:
        clean_manifest = {
            key: {
                field: value
                for field, value in paper.items()
                if field != "local_pdf"
            }
            for key, paper in manifest.items()
        }

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    MANIFEST_PATH.write_text(
        json.dumps(
            clean_manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "papers": len(papers),
        "vectors": len(vectors),
        "restored": len(missing),
    }
