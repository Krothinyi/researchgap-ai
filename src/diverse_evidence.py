
from pathlib import Path
from collections import defaultdict

import numpy as np
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent.parent
QDRANT_PATH = BASE_DIR / "data" / "qdrant_storage"

COLLECTION = "research_papers"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ==========================================
# REUSABLE RETRIEVAL FUNCTION
# ==========================================

def retrieve_diverse_evidence(
    client,
    model,
    paper_id,
    question,
    section_keyword=None,
    top_k=4
):

    # Step 1: Retrieve vectors for the paper
    offset = None
    candidates = []

    paper_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="paper_id",
                match=models.MatchValue(value=paper_id)
            )
        ]
    )

    while True:

        points, offset = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=paper_filter,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=True
        )

        for point in points:

            payload = point.payload or {}
            section = payload.get("section") or "Unknown"

            if section_keyword:
                if section_keyword.lower() not in section.lower():
                    continue

            if point.vector is None or isinstance(point.vector, dict):
                continue

            candidates.append(point)

        if offset is None:
            break

    if not candidates:
        return []

    # Step 2: Embed the research question
    query_vector = model.encode(
        question,
        normalize_embeddings=True
    )

    # Step 3: Calculate similarity
    groups = defaultdict(list)

    for point in candidates:

        payload = point.payload or {}

        passage_vector = np.asarray(
            point.vector,
            dtype=np.float32
        )

        score = float(
            np.dot(query_vector, passage_vector)
            / (
                np.linalg.norm(query_vector)
                * np.linalg.norm(passage_vector)
                + 1e-12
            )
        )

        section = payload.get("section") or "Unknown"

        groups[section].append({
            "paper_id": payload.get("paper_id"),
            "title": payload.get("title"),
            "section": section,
            "embedding_id": payload.get("embedding_id"),
            "source_url": payload.get("source_url"),
            "page": payload.get("page"),
            "score": score,
            "text": payload.get("text", "")
        })

    # Step 4: Rank passages within each section
    for section in groups:
        groups[section].sort(
            key=lambda x: x["score"],
            reverse=True
        )

    # Step 5: Prioritize sections by their best result
    ordered_sections = sorted(
        groups.keys(),
        key=lambda s: groups[s][0]["score"],
        reverse=True
    )

    # Step 6: Round-robin evidence selection
    selected = []
    positions = {section: 0 for section in ordered_sections}

    while len(selected) < top_k:

        added = False

        for section in ordered_sections:

            index = positions[section]

            if index < len(groups[section]):

                selected.append(groups[section][index])
                positions[section] += 1
                added = True

                if len(selected) >= top_k:
                    break

        if not added:
            break

    return selected


# ==========================================
# TEST THE FUNCTION
# ==========================================

if __name__ == "__main__":

    print("=" * 60)
    print("REUSABLE DIVERSE EVIDENCE RETRIEVAL")
    print("=" * 60)

    client = QdrantClient(path=str(QDRANT_PATH))

    try:
        model = SentenceTransformer(MODEL_NAME)

        results = retrieve_diverse_evidence(
            client=client,
            model=model,
            paper_id="https://openalex.org/W4406241919",
            question=(
                "What are the main findings about "
                "multilingual LLM performance and evaluation?"
            ),
            section_keyword="performance analysis",
            top_k=4
        )

        print("\nEvidence retrieved:", len(results))

        for i, item in enumerate(results, 1):

            print("\n" + "=" * 60)
            print("RESULT", i)
            print("=" * 60)

            print("Section:", item["section"])
            print("Similarity:", round(item["score"], 4))
            print("Passage ID:", item["embedding_id"])

            print("\nEvidence:")
            print(item["text"][:600])

    finally:
        client.close()

    print("\nRETRIEVAL TEST COMPLETED")
