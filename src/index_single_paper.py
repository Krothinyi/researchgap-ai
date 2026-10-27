
import json
import sys
import uuid
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

MANIFEST_PATH = BASE_DIR / "data" / "downloaded_papers.json"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
EMBEDDINGS_DIR = BASE_DIR / "data" / "embeddings"
QDRANT_PATH = BASE_DIR / "data" / "qdrant_storage"

COLLECTION_NAME = "research_papers"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

MAX_TOKENS = 220
MAX_WORDS = 140
OVERLAP_WORDS = 20
BATCH_SIZE = 32

EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. SPLIT CHUNKS INTO SAFE PASSAGES
# ==========================================

def split_passages(text, tokenizer, token_limit):

    words = text.split()
    passages = []
    start = 0

    while start < len(words):

        end = min(start + MAX_WORDS, len(words))

        # Ensure the passage fits the model's token limit
        while end > start:

            candidate = " ".join(words[start:end])

            token_count = (
    len(tokenizer.tokenize(candidate))
    + tokenizer.num_special_tokens_to_add(pair=False)
)

            if token_count <= token_limit:
                break

            end -= 1

        if end == start:
            raise ValueError(
                "A word exceeds the embedding token limit."
            )

        passages.append(" ".join(words[start:end]))

        if end >= len(words):
            break

        start = max(start + 1, end - OVERLAP_WORDS)

    return passages


# ==========================================
# 3. INDEX ONE PAPER
# ==========================================

def index_paper(short_id, embedding_model, qdrant):

    manifest = json.loads(
        MANIFEST_PATH.read_text(encoding="utf-8")
    )

    paper = None
    paper_id = None

    for full_id, info in manifest.items():
        if full_id.split("/")[-1] == short_id:
            paper_id = full_id
            paper = info
            break

    if paper is None:
        raise ValueError(
            f"Paper {short_id} not found in manifest."
        )

    chunks_path = (
        PROCESSED_DIR / f"{short_id}_chunks.json"
    )

    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Chunks file missing: {chunks_path}"
        )

    chunks = json.loads(
        chunks_path.read_text(encoding="utf-8")
    )

    tokenizer = embedding_model.tokenizer

    token_limit = min(
        MAX_TOKENS,
        embedding_model.max_seq_length
    )

    print("\nPaper:", paper["title"])
    print("Original chunks:", len(chunks))

    # ======================================
    # 4. PREPARE PASSAGES
    # ======================================

    passages = []

    for chunk in chunks:

        smaller_passages = split_passages(
            chunk["text"],
            tokenizer,
            token_limit
        )

        for passage in smaller_passages:

            embedding_id = (
                f"{short_id}_e{len(passages) + 1:05d}"
            )

            passages.append({
                "embedding_id": embedding_id,
                "source_chunk_id": chunk["chunk_id"],
                "paper_id": paper_id,
                "title": paper["title"],
                "year": paper.get("year"),
                "pdf_url": paper.get("pdf_url"),
                "page": chunk["page"],
                "text": passage
            })

    if not passages:
        raise ValueError("No passages available to index.")

    print("Embedding passages:", len(passages))

    # ======================================
    # 5. GENERATE EMBEDDINGS
    # ======================================

    print("Generating embeddings...")

    vectors = embedding_model.encode(
        [item["text"] for item in passages],
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    vectors = np.asarray(vectors, dtype=np.float32)

    print("Embedding shape:", vectors.shape)

    # ======================================
    # 6. SAVE VECTORS AND METADATA
    # ======================================

    np.save(
        EMBEDDINGS_DIR / f"{short_id}_embeddings.npy",
        vectors
    )

    (
        EMBEDDINGS_DIR
        / f"{short_id}_embedding_chunks.json"
    ).write_text(
        json.dumps(passages, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    # ======================================
    # 7. VERIFY QDRANT COLLECTION
    # ======================================

    if not qdrant.collection_exists(COLLECTION_NAME):
        raise RuntimeError(
            "Existing Qdrant collection not found."
        )

    collection = qdrant.get_collection(COLLECTION_NAME)

    dimension = collection.config.params.vectors.size

    if vectors.shape[1] != dimension:
        raise ValueError("Embedding dimension mismatch.")

    count_before = qdrant.count(
        collection_name=COLLECTION_NAME,
        exact=True
    ).count

    # ======================================
    # 8. UPSERT VECTORS
    # ======================================

    print("Indexing passages in Qdrant...")

    for start in range(0, len(vectors), BATCH_SIZE):

        end = min(start + BATCH_SIZE, len(vectors))
        points = []

        for i in range(start, end):

            item = passages[i]

            stable_key = (
                f"{item['paper_id']}:{item['embedding_id']}"
            )

            point_id = str(
                uuid.uuid5(uuid.NAMESPACE_URL, stable_key)
            )

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vectors[i].tolist(),
                    payload=item
                )
            )

        qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True
        )

    count_after = qdrant.count(
        collection_name=COLLECTION_NAME,
        exact=True
    ).count

    # ======================================
    # 9. PRINT RESULTS
    # ======================================

    print("\n" + "=" * 60)
    print("PAPER INDEXING COMPLETED")
    print("=" * 60)

    print("Paper:", paper["title"])
    print("Passages processed:", len(passages))
    print("Vectors before:", count_before)
    print("Vectors after:", count_after)
    print("Net increase:", count_after - count_before)

    return {
        "paper": paper["title"],
        "passages": len(passages),
        "vectors_before": count_before,
        "vectors_after": count_after
    }


# ==========================================
# 10. RUN FROM TERMINAL
# ==========================================

if __name__ == "__main__":

    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python src/index_single_paper.py PAPER_ID"
        )

    short_id = sys.argv[1]

    print("Loading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    print("Connecting to Qdrant...")

    client = QdrantClient(path=str(QDRANT_PATH))

    try:
        index_paper(short_id, model, client)
    finally:
        client.close()
