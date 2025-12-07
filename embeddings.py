
import json
import os
import uuid
from typing import List, Dict, Any, Optional

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# --------------------------- Configuration ---------------------------
MODEL_NAME = os.getenv("ST_MODEL", "all-MiniLM-L6-v2")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
DEFAULT_COLLECTION = os.getenv("QDRANT_COLLECTION", "github_code_review_comments_last_final")

# Lazily loaded model (to reuse across calls)
_model: Optional[SentenceTransformer] = None


def load_model(name: str = MODEL_NAME) -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(name)
    return _model


def get_client(host: str = QDRANT_HOST, port: int = QDRANT_PORT) -> QdrantClient:
    return QdrantClient(host=host, port=port, prefer_grpc=False)


def ensure_collection(client: QdrantClient, collection_name: str, vector_size: int,
                      distance: Distance = Distance.COSINE) -> None:
    """Create collection if it doesn't exist."""
    try:
        client.get_collection(collection_name=collection_name)
        return
    except Exception:
        # Create collection
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=distance),
        )


def embed_texts(texts: List[str], model: Optional[SentenceTransformer] = None) -> List[List[float]]:
    m = model or load_model()
    # Convert to numpy then to Python lists for Qdrant
    return m.encode(texts, convert_to_numpy=True).tolist()


def embed_code(code_snippet: str, model: Optional[SentenceTransformer] = None) -> List[float]:
    return embed_texts([code_snippet], model=model)[0]


def upsert_items(client: QdrantClient, collection_name: str, items: List[Dict[str, Any]],
                 model: Optional[SentenceTransformer] = None,
                 id_field: Optional[str] = None) -> int:
    """Upsert a list of items with fields like {comment, code, category, subcategory}.
    If id_field is provided and exists in item, it's used; otherwise generate UUIDs.
    Vector is computed from 'code' (fallback to 'comment' if 'code' missing).
    Returns number of points upserted.
    """
    m = model or load_model()
    vectors = []
    payloads = []
    ids = []

    for it in items:
        text = str(it.get("code") or it.get("comment") or "")
        vector = m.encode([text], convert_to_numpy=True)[0].tolist()
        vectors.append(vector)
        payloads.append({
            "comment": it.get("comment"),
            "code": it.get("code"),
            "category": it.get("category"),
            "subcategory": it.get("subcategory"),
        })
        if id_field and it.get(id_field) is not None:
            ids.append(it[id_field])
        else:
            ids.append(str(uuid.uuid4()))

    points = [PointStruct(id=pid, vector=v, payload=pl) for pid, v, pl in zip(ids, vectors, payloads)]
    client.upsert(collection_name=collection_name, points=points)
    return len(points)


def bulk_upsert_from_json(client: QdrantClient, collection_name: str, json_path: str,
                          model: Optional[SentenceTransformer] = None) -> int:
    """Load items from a JSON file and upsert into Qdrant.
    Supports both array JSON and line-delimited JSON (one JSON per line).
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            data = [json.loads(line) for line in f if line.strip()]

    if not isinstance(data, list):
        raise ValueError("Input JSON must be an array or line-delimited JSON items.")

    # Ensure collection exists with correct vector size based on model
    m = model or load_model()
    vector_size = m.get_sentence_embedding_dimension()
    ensure_collection(client, collection_name, vector_size)

    return upsert_items(client, collection_name, data, model=m)


def retrieve_similar_comments(code_snippet: str, top_k: int = 3,
                              host: str = QDRANT_HOST, port: int = QDRANT_PORT,
                              collection_name: str = DEFAULT_COLLECTION,
                              model: Optional[SentenceTransformer] = None) -> List[Dict[str, Any]]:
    client = get_client(host=host, port=port)
    m = model or load_model()
    vector = embed_code(code_snippet, model=m)
    results = client.search(
        collection_name=collection_name,
        query_vector=vector,
        limit=top_k,
        with_payload=True,
    )
    out = []
    for r in results:
        out.append({
            "comment": (r.payload or {}).get("comment"),
            "category": (r.payload or {}).get("category"),
            "subcategory": (r.payload or {}).get("subcategory"),
            "score": r.score,
        })
    return out
