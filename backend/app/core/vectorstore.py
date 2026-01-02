# backend/app/core/vectorstore.py
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any
import re

# Use PersistentClient for on-disk persistence with Chroma 0.5.x
client = chromadb.PersistentClient(path="./chroma_db")

# Use a local SentenceTransformer embedding function for automatic embedding
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = client.get_or_create_collection(
    "agentic_tutor",
    embedding_function=sentence_transformer_ef
)

# Separate collection for taught summaries (long-term memory, small texts)
taught_summaries = client.get_or_create_collection(
    "taught_summaries",
    embedding_function=sentence_transformer_ef
)

# -------------------- Lightweight BM25 Support --------------------
try:
    from rank_bm25 import BM25Okapi  # type: ignore
    _HAS_BM25 = True
except Exception:
    BM25Okapi = None  # type: ignore
    _HAS_BM25 = False

_bm25_index = None
_bm25_tokens: List[List[str]] = []
_bm25_docs: List[str] = []
_bm25_metas: List[Dict[str, Any]] = []
_bm25_ids: List[str] = []

_token_pattern = re.compile(r"\w+")

def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _token_pattern.findall(text or "")] 

def _build_bm25_index():
    global _bm25_index, _bm25_tokens, _bm25_docs, _bm25_metas, _bm25_ids
    if not _HAS_BM25:
        return
    try:
        data = collection.get(include=["documents", "metadatas", "ids"])  # may be large; acceptable for MVP
        _bm25_docs = data.get("documents", []) or []
        _bm25_metas = data.get("metadatas", []) or []
        _bm25_ids = data.get("ids", []) or []
        _bm25_tokens = [_tokenize(doc) for doc in _bm25_docs]
        _bm25_index = BM25Okapi(_bm25_tokens) if _bm25_tokens else None
    except Exception:
        _bm25_index = None

def _reset_bm25_cache():
    global _bm25_index, _bm25_tokens, _bm25_docs, _bm25_metas, _bm25_ids
    _bm25_index = None
    _bm25_tokens = []
    _bm25_docs = []
    _bm25_metas = []
    _bm25_ids = []

def _bm25_query(query: str, k: int) -> Dict[str, List[List[Any]]]:
    """Return results in Chroma-like shape using BM25 only."""
    if not (_HAS_BM25 and _bm25_index and _bm25_docs):
        return {"documents": [[]], "metadatas": [[]], "ids": [[]], "distances": [[]]}
    q_tokens = _tokenize(query)
    scores = _bm25_index.get_scores(q_tokens)
    idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    documents = [[_bm25_docs[i] for i in idxs]]
    metadatas = [[_bm25_metas[i] for i in idxs]]
    ids = [[_bm25_ids[i] for i in idxs]]
    # Convert to distances-like values (lower is better)
    if idxs:
        s_max = max(scores[i] for i in idxs) or 1.0
        s_min = min(scores[i] for i in idxs)
        norm = [(scores[i] - s_min) / (s_max - s_min + 1e-6) for i in idxs]
        distances = [[1.0 - s for s in norm]]
    else:
        distances = [[]]
    return {"documents": documents, "metadatas": metadatas, "ids": ids, "distances": distances}

def upsert_chunks(chunks):
    # chunks: list of dict {id, text, metadata}
    ids = [c["id"] for c in chunks]
    texts = [c["text"] for c in chunks]
    metadatas = [c.get("metadata", {}) for c in chunks]
    collection.add(ids=ids, documents=texts, metadatas=metadatas)
    # Rebuild BM25 index after each upsert (simple but effective)
    _build_bm25_index()

def query_top_k(query: str, k: int = 5):
    results = collection.query(query_texts=[query], n_results=k)
    return results

def upsert_taught_summary(_id: str, text: str, metadata: Dict[str, Any]):
    taught_summaries.add(ids=[_id], documents=[text], metadatas=[metadata])

def query_taught_summaries(query: str, k: int = 3):
    return taught_summaries.query(query_texts=[query], n_results=k)

def query_with_topic_filter(query: str, topic: str = None, k: int = 7):
    """
    Simple semantic search with optional topic boost.
    No rigid filtering - just better retrieval.
    """
    # Simple approach: just do semantic search with higher k
    # The topic is already in the query, so semantic search will find relevant chunks
    print(f"[VectorStore] Semantic search for: {query[:100]}...")
    return query_top_k(query, k)

def query_hybrid(query: str, k: int = 7, alpha: float = 0.5) -> Dict[str, List[List[Any]]]:
    """
    Hybrid search combining vector similarity and BM25 lexical relevance.
    alpha is the weight for vector score (0..1). (1-alpha) for BM25.
    Returns a Chroma-like dict.
    """
    # Vector results
    vec = query_top_k(query, k=k)
    v_docs = vec.get("documents", [[]])[0] if vec.get("documents") else []
    v_meta = vec.get("metadatas", [[]])[0] if vec.get("metadatas") else []
    v_ids = vec.get("ids", [[]])[0] if vec.get("ids") else []
    v_dist = vec.get("distances", [[]])[0] if vec.get("distances") else []
    # Convert distances (lower better) to scores [0,1]
    if v_dist:
        d_max = max(v_dist)
        d_min = min(v_dist)
        v_scores = [1.0 - ((d - d_min) / (d_max - d_min + 1e-6)) for d in v_dist]
    else:
        v_scores = [0.0] * len(v_docs)

    # BM25 results
    bm = _bm25_query(query, k=k)
    b_docs = bm.get("documents", [[]])[0] if bm.get("documents") else []
    b_meta = bm.get("metadatas", [[]])[0] if bm.get("metadatas") else []
    b_ids = bm.get("ids", [[]])[0] if bm.get("ids") else []
    b_dist = bm.get("distances", [[]])[0] if bm.get("distances") else []
    if b_dist:
        bd_max = max(b_dist)
        bd_min = min(b_dist)
        b_scores = [1.0 - ((d - bd_min) / (bd_max - bd_min + 1e-6)) for d in b_dist]
    else:
        b_scores = [0.0] * len(b_docs)

    # Merge by id (fallback to doc text prefix as key)
    combined: Dict[str, Dict[str, Any]] = {}
    for doc, meta, _id, score in zip(v_docs, v_meta, v_ids, v_scores):
        key = _id or (doc[:64] if isinstance(doc, str) else str(meta))
        combined[key] = {"doc": doc, "meta": meta, "id": _id, "v": score, "b": 0.0}
    for doc, meta, _id, score in zip(b_docs, b_meta, b_ids, b_scores):
        key = _id or (doc[:64] if isinstance(doc, str) else str(meta))
        if key in combined:
            combined[key]["b"] = max(combined[key]["b"], score)
        else:
            combined[key] = {"doc": doc, "meta": meta, "id": _id, "v": 0.0, "b": score}

    ranked = sorted(
        combined.values(),
        key=lambda x: (alpha * x["v"]) + ((1 - alpha) * x["b"]),
        reverse=True
    )[:k]

    documents = [[r["doc"] for r in ranked]]
    metadatas = [[r["meta"] for r in ranked]]
    ids = [[r["id"] for r in ranked]]
    scores = [(alpha * r["v"]) + ((1 - alpha) * r["b"]) for r in ranked]
    if scores:
        s_max = max(scores)
        s_min = min(scores)
        distances = [[1.0 - ((s - s_min) / (s_max - s_min + 1e-6)) for s in scores]]
    else:
        distances = [[]]

    return {"documents": documents, "metadatas": metadatas, "ids": ids, "distances": distances}

def get_collection_stats():
    """Get statistics about the vector collection"""
    count = collection.count()
    return {
        "total_chunks": count,
        "collection_name": collection.name
    }

def clear_taught_summaries():
    """Clear only the summaries collection (cross-session memory). Preserve document index."""
    try:
        taught_summaries.delete(where={})
    except Exception:
        # Fallback: drop and recreate summaries collection only
        try:
            client.delete_collection("taught_summaries")
        except Exception:
            pass
        globals()["taught_summaries"] = client.get_or_create_collection(
            "taught_summaries",
            embedding_function=sentence_transformer_ef
        )

# Create a simple vectorstore class for easier imports
class VectorStore:
    def __init__(self):
        self.collection = collection
        self.taught_summaries = taught_summaries
    
    def upsert_chunks(self, chunks):
        return upsert_chunks(chunks)
    
    def query_top_k(self, query: str, k: int = 5):
        return query_top_k(query, k)
    
    def query_with_topic_filter(self, query: str, topic: str = None, k: int = 5):
        """Topic-aware retrieval: vector search + section filter"""
        return query_with_topic_filter(query, topic, k)
    
    def query_hybrid(self, query: str, k: int = 7, alpha: float = 0.5):
        return query_hybrid(query, k=k, alpha=alpha)
    
    def get_stats(self):
        return get_collection_stats()

    # Long-term memory APIs
    def upsert_taught_summary(self, _id: str, text: str, metadata: Dict[str, Any]):
        return upsert_taught_summary(_id, text, metadata)

    def query_taught_summaries(self, query: str, k: int = 3):
        return query_taught_summaries(query, k)

# Global instance
vectorstore = VectorStore()

# Build BM25 index at import (best-effort)
_build_bm25_index()
