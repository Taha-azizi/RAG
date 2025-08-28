from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

def get_embedder(model_name="sentence-transformers/all-MiniLM-L6-v2") -> SentenceTransformer:
    """Create and return a SentenceTransformer embedder."""
    return SentenceTransformer(model_name)

def embed_texts(texts: List[str], embedder: SentenceTransformer, batch_size: int = 32, show_progress_bar: bool = True, normalize: bool = True) -> np.ndarray:
    """Compute embeddings for a list of texts and return a numpy array."""
    if hasattr(embedder, 'encode'):
        return embedder.encode(texts)
    else:
        raise ValueError(f"embedder object of type {type(embedder)} has no 'encode' method")
    return embedder.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=show_progress_bar,
        normalize_embeddings=normalize,
    )

def embed_chunks(chunks: List[str], embedder: SentenceTransformer) -> np.ndarray:
    """Convenience wrapper to embed a list of text chunks."""
    return embed_texts(chunks, embedder)

def semantic_search(query: str, top_k: int, chunks: List[str], embedder: SentenceTransformer, chunk_embeddings: np.ndarray):
    """Perform semantic search given precomputed embeddings."""
    query_emb = embed_texts([query], embedder)[0]
    scores = np.dot(chunk_embeddings, query_emb)
    ranked_ids = np.argsort(scores)[::-1][:top_k]
    return [chunks[i] for i in ranked_ids], [scores[i] for i in ranked_ids]
