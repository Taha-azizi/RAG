from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

def get_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = None) -> SentenceTransformer:
    """Create and return a SentenceTransformer embedder."""
    return SentenceTransformer(model_name, device=device)

def embed_texts(texts: List[str], embedder: SentenceTransformer, batch_size: int = 32, show_progress_bar: bool = True, normalize: bool = True) -> np.ndarray:
    """Compute embeddings for a list of texts and return a numpy array."""
    return embedder.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=show_progress_bar,
        normalize_embeddings=normalize,
    )