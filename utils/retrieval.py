import re
from typing import List, Tuple
from store import query_collection

def keyword_search(chunks: List[str], query: str, top_k: int = 3) -> List[Tuple[str, float]]:
    """
    Very simple keyword search: rank chunks by keyword frequency.
    """
    query_terms = query.lower().split()
    scored = []
    for chunk in chunks:
        score = sum(chunk.lower().count(term) for term in query_terms)
        if score > 0:
            scored.append((chunk, float(score)))
    # Sort by frequency and return top_k
    return sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]

def semantic_search(collection, query_embedding, top_k: int = 3):
    """
    Retrieve top_k chunks from Chroma collection via embeddings.
    """
    results = query_collection(collection, query_embedding, n_results=top_k)
    docs = results["documents"][0]
    scores = results["distances"][0]
    return list(zip(docs, scores))

def hybrid_search(chunks: List[str], collection, query: str, query_embedding, top_k: int = 5):
    """
    Combine semantic search (Chroma) + keyword search.
    """
    sem_results = semantic_search(collection, query_embedding, top_k=top_k)
    kw_results = keyword_search(chunks, query, top_k=top_k)

    # Merge with simple normalization (semantic score = distance, so invert it)
    merged = {}
    for doc, score in sem_results:
        merged[doc] = merged.get(doc, 0) + (1 / (1 + score))  # higher = better
    for doc, score in kw_results:
        merged[doc] = merged.get(doc, 0) + score

    # Sort merged results
    return sorted(merged.items(), key=lambda x: x[1], reverse=True)[:top_k]

