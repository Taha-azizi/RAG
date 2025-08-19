import nltk
from rank_bm25 import BM25Okapi
from utils.chunking import chunk_text
from utils.embeddings import embed_chunks, semantic_search

# Make sure NLTK tokenizer is ready
nltk.download("punkt", quiet=True)

bm25 = None
all_chunks = []

def build_bm25_index(chunks):
    """
    Build BM25 index from text chunks.
    """
    global bm25, all_chunks
    all_chunks = chunks
    tokenized_chunks = [nltk.word_tokenize(chunk.lower()) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)

def keyword_search(query, top_k=5):
    """
    Keyword-based BM25 search.
    """
    tokenized_query = nltk.word_tokenize(query.lower())
    scores = bm25.get_scores(tokenized_query)
    ranked_ids = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [all_chunks[i] for i in ranked_ids], [scores[i] for i in ranked_ids]

def hybrid_search(query, top_k=5, weight_semantic=0.6, weight_keyword=0.4):
    """
    Combine semantic search + keyword search into a hybrid retrieval.
    """
    sem_docs, sem_scores = semantic_search(query, top_k)
    key_docs, key_scores = keyword_search(query, top_k)

    combined = {}
    # Convert cosine distances to similarity
    sem_scores = [1 - s for s in sem_scores]

    for doc, score in zip(sem_docs, sem_scores):
        combined[doc] = combined.get(doc, 0) + score * weight_semantic

    for doc, score in zip(key_docs, key_scores):
        combined[doc] = combined.get(doc, 0) + score * weight_keyword

    ranked_docs = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked_docs[:top_k]]