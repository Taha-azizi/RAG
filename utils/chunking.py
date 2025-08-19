from typing import List, Dict, Tuple

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks.
    Ensures chunk_size > overlap to avoid infinite loops.
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def chunk_documents(texts: List[str], sources: List[str], chunk_size: int = 500, overlap: int = 50) -> Tuple[List[str], List[Dict]]:
    """Chunk multiple documents and build matching metadata entries."""
    all_chunks: List[str] = []
    metadatas: List[Dict] = []
    for i, text in enumerate(texts):
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        for j, c in enumerate(chunks):
            all_chunks.append(c)
            metadatas.append({"source": sources[i], "chunk": j})
    return all_chunks, metadatas
