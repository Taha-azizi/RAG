from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings

def get_client(persist_directory: Optional[str] = None):
    """Return a Chroma client. If persist_directory is provided, use a persistent client."""
    if persist_directory:
        # Newer Chroma versions
        try:
            client = chromadb.PersistentClient(path=persist_directory)
        except AttributeError:
            # Backward compatibility
            client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_directory))
    else:
        client = chromadb.Client(Settings(allow_reset=True))
    return client

def get_or_create_collection(client, name: str = "local_rag"):
    try:
        return client.get_collection(name)
    except Exception:
        return client.create_collection(name=name)

def reset_collection(client, name: str):
    try:
        client.delete_collection(name)
    except Exception:
        pass
    return client.create_collection(name=name)

def add_documents(collection, chunks: List[str], embeddings, metadatas: List[Dict] = None):
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(ids=ids, documents=chunks, embeddings=list(embeddings), metadatas=metadatas)

def query_collection(collection, query_embeddings, n_results: int = 3):
    return collection.query(
        query_embeddings=[query_embeddings],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )