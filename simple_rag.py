import argparse
from utils.load import load_text_files
from utils.chunking import chunk_documents
from utils.embeddings import get_embedder, embed_texts
from utils.store import get_client, get_or_create_collection, reset_collection, add_documents, query_collection
from utils.generation import generate_with_ollama

def build_and_query(
    data_dir: str,
    persist_dir: str,
    collection_name: str,
    embed_model: str,
    llm_model: str,
    chunk_size: int,
    overlap: int,
    n_results: int,
    query: str,
    reset: bool,
):
    # 1) Load raw docs
    texts, sources = load_text_files(data_dir)

    # 2) Chunk
    chunks, metadatas = chunk_documents(texts, sources, chunk_size=chunk_size, overlap=overlap)

    # 3) Embeddings
    embedder = get_embedder(embed_model)
    embeddings = embed_texts(chunks, embedder)

    # 4) Vector store
    client = get_client(persist_dir if persist_dir else None)
    if reset:
        collection = reset_collection(client, collection_name)
    else:
        collection = get_or_create_collection(client, collection_name)

    add_documents(collection, chunks, embeddings, metadatas)

    # 5) Query
    q_emb = embed_texts([query], embedder)[0]
    results = query_collection(collection, q_emb, n_results=n_results)

    # 6) Compose context
    top_docs = results["documents"][0]
    context = "\n\n".join(top_docs)

    # 7) Generate
    prompt = (
        "Answer the question using ONLY the following context.\n"
        "If the answer cannot be found in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    )
    answer = generate_with_ollama(llm_model, prompt)

    # Print nicely
    print("")
    print(f"Question: {query}\n")
    print("")
    print("\n=== Top Retrieved Passages ===\n")
    for i, doc in enumerate(top_docs, start=1):
        print(f"[{i}] {doc[:500]}" + ("..." if len(doc) > 500 else ""), "\n")
    print("=== LLM Answer ===\n")
    print(answer)

def main():
    ap = argparse.ArgumentParser(description="Simple Local RAG Pipeline (Ollama + Chroma + SentenceTransformers)")
    ap.add_argument("--data_dir", type=str, default="data", help="Folder with .txt files")
    ap.add_argument("--persist_dir", type=str, default=".chroma", help="Directory for Chroma persistence (use '' for in-memory)")
    ap.add_argument("--collection", type=str, default="local_rag", help="Chroma collection name")
    ap.add_argument("--embed_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Embedding model name")
    ap.add_argument("--llm_model", type=str, default="mistral", help="Ollama model name (e.g., mistral, llama3, qwen)")
    ap.add_argument("--chunk_size", type=int, default=500, help="Chunk size in characters")
    ap.add_argument("--overlap", type=int, default=50, help="Overlap between chunks in characters")
    ap.add_argument("--n_results", type=int, default=3, help="How many chunks to retrieve")
    ap.add_argument("--query", type=str, default="What does the document say about renewable energy?", help="Your question")
    ap.add_argument("--reset", action="store_true", help="Reset the collection before indexing (fresh run)")
    args = ap.parse_args()

    build_and_query(
        data_dir=args.data_dir,
        persist_dir=args.persist_dir,
        collection_name=args.collection,
        embed_model=args.embed_model,
        llm_model=args.llm_model,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        n_results=args.n_results,
        query=args.query,
        reset=args.reset,
    )

if __name__ == "__main__":
    main()
