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

    # 5) Hybrid retrieval: semantic + keyword
    q_emb = embed_texts([query], embedder)[0]
    semantic_results = query_collection(collection, q_emb, n_results=n_results)

    keyword_results = collection.query(
        query_texts=[query],
        n_results=n_results
    )

    # Merge results (naive: just concatenate and deduplicate)
    combined_docs = list(dict.fromkeys(
        semantic_results["documents"][0] + keyword_results["documents"][0]
    ))

    # 6) Compose context
    context = "\n\n".join(combined_docs)

    # 7) Generate
    prompt = (
        "You are a helpful product assistant for an e-commerce store.\n"
        "Use ONLY the information provided in the context below to answer the customer's question.\n"
        "If the answer is not present in the context, say: 'I don’t know based on the available product information.'\n\n"
        "When possible, include product names, key features, and benefits in a clear, customer-friendly way.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    )
    answer = generate_with_ollama(llm_model, prompt)

    # Print nicely
    print("")
    print(f"Question: {query}\n")
    print("=== Top Retrieved Passages (Hybrid) ===\n")
    for i, doc in enumerate(combined_docs, start=1):
        print(f"[{i}] {doc[:500]}" + ("..." if len(doc) > 500 else ""), "\n")
    print("=== LLM Answer ===\n")
    print(answer)


def main():
    ap = argparse.ArgumentParser(description="Hybrid RAG Pipeline (Semantic + Keyword)")
    ap.add_argument("--data_dir", type=str, default="data", help="Folder with .txt files")
    ap.add_argument("--persist_dir", type=str, default=".chroma", help="Directory for Chroma persistence (use '' for in-memory)")
    ap.add_argument("--collection", type=str, default="hybrid_rag", help="Chroma collection name")
    ap.add_argument("--embed_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Embedding model name")
    ap.add_argument("--llm_model", type=str, default="mistral", help="Ollama model name (e.g., mistral, llama3, qwen)")
    ap.add_argument("--chunk_size", type=int, default=500, help="Chunk size in characters")
    ap.add_argument("--overlap", type=int, default=50, help="Overlap between chunks in characters")
    ap.add_argument("--n_results", type=int, default=3, help="How many chunks to retrieve (per method)")
    ap.add_argument("--query", type=str, default="Which laptop is best for competitive gaming?", help="Your question")
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
