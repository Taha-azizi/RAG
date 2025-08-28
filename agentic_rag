import argparse
import math
from typing import List, Tuple, Dict, Any

from utils.load import load_text_files
from utils.chunking import chunk_documents
from utils.embeddings import get_embedder, embed_texts, semantic_search
from utils.store import (
    get_client,
    get_or_create_collection,
    reset_collection,
    add_documents,
)
from utils.generation import generate_with_ollama
from utils.hybrid_search import build_bm25_index, keyword_search, hybrid_search


# Tools
def tool_rag_retrieve(
    query: str,
    mode: str,
    *,
    collection,
    embedder,
    bm25,
    all_chunks: List[str],
    chunk_embeddings,
    top_k: int,
    weight_semantic: float,
    weight_keyword: float,
) -> Tuple[str, Dict[str, Any]]:
    # Agent returns retrieved context 
    
    mode = mode.lower().strip()
    dbg: Dict[str, Any] = {"mode": mode, "top_k": top_k}

    if mode == "semantic":
        docs, distances = semantic_search(query, top_k, all_chunks, embedder, chunk_embeddings)
        dbg["distances"] = distances
        context = "\n\n".join(docs)

    elif mode == "keyword":
        docs, scores = keyword_search(query, top_k, bm25=bm25, all_chunks=all_chunks)
        dbg["scores"] = scores
        context = "\n\n".join(docs)

    elif mode == "hybrid":
        docs = hybrid_search(
            query=query,
            top_k=top_k,
            weight_semantic=weight_semantic,
            weight_keyword=weight_keyword,
            embedder=embedder,
            bm25=bm25,
            all_chunks=all_chunks,
            chunk_embeddings=chunk_embeddings,
        )
        context = "\n\n".join(docs)
        dbg["weights"] = {"semantic": weight_semantic, "keyword": weight_keyword}
    else:
        raise ValueError(f"Unknown retrieval mode: {mode}")

    return context, dbg


def tool_calculator(expression: str) -> str:
    """
    Minimal calculator for agent use.
    Evaluates a safe subset of Python arithmetic. No builtins.
    """
    try:
        # Allow only math ops; no names other than 'math' exposed
        return str(eval(expression, {"__builtins__": {}}, {"math": math}))
    except Exception as e:
        return f"CALC_ERROR: {e}"


# Main logic
def build_and_answer(
    data_dir: str,
    pattern: str,
    persist_dir: str,
    collection_name: str,
    embed_model: str,
    llm_model: str,
    chunk_size: int,
    overlap: int,
    n_results: int,
    query: str,
    reset: bool,
    mode: str,
    weight_semantic: float,
    weight_keyword: float,
    verbose: bool,
):
    # 1) Load raw docs
    texts, sources = load_text_files(data_dir, pattern=pattern)
    if verbose:
        print(f"[load] {len(texts)} files matched pattern '{pattern}':")
        for s in sources:
            print(f"  - {s}")

    # 2) Chunk
    chunks, metadatas = chunk_documents(texts, sources, chunk_size=chunk_size, overlap=overlap)
    if verbose:
        print(f"[chunk] Produced {len(chunks)} chunks (size={chunk_size}, overlap={overlap})")

    # 3) Embeddings
    embedder = get_embedder(embed_model)
    chunk_embeddings = embed_texts(chunks, embedder)
    if verbose:
        print(f"[embed] Created embeddings for {len(chunks)} chunks")

    # 4) Vector store
    client = get_client(persist_dir if persist_dir else None)
    if reset:
        collection = reset_collection(client, collection_name)
        if verbose:
            print(f"[store] Reset collection '{collection_name}' in '{persist_dir or 'memory'}'")
    else:
        collection = get_or_create_collection(client, collection_name)
        if verbose:
            print(f"[store] Using collection '{collection_name}' in '{persist_dir or 'memory'}'")

    add_documents(collection, chunks, chunk_embeddings, metadatas)
    if verbose:
        print(f"[store] Added {len(chunks)} docs to collection")

    # 5) Build BM25 (for keyword/hybrid)
    bm25, _ = build_bm25_index(chunks, return_tokenized=True)
    if verbose:
        print(f"[bm25] Built BM25 index over {len(chunks)} chunks")

    # 6) Determine tool to use
    if any(op in query for op in ["+", "-", "*", "/", "^", "%"]) and not query.lower().startswith("explain"):
        if verbose:
            print("[agent] Route: Calculator")
        calc_result = tool_calculator(query)
        print("\n=== Calculator Result ===\n")
        print(calc_result)
        return

    # Retrieval path
    if verbose:
        print(f"[agent] Route: RAG Retriever ({mode})")

    context, dbg = tool_rag_retrieve(
        query=query,
        mode=mode,
        collection=collection,
        embedder=embedder,
        bm25=bm25,
        all_chunks=chunks,
        chunk_embeddings=chunk_embeddings,
        top_k=n_results,
        weight_semantic=weight_semantic,
        weight_keyword=weight_keyword,
    )

    if verbose:
        print("\n=== Top Retrieved Passages ===\n")
        for i, passage in enumerate(context.split("\n\n")[:n_results], start=1):
            snippet = passage[:500] + ("..." if len(passage) > 500 else "")
            print(f"[{i}] {snippet}\n")
        print("[debug]", dbg)

    # 7) Generate answer
    grounded_prompt = (
        "You are a helpful assistant. Use ONLY the context below to answer the user's question.\n"
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )

    answer = generate_with_ollama(llm_model, grounded_prompt)
    print("\n=== LLM Answer ===\n")
    print(answer)


def main():
    ap = argparse.ArgumentParser(description="Agentic RAG (Local) — RAG Retriever + Calculator")
    ap.add_argument("--data_dir", type=str, default="data", help="Folder with .txt files")
    ap.add_argument("--pattern", type=str, default="*.txt", help="Glob to select files, e.g., 'data2.txt'")
    ap.add_argument("--persist_dir", type=str, default=".chroma", help="Chroma persistence directory ('' for in-memory)")
    ap.add_argument("--collection", type=str, default="local_rag", help="Chroma collection name")
    ap.add_argument("--embed_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Embedding model name")
    ap.add_argument("--llm_model", type=str, default="mistral", help="Ollama model name (e.g., mistral, llama3, qwen)")
    ap.add_argument("--chunk_size", type=int, default=500, help="Chunk size in characters")
    ap.add_argument("--overlap", type=int, default=50, help="Overlap between chunks in characters")
    ap.add_argument("--n_results", type=int, default=3, help="How many chunks to retrieve")
    ap.add_argument("--query", type=str, default="Which laptop is best for competitive gaming?", help="Your question")
    ap.add_argument("--reset", action="store_true", help="Reset the collection before indexing (fresh run)")
    ap.add_argument("--mode", type=str, default="hybrid", choices=["semantic", "keyword", "hybrid"], help="Retrieval mode")
    ap.add_argument("--w_sem", type=float, default=0.6, help="Hybrid weight for semantic")
    ap.add_argument("--w_key", type=float, default=0.4, help="Hybrid weight for keyword")
    ap.add_argument("--verbose", action="store_true", help="Print debug info")
    args = ap.parse_args()

    build_and_answer(
        data_dir=args.data_dir,
        pattern=args.pattern,
        persist_dir=args.persist_dir,
        collection_name=args.collection,
        embed_model=args.embed_model,
        llm_model=args.llm_model,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        n_results=args.n_results,
        query=args.query,
        reset=args.reset,
        mode=args.mode,
        weight_semantic=args.w_sem,
        weight_keyword=args.w_key,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()