
import argparse
import math
import sys
from typing import List, Tuple, Dict, Any
from pathlib import Path
from flask import Flask, request, jsonify
import threading
import time

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

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

# Global RAG state
class RAGState:
    def __init__(self):
        self.initialized = False
        self.embedder = None
        self.collection = None
        self.bm25 = None
        self.chunks = []
        self.chunk_embeddings = None
        self.client = None
        self.config = {}

rag_state = RAGState()

# Flask app for HTTP endpoints
app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "initialized": rag_state.initialized})

@app.route('/call/<tool_name>', methods=['POST'])
def call_tool(tool_name: str):
    """Call an MCP tool via HTTP."""
    try:
        data = request.get_json() or {}
        
        if tool_name == "initialize_rag":
            result = tool_initialize_rag(**data)
        elif tool_name == "rag_search":
            result = tool_rag_search(**data)
        elif tool_name == "rag_answer":
            result = tool_rag_answer(**data)
        elif tool_name == "calculator":
            result = tool_calculator(**data)
        elif tool_name == "rag_status":
            result = tool_rag_status()
        elif tool_name == "list_available_models":
            result = tool_list_models()
        else:
            return jsonify({"error": f"Unknown tool: {tool_name}"}), 400
        
        return jsonify({"result": result})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# MCP Tools Implementation
def tool_initialize_rag(
    data_dir: str = "data",
    pattern: str = "*.txt", 
    persist_dir: str = ".chroma",
    collection_name: str = "mcp_rag",
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    chunk_size: int = 500,
    overlap: int = 50,
    reset: bool = False
) -> str:
    """Initialize the RAG system."""
    try:
        global rag_state
        
        rag_state.config = {
            'data_dir': data_dir,
            'pattern': pattern,
            'persist_dir': persist_dir,
            'collection_name': collection_name,
            'embed_model': embed_model,
            'chunk_size': chunk_size,
            'overlap': overlap
        }
        
        # 1) Load documents
        texts, sources = load_text_files(data_dir, pattern=pattern)
        
        # 2) Chunk documents
        chunks, metadatas = chunk_documents(texts, sources, chunk_size=chunk_size, overlap=overlap)
        rag_state.chunks = chunks
        
        # 3) Initialize embedder
        rag_state.embedder = get_embedder(embed_model)
        
        # 4) Create embeddings
        rag_state.chunk_embeddings = embed_texts(chunks, rag_state.embedder)
        
        # 5) Setup vector store
        rag_state.client = get_client(persist_dir if persist_dir else None)
        if reset:
            rag_state.collection = reset_collection(rag_state.client, collection_name)
        else:
            rag_state.collection = get_or_create_collection(rag_state.client, collection_name)
        
        add_documents(rag_state.collection, chunks, rag_state.chunk_embeddings, metadatas)
        
        # 6) Build BM25 index
        rag_state.bm25, _ = build_bm25_index(chunks, return_tokenized=True)
        
        rag_state.initialized = True
        
        return f"✅ RAG system initialized successfully!\n" \
               f"📁 Loaded {len(texts)} files from '{data_dir}'\n" \
               f"📄 Created {len(chunks)} chunks\n" \
               f"🧠 Using embedder: {embed_model}\n" \
               f"💾 Collection: {collection_name}"
               
    except Exception as e:
        return f"❌ Error initializing RAG system: {str(e)}"

def tool_rag_search(
    query: str,
    mode: str = "hybrid",
    top_k: int = 3,
    weight_semantic: float = 0.6,
    weight_keyword: float = 0.4
) -> str:
    """Search the RAG knowledge base."""
    if not rag_state.initialized:
        return "❌ RAG system not initialized. Call initialize_rag() first."
    
    try:
        mode = mode.lower().strip()
        
        if mode == "semantic":
            docs, distances = semantic_search(
                query, top_k, rag_state.chunks, rag_state.embedder, rag_state.chunk_embeddings
            )
            context = "\n\n".join(docs)
            
        elif mode == "keyword":
            docs, scores = keyword_search(
                query, top_k, bm25=rag_state.bm25, all_chunks=rag_state.chunks
            )
            context = "\n\n".join(docs)
            
        elif mode == "hybrid":
            docs = hybrid_search(
                query=query,
                top_k=top_k,
                weight_semantic=weight_semantic,
                weight_keyword=weight_keyword,
                embedder=rag_state.embedder,
                bm25=rag_state.bm25,
                all_chunks=rag_state.chunks,
                chunk_embeddings=rag_state.chunk_embeddings,
            )
            context = "\n\n".join(docs)
            
        else:
            return f"❌ Unknown search mode: {mode}. Use 'semantic', 'keyword', or 'hybrid'."
        
        return f"🔍 Search Results ({mode} mode, top {top_k}):\n\n{context}"
        
    except Exception as e:
        return f"❌ Error during search: {str(e)}"

def tool_rag_answer(
    query: str,
    mode: str = "hybrid",
    top_k: int = 3,
    llm_model: str = "mistral",
    weight_semantic: float = 0.6,
    weight_keyword: float = 0.4
) -> str:
    """Get an AI-generated answer based on RAG retrieval."""
    if not rag_state.initialized:
        return "❌ RAG system not initialized. Call initialize_rag() first."
    
    try:
        # First get the context
        context_result = tool_rag_search(query, mode, top_k, weight_semantic, weight_keyword)
        
        # Extract just the context part
        context_lines = context_result.split('\n\n')
        if len(context_lines) > 1:
            context = '\n\n'.join(context_lines[1:])
        else:
            context = context_result
        
        # Generate grounded response
        grounded_prompt = (
            "You are a helpful assistant. Use ONLY the context below to answer the user's question.\n"
            "If the answer is not in the context, say you don't know.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            "Answer:"
        )
        
        answer = generate_with_ollama(llm_model, grounded_prompt)
        
        return f"🤖 AI Answer (using {llm_model}):\n\n{answer}\n\n" \
               f"📚 Based on {top_k} retrieved passages using {mode} search."
        
    except Exception as e:
        return f"❌ Error generating answer: {str(e)}"

def tool_calculator(expression: str) -> str:
    """Evaluate mathematical expressions safely."""
    try:
        result = eval(expression, {"__builtins__": {}}, {"math": math})
        return f"🧮 Calculation: {expression} = {result}"
    except Exception as e:
        return f"❌ Calculation error: {e}"

def tool_rag_status() -> str:
    """Get current RAG system status."""
    if not rag_state.initialized:
        return "❌ RAG system not initialized. Call initialize_rag() first."
    
    config = rag_state.config
    return f"✅ RAG System Status:\n\n" \
           f"📁 Data directory: {config.get('data_dir', 'N/A')}\n" \
           f"📄 Pattern: {config.get('pattern', 'N/A')}\n" \
           f"🗂️ Total chunks: {len(rag_state.chunks)}\n" \
           f"🧠 Embedding model: {config.get('embed_model', 'N/A')}\n" \
           f"💾 Collection: {config.get('collection_name', 'N/A')}\n" \
           f"🔧 Chunk size: {config.get('chunk_size', 'N/A')}\n" \
           f"🔗 Overlap: {config.get('overlap', 'N/A')}"

def tool_list_models() -> str:
    """List available Ollama models."""
    try:
        import subprocess
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return f"🤖 Available Ollama Models:\n\n{result.stdout}"
        else:
            return f"❌ Error listing models: {result.stderr}"
    except Exception as e:
        return f"❌ Error accessing Ollama: {str(e)}\nMake sure Ollama is installed and running."

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="MCP RAG Server (Fixed)")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--auto-init", action="store_true", help="Auto-initialize RAG on startup")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory for auto-init")
    
    args = parser.parse_args()
    
    print("🚀 Starting MCP RAG Server (Fixed Version)...")
    print(f"📡 Server will run on {args.host}:{args.port}")
    
    # Auto-initialize if requested
    if args.auto_init:
        print(f"\n🔄 Auto-initializing RAG with data from '{args.data_dir}'...")
        result = tool_initialize_rag(data_dir=args.data_dir)
        print(result)
    
    print(f"\n✅ MCP RAG Server ready at http://{args.host}:{args.port}")
    print("📖 Available endpoints:")
    print("   • GET  /health - Health check")
    print("   • POST /call/<tool_name> - Call MCP tools")
    
    # Run Flask server
    app.run(host=args.host, port=args.port, debug=False)

if __name__ == "__main__":
    main()