This repo demonstrates a minimal Retrieval-Augmented Generation (RAG) pipeline that runs fully offline on your machine. ## Stack - **Ollama** to run a local LLM (e.g., mistral, llama3) - **SentenceTransformers** for text embeddings - **ChromaDB** as the local vector database ## Quickstart 1) **Install Ollama** and pull a model:
bash
   # https://ollama.com/download
   ollama pull mistral
2) **Create a virtual environment & install dependencies**:
bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   # source .venv/bin/activate

   pip install -r requirements.txt
> Note: For GPU acceleration, install the appropriate PyTorch build following the instructions at https://pytorch.org/get-started/locally/ before installing sentence-transformers. 3) **Add your .txt files** to the data/ folder (a sample is provided). 4) **Run the pipeline**:
bash
   python simple_rag.py --reset --query "What does the document say about renewable energy?"
Common flags: - --data_dir data Folder with .txt files - --persist_dir .chroma Directory for Chroma persistence ('' for in-memory) - --collection local_rag Chroma collection name - --embed_model sentence-transformers/all-MiniLM-L6-v2 - --llm_model mistral (try llama3 or qwen if installed) - --chunk_size 500 --overlap 50 - --n_results 3 5) **Example**:
bash
   python simple_rag.py --reset --n_results 5 --query "List the barriers to renewable deployment mentioned in the documents."
## Notes
- If `ollama` is not found in your PATH, start the Ollama app/daemon and ensure the CLI is available.
- To start fresh, pass `--reset` to recreate the Chroma collection.
- You can switch models with `--llm_model llama3` (after `ollama pull llama3`).
- For larger datasets, consider a persistent DB (`--persist_dir .chroma`) so you don’t re-index every run.

# MCP RAG Setup Guide

This guide shows how to set up a local RAG system using MCP (Model Context Protocol) with FastMCP.

## 🎯 What You'll Build

- A local MCP server that exposes RAG capabilities as tools
- Search your documents using semantic, keyword, or hybrid search
- Get AI-generated answers grounded in your documents
- All running locally with free, open-source tools

## 📋 Prerequisites

1. **Python 3.8+** installed
2. **Ollama** installed and running ([https://ollama.com](https://ollama.com))
3. At least one Ollama model downloaded (e.g., `ollama pull mistral`)

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install MCP RAG requirements
pip install -r requirements-mcp.txt

# Download NLTK data (for tokenization)
python -c "import nltk; nltk.download('punkt')"
```

### 2. Prepare 

Make sure you installed all required dependencies and downloaded an Ollama model before proceeding.


### 3. Start the MCP Server

```bash
# Start with auto-initialization
python mcp_rag.py --auto-init

# Or start and initialize manually
python mcp_rag.py
```

The server will run on `http://localhost:8000` by default.

### 4. Test with the Client

```bash
# Interactive mode
python mcp_client.py --mode interactive

# Demo mode
python mcp_client.py --mode demo
```

## 🛠️ MCP Tools Available

Your MCP server exposes these tools:

### 1. `initialize_rag`
Set up the RAG system with your documents.

**Parameters:**
- `data_dir` (str): Directory with text files (default: "data")
- `pattern` (str): File pattern to match (default: "*.txt")
- `embed_model` (str): Sentence transformer model
- `chunk_size` (int): Text chunk size (default: 500)
- `overlap` (int): Chunk overlap (default: 50)
- `reset` (bool): Reset collection (default: false)

### 2. `rag_search`
Search your knowledge base.

**Parameters:**
- `query` (str): Search query
- `mode` (str): "semantic", "keyword", or "hybrid"
- `top_k` (int): Number of results (default: 3)

### 3. `rag_answer`
Get AI-generated answers using RAG.

**Parameters:**
- `query` (str): Question to answer
- `mode` (str): Search mode
- `llm_model` (str): Ollama model (default: "mistral")
- `top_k` (int): Context chunks to use

### 4. `calculator`
Perform mathematical calculations.

**Parameters:**
- `expression` (str): Math expression to evaluate

### 5. `rag_status`
Get current system status.

### 6. `list_available_models`
List available Ollama models.

## 🔧 Configuration Options

### Server Configuration

```bash
# Custom host and port
python mcp_rag.py --host 0.0.0.0 --port 8080

# Auto-initialize with custom data directory
python mcp_rag.py --auto-init --data-dir /path/to/docs
```

### Client Configuration

```bash
# Connect to different server
python mcp_client.py --server http://your-server:8080
```

## 📱 Using with MCP Clients

### Claude Desktop Integration

Add to your Claude Desktop configuration:

```json
{
  "mcpServers": {
    "local-rag": {
      "command": "python",
      "args": ["/path/to/your/mcp_rag.py"],
      "env": {}
    }
  }
}
```

### Other MCP Clients

The server implements the standard MCP protocol and works with:
- Claude Desktop
- Continue.dev
- Any MCP-compatible client

## 🧪 Example Usage

### Interactive Client Session

```
💬 Enter command: init data_dir=data reset=true
✅ RAG system initialized successfully!
📁 Loaded 2 files from 'data'
📄 Created 45 chunks

💬 Enter command: search renewable energy
🔍 Search Results (hybrid mode, top 3):
[Retrieved context about renewable energy...]

💬 Enter command: answer What are the benefits of solar power?
🤖 AI Answer (using mistral):
Based on the provided context, solar power offers several benefits...

💬 Enter command: calc 25 * 8 + 150
🧮 Calculation: 25 * 8 + 150 = 350
```

## 

This implementation demonstrates:

1. **Local MCP Server**: No external dependencies or API keys needed
2. **RAG Integration**: Full semantic, keyword, and hybrid search
3. **Tool Composition**: Multiple tools working together (RAG + calculator)
4. **State Management**: Persistent RAG state across tool calls
5. **Error Handling**: Graceful error handling and user feedback
6. **Extensibility**: Easy to add new tools and capabilities

## 🔍 What Makes This Special

- **100% Local**: No cloud services or API keys required
- **Production Ready**: Proper error handling and state management
- **MCP Standard**: Works with any MCP-compatible client
- **Backwards Compatible**: Your existing `agentic_rag.py` still works
- **Extensible**: Easy to add new MCP tools

## 🚨 Troubleshooting

### Common Issues

1. **"Ollama not found"**
   - Install Ollama from https://ollama.com
   - Make sure `ollama` command is in your PATH

2. **"FastMCP not installed"**
   - Run: `pip install fastmcp`

3. **"No files found"**
   - Check your `data/` directory has .txt files
   - Verify file permissions

4. **"Collection already exists"**
   - Use `reset=true` parameter in `initialize_rag`

### Getting Help

- Check server logs for detailed error messages
- Use `rag_status` tool to check system state
- Test with the included client before using other MCP clients

## 🎉 Next Steps

- Add more document formats (PDF, Word, etc.)
- Implement document management tools
- Add query history and analytics
- Create custom embedding models
- Scale to larger document collections

Happy RAG-ing with MCP! 🚀