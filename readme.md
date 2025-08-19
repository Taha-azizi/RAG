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