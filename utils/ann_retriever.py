"""
ANN (Approximate Nearest Neighbors) Retriever using FAISS
Provides fast semantic search for large product catalogs
"""

import numpy as np
from typing import List, Tuple, Optional
import logging
from sentence_transformers import SentenceTransformer

try:
    import faiss
except ImportError:
    print("FAISS not installed. Install with: pip install faiss-cpu")
    print("For GPU support: pip install faiss-gpu")
    raise

logger = logging.getLogger(__name__)

class FAISSANNRetriever:
    """FAISS-based ANN retriever for fast semantic search."""
    
    def __init__(self, 
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 index_type: str = "hnsw",
                 m: int = 32,
                 ef_construction: int = 200,
                 ef_search: int = 50):
        """
        Initialize FAISS ANN Retriever.
        
        Args:
            model_name: Sentence transformer model for embeddings
            index_type: Type of FAISS index ('hnsw', 'ivf', 'flat')
            m: Number of neighbors per node (HNSW parameter)
            ef_construction: Construction parameter for HNSW
            ef_search: Search parameter for HNSW
        """
        self.model_name = model_name
        self.embedder = SentenceTransformer(model_name)
        self.index_type = index_type
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        
        self.index = None
        self.documents = []
        self.embeddings = None
        self.is_trained = False
        
        logger.info(f"Initialized FAISS ANN Retriever with model: {model_name}")
    
    def create_index(self, dimension: int) -> faiss.Index:
        """Create FAISS index based on specified type."""
        if self.index_type.lower() == "hnsw":
            index = faiss.IndexHNSWFlat(dimension, self.m)
            index.hnsw.efConstruction = self.ef_construction
            index.hnsw.efSearch = self.ef_search
            
        elif self.index_type.lower() == "ivf":
            # IVF with 100 centroids, good for larger datasets
            nlist = min(100, len(self.documents) // 10) if self.documents else 100
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
            
        elif self.index_type.lower() == "flat":
            # Exact search (for comparison)
            index = faiss.IndexFlatIP(dimension)
            
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
        
        logger.info(f"Created {self.index_type.upper()} index with dimension {dimension}")
        return index
    
    def encode_documents(self, documents: List[str], show_progress: bool = True) -> np.ndarray:
        """Encode documents into embeddings."""
        logger.info(f"Encoding {len(documents)} documents...")
        
        embeddings = self.embedder.encode(
            documents, 
            convert_to_numpy=True, 
            show_progress_bar=show_progress,
            normalize_embeddings=True  # Important for inner product similarity
        )
        
        return embeddings.astype('float32')
    
    def build_index(self, documents: List[str], show_progress: bool = True):
        """Build the FAISS index from documents."""
        if not documents:
            raise ValueError("Cannot build index with empty document list")
        
        self.documents = documents.copy()
        
        # Encode documents
        self.embeddings = self.encode_documents(documents, show_progress)
        
        # Create and populate index
        dimension = self.embeddings.shape[1]
        self.index = self.create_index(dimension)
        
        # Train index if needed (for IVF)
        if hasattr(self.index, 'train'):
            logger.info("Training index...")
            self.index.train(self.embeddings)
            self.is_trained = True
        
        # Add embeddings to index
        logger.info(f"Adding {len(self.embeddings)} vectors to index...")
        self.index.add(self.embeddings)
        
        logger.info(f"Index built successfully. Total vectors: {self.index.ntotal}")
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Search for similar documents using ANN.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of (document, similarity_score) tuples
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Encode query
        query_embedding = self.embedder.encode(
            [query], 
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype('float32')
        
        # Search index
        similarities, indices = self.index.search(query_embedding, top_k)
        
        # Format results
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx != -1:  # Valid result
                results.append((self.documents[idx], float(sim)))
        
        return results
    
    def update_search_params(self, ef_search: Optional[int] = None):
        """Update search parameters for HNSW index."""
        if self.index_type.lower() == "hnsw" and self.index is not None:
            if ef_search is not None:
                self.ef_search = ef_search
                self.index.hnsw.efSearch = ef_search
                logger.info(f"Updated efSearch to {ef_search}")
    
    def get_index_info(self) -> dict:
        """Get information about the current index."""
        if self.index is None:
            return {"status": "not_built"}
        
        info = {
            "status": "built",
            "index_type": self.index_type,
            "total_vectors": self.index.ntotal,
            "dimension": self.embeddings.shape[1] if self.embeddings is not None else 0,
            "model_name": self.model_name,
            "num_documents": len(self.documents)
        }
        
        if self.index_type.lower() == "hnsw":
            info.update({
                "m": self.m,
                "ef_construction": self.ef_construction,
                "ef_search": self.ef_search
            })
        
        return info
    
    def save_index(self, filepath: str):
        """Save the FAISS index to disk."""
        if self.index is None:
            raise ValueError("No index to save")
        
        faiss.write_index(self.index, filepath)
        logger.info(f"Index saved to {filepath}")
    
    def load_index(self, filepath: str, documents: List[str]):
        """Load a FAISS index from disk."""
        self.index = faiss.read_index(filepath)
        self.documents = documents.copy()
        logger.info(f"Index loaded from {filepath}")
    
    def benchmark_search(self, queries: List[str], top_k: int = 10) -> dict:
        """Benchmark search performance."""
        import time
        
        if not queries:
            return {"error": "No queries provided"}
        
        times = []
        total_results = 0
        
        for query in queries:
            start_time = time.time()
            results = self.search(query, top_k)
            end_time = time.time()
            
            times.append(end_time - start_time)
            total_results += len(results)
        
        return {
            "num_queries": len(queries),
            "avg_time_ms": np.mean(times) * 1000,
            "total_time_ms": sum(times) * 1000,
            "avg_results_per_query": total_results / len(queries),
            "queries_per_second": len(queries) / sum(times)
        }