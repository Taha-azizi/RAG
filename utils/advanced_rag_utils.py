"""
Utilities for Advanced RAG with ANN and Reranking
Compatible with existing hybrid_rag.py structure
"""

import time
import logging
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
import json

# Import existing utilities to maintain compatibility
try:
    from utils.embeddings import EmbeddingManager
except ImportError:
    # Fallback if utils not in path
    logging.warning("Could not import existing utilities. Some features may not be available.")

logger = logging.getLogger(__name__)

class AdvancedRAGPipeline:
    """Advanced RAG pipeline combining ANN retrieval and reranking."""
    
    def __init__(self, 
                 ann_retriever,
                 reranker,
                 llm_client: Optional[Any] = None,
                 ann_top_k: int = 20,
                 final_top_k: int = 5,
                 min_relevance_score: float = 0.1):
        """
        Initialize advanced RAG pipeline.
        
        Args:
            ann_retriever: ANN retriever instance
            reranker: Reranker instance  
            llm_client: LLM client for answer generation
            ann_top_k: Number of candidates from ANN retrieval
            final_top_k: Number of final results after reranking
            min_relevance_score: Minimum relevance threshold
        """
        self.ann_retriever = ann_retriever
        self.reranker = reranker
        self.llm_client = llm_client
        self.ann_top_k = ann_top_k
        self.final_top_k = final_top_k
        self.min_relevance_score = min_relevance_score
        
        logger.info("Advanced RAG pipeline initialized")
    
    def search(self, query: str, return_scores: bool = False) -> List[str]:
        """
        Perform ANN + reranking search.
        
        Args:
            query: Search query
            return_scores: Whether to return relevance scores
            
        Returns:
            List of documents or (document, score) tuples
        """
        # Stage 1: ANN retrieval
        start_time = time.time()
        ann_candidates = self.ann_retriever.search(query, self.ann_top_k)
        ann_time = time.time() - start_time
        
        if not ann_candidates:
            return []
        
        # Stage 2: Reranking
        start_time = time.time()
        reranked_results = self.reranker.rerank_with_candidates(
            query, ann_candidates, self.final_top_k, preserve_ann_scores=True
        )
        rerank_time = time.time() - start_time
        
        # Filter by minimum score
        filtered_results = [
            (doc, score, ann_score) for doc, score, ann_score in reranked_results
            if score >= self.min_relevance_score
        ]
        
        logger.debug(f"ANN: {ann_time:.3f}s, Rerank: {rerank_time:.3f}s")
        logger.debug(f"ANN candidates: {len(ann_candidates)}, Final: {len(filtered_results)}")
        
        if return_scores:
            return [(doc, score) for doc, score, _ in filtered_results]
        else:
            return [doc for doc, _, _ in filtered_results]
    
    def answer(self, query: str, context_template: Optional[str] = None) -> str:
        """
        Generate answer using RAG pipeline.
        
        Args:
            query: User question
            context_template: Template for formatting context
            
        Returns:
            Generated answer
        """
        if self.llm_client is None:
            return "LLM client not configured"
        
        # Retrieve relevant documents
        relevant_docs = self.search(query, return_scores=False)
        
        if not relevant_docs:
            return "I couldn't find relevant information to answer your question."
        
        # Format context
        if context_template is None:
            context_template = "Relevant products:\n{context}\n\nQuestion: {query}\nAnswer:"
        
        context = "\n".join(f"- {doc}" for doc in relevant_docs)
        prompt = context_template.format(context=context, query=query)
        
        # Generate answer
        try:
            response = self.llm_client.generate(prompt)
            return response
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Error generating answer: {str(e)}"
    
    def batch_search(self, queries: List[str]) -> List[List[str]]:
        """Perform batch search for multiple queries."""
        results = []
        for query in queries:
            query_results = self.search(query)
            results.append(query_results)
        return results
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline configuration and statistics."""
        stats = {
            "ann_top_k": self.ann_top_k,
            "final_top_k": self.final_top_k,
            "min_relevance_score": self.min_relevance_score,
            "ann_retriever": self.ann_retriever.get_index_info(),
            "reranker": self.reranker.get_model_info(),
            "llm_available": self.llm_client is not None
        }
        return stats

class PerformanceBenchmark:
    """Benchmark advanced RAG pipeline performance."""
    
    @staticmethod
    def benchmark_retrieval(pipeline, test_queries: List[str]) -> Dict[str, Any]:
        """Benchmark retrieval performance."""
        if not test_queries:
            return {"error": "No test queries provided"}
        
        ann_times = []
        rerank_times = []
        total_times = []
        result_counts = []
        
        for query in test_queries:
            # Time ANN retrieval
            start_time = time.time()
            ann_candidates = pipeline.ann_retriever.search(query, pipeline.ann_top_k)
            ann_time = time.time() - start_time
            ann_times.append(ann_time)
            
            if ann_candidates:
                # Time reranking
                start_time = time.time()
                reranked = pipeline.reranker.rerank_with_candidates(
                    query, ann_candidates, pipeline.final_top_k
                )
                rerank_time = time.time() - start_time
                rerank_times.append(rerank_time)
                result_counts.append(len(reranked))
            else:
                rerank_times.append(0)
                result_counts.append(0)
            
            total_times.append(ann_time + (rerank_times[-1] if rerank_times else 0))
        
        return {
            "num_queries": len(test_queries),
            "ann_avg_ms": sum(ann_times) / len(ann_times) * 1000,
            "rerank_avg_ms": sum(rerank_times) / len(rerank_times) * 1000,
            "total_avg_ms": sum(total_times) / len(total_times) * 1000,
            "avg_results": sum(result_counts) / len(result_counts),
            "throughput_qps": len(test_queries) / sum(total_times)
        }
    
    @staticmethod
    def compare_with_baseline(advanced_pipeline, baseline_search_func, 
                             test_queries: List[str]) -> Dict[str, Any]:
        """Compare advanced pipeline with baseline search."""
        # Benchmark advanced pipeline
        advanced_stats = PerformanceBenchmark.benchmark_retrieval(
            advanced_pipeline, test_queries
        )
        
        # Benchmark baseline
        baseline_times = []
        baseline_results = []
        
        for query in test_queries:
            start_time = time.time()
            results = baseline_search_func(query)
            end_time = time.time()
            
            baseline_times.append(end_time - start_time)
            baseline_results.append(len(results) if results else 0)
        
        baseline_stats = {
            "avg_time_ms": sum(baseline_times) / len(baseline_times) * 1000,
            "avg_results": sum(baseline_results) / len(baseline_results),
            "throughput_qps": len(test_queries) / sum(baseline_times)
        }
        
        return {
            "advanced": advanced_stats,
            "baseline": baseline_stats,
            "speedup": baseline_stats["avg_time_ms"] / advanced_stats["total_avg_ms"],
            "throughput_improvement": advanced_stats["throughput_qps"] / baseline_stats["throughput_qps"]
        }

def create_test_queries() -> List[str]:
    """Generate test queries for benchmarking."""
    return [
        "best laptop for gaming",
        "wireless noise cancelling headphones",
        "smartphone with good camera",
        "ergonomic office chair",
        "fitness tracker with heart rate",
        "4K smart TV",
        "gaming mechanical keyboard",
        "portable bluetooth speaker",
        "electric toothbrush",
        "running shoes",
        "coffee machine",
        "robot vacuum",
        "standing desk",
        "air purifier",
        "wireless charging pad",
        "camping tent",
        "gaming monitor",
        "smart watch",
        "external SSD",
        "yoga mat"
    ]

def format_search_results(results: List[Tuple[str, float]], 
                         show_scores: bool = True,
                         max_results: int = 10) -> str:
    """Format search results for display."""
    if not results:
        return "No results found."
    
    formatted = []
    for i, (doc, score) in enumerate(results[:max_results], 1):
        if show_scores:
            formatted.append(f"{i:2d}. {doc} (score: {score:.3f})")
        else:
            formatted.append(f"{i:2d}. {doc}")
    
    return "\n".join(formatted)

def save_benchmark_results(results: Dict[str, Any], filepath: str):
    """Save benchmark results to JSON file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Benchmark results saved to {filepath}")

def load_benchmark_results(filepath: str) -> Dict[str, Any]:
    """Load benchmark results from JSON file."""
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    logger.info(f"Benchmark results loaded from {filepath}")
    return results