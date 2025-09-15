"""
Cross-Encoder Reranker for refining ANN search results
Provides accurate relevance scoring for top candidates
"""

import logging
from typing import List, Tuple, Optional
import numpy as np

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    print("sentence-transformers not installed. Install with: pip install sentence-transformers")
    raise

logger = logging.getLogger(__name__)

class CrossEncoderReranker:
    """Cross-encoder model for reranking search results."""
    
    def __init__(self, 
                 model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 max_length: int = 512):
        """
        Initialize cross-encoder reranker.
        
        Args:
            model_name: Pretrained cross-encoder model
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.max_length = max_length
        
        logger.info(f"Loading cross-encoder model: {model_name}")
        self.model = CrossEncoder(model_name, max_length=max_length)
        logger.info("Cross-encoder model loaded successfully")
    
    def rerank(self, 
               query: str, 
               documents: List[str], 
               top_k: Optional[int] = None) -> List[Tuple[str, float]]:
        """
        Rerank documents based on relevance to query.
        
        Args:
            query: Search query
            documents: List of candidate documents
            top_k: Number of top results to return (None = all)
            
        Returns:
            List of (document, relevance_score) tuples, sorted by relevance
        """
        if not documents:
            return []
        
        logger.debug(f"Reranking {len(documents)} documents for query: '{query[:50]}...'")
        
        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]
        
        # Predict relevance scores
        scores = self.model.predict(pairs)
        
        # Combine documents with scores and sort
        doc_scores = list(zip(documents, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k if specified
        if top_k is not None:
            doc_scores = doc_scores[:top_k]
        
        return doc_scores
    
    def rerank_with_candidates(self, 
                              query: str, 
                              candidates: List[Tuple[str, float]], 
                              top_k: Optional[int] = None,
                              preserve_ann_scores: bool = False) -> List[Tuple[str, float, Optional[float]]]:
        """
        Rerank candidates from ANN retrieval.
        
        Args:
            query: Search query
            candidates: List of (document, ann_score) tuples from ANN
            top_k: Number of top results to return
            preserve_ann_scores: Whether to return original ANN scores
            
        Returns:
            List of (document, rerank_score, ann_score) if preserve_ann_scores
            List of (document, rerank_score) otherwise
        """
        if not candidates:
            return []
        
        # Extract documents and ANN scores
        documents = [doc for doc, _ in candidates]
        ann_scores = [score for _, score in candidates]
        
        # Rerank documents
        reranked = self.rerank(query, documents, top_k)
        
        if preserve_ann_scores:
            # Find original ANN scores for reranked documents
            ann_score_map = {doc: score for doc, score in candidates}
            result = []
            for doc, rerank_score in reranked:
                ann_score = ann_score_map.get(doc)
                result.append((doc, rerank_score, ann_score))
            return result
        else:
            return reranked
    
    def batch_rerank(self, 
                     queries_and_docs: List[Tuple[str, List[str]]], 
                     top_k: Optional[int] = None) -> List[List[Tuple[str, float]]]:
        """
        Rerank multiple query-document sets in batch.
        
        Args:
            queries_and_docs: List of (query, documents) tuples
            top_k: Number of top results per query
            
        Returns:
            List of reranked results for each query
        """
        results = []
        
        for query, documents in queries_and_docs:
            reranked = self.rerank(query, documents, top_k)
            results.append(reranked)
        
        return results
    
    def score_threshold_filter(self, 
                              results: List[Tuple[str, float]], 
                              min_score: float) -> List[Tuple[str, float]]:
        """
        Filter results by minimum relevance score.
        
        Args:
            results: List of (document, score) tuples
            min_score: Minimum relevance score threshold
            
        Returns:
            Filtered list of results above threshold
        """
        return [(doc, score) for doc, score in results if score >= min_score]
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "max_length": self.max_length,
            "model_type": "cross-encoder"
        }

class HybridReranker:
    """Hybrid reranker combining multiple scoring methods."""
    
    def __init__(self, 
                 cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 ann_weight: float = 0.3,
                 cross_encoder_weight: float = 0.7):
        """
        Initialize hybrid reranker.
        
        Args:
            cross_encoder_model: Cross-encoder model name
            ann_weight: Weight for ANN similarity scores
            cross_encoder_weight: Weight for cross-encoder scores
        """
        self.ann_weight = ann_weight
        self.cross_encoder_weight = cross_encoder_weight
        self.reranker = CrossEncoderReranker(cross_encoder_model)
        
        logger.info(f"Initialized hybrid reranker (ANN: {ann_weight}, CE: {cross_encoder_weight})")
    
    def hybrid_rerank(self, 
                     query: str, 
                     candidates: List[Tuple[str, float]], 
                     top_k: Optional[int] = None) -> List[Tuple[str, float]]:
        """
        Rerank using weighted combination of ANN and cross-encoder scores.
        
        Args:
            query: Search query
            candidates: List of (document, ann_score) tuples
            top_k: Number of results to return
            
        Returns:
            List of (document, hybrid_score) tuples
        """
        if not candidates:
            return []
        
        # Get cross-encoder scores with preserved ANN scores
        reranked_with_ann = self.reranker.rerank_with_candidates(
            query, candidates, top_k=None, preserve_ann_scores=True
        )
        
        # Calculate hybrid scores
        hybrid_results = []
        
        for doc, ce_score, ann_score in reranked_with_ann:
            if ann_score is not None:
                # Normalize scores to [0, 1] range if needed
                normalized_ann = max(0, min(1, ann_score))
                normalized_ce = max(0, min(1, (ce_score + 1) / 2))  # CE scores can be negative
                
                hybrid_score = (self.ann_weight * normalized_ann + 
                              self.cross_encoder_weight * normalized_ce)
            else:
                hybrid_score = ce_score
            
            hybrid_results.append((doc, hybrid_score))
        
        # Sort by hybrid score
        hybrid_results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k
        if top_k is not None:
            hybrid_results = hybrid_results[:top_k]
        
        return hybrid_results
    
    def update_weights(self, ann_weight: float, cross_encoder_weight: float):
        """Update scoring weights."""
        self.ann_weight = ann_weight
        self.cross_encoder_weight = cross_encoder_weight
        logger.info(f"Updated weights: ANN={ann_weight}, CE={cross_encoder_weight}")