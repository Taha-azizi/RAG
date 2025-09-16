#!/usr/bin/env python3
"""
RAG Pipeline Evaluation Script
=============================

This script provides comprehensive evaluation of RAG systems using:
- Precision@k: How many retrieved docs are relevant
- Recall@k: How many relevant docs were retrieved  
- Mean Reciprocal Rank (MRR): Position of first relevant document
- RAGAS metrics for advanced evaluation

Usage:
    python rag_evaluation.py

Requirements:
    pip install ragas datasets numpy pandas scikit-learn sentence-transformers
"""

import os
import sys
import json
import time
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Try to import ragas, fallback to manual implementation
try:
    from ragas.metrics import (
        context_precision, 
        context_recall, 
        faithfulness,
        answer_relevancy
    )
    from ragas import evaluate
    RAGAS_AVAILABLE = True
    print("RAGAS library found - using advanced evaluation metrics")
except ImportError:
    RAGAS_AVAILABLE = False
    print("RAGAS not found - using basic evaluation metrics only")
    print("   Install with: pip install ragas")

@dataclass
class EvaluationResult:
    """Container for evaluation metrics"""
    precision_at_k: float
    recall_at_k: float
    mrr: float
    f1_score: float
    retrieval_time: float
    total_queries: int
    ragas_scores: Optional[Dict[str, float]] = None

class RAGEvaluator:
    """
    Comprehensive RAG evaluation system
    """
    
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the evaluator
        
        Args:
            embedding_model_name: Name of the sentence transformer model
        """
        print(f"Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        print("Embedding model loaded successfully")
        
    def create_sample_evaluation_dataset(self) -> Dict[str, List[str]]:
        """
        Create a sample evaluation dataset for demonstration
        
        Returns:
            Dictionary with questions, answers, and ground truth contexts
        """
        return {
            "question": [
                "What are the key metrics used in RAG evaluation?",
                "Why is evaluation important for RAG systems?",
                "What is the difference between precision and recall in information retrieval?",
                "How does Mean Reciprocal Rank (MRR) work?",
                "What are some optimization tips for RAG systems?",
                "What is the purpose of chunk size tuning?",
                "How can hybrid search improve RAG performance?",
                "What role do embedding models play in retrieval quality?"
            ],
            "answer": [
                "The key metrics are Precision@k, Recall@k, and Mean Reciprocal Rank (MRR). These measure accuracy, completeness, and ranking quality respectively.",
                "Evaluation is critical because RAG systems are only as strong as the documents they retrieve. Without systematic evaluation, you're relying on gut feeling instead of quantified quality metrics.",
                "Precision@k measures how many of the top k retrieved documents are truly relevant (clean context). Recall@k measures how many of all relevant documents in the corpus were retrieved (coverage).",
                "MRR looks at the position of the first relevant document in the ranked list. If the right document is always buried at rank 10, your LLM may never use it effectively.",
                "Key optimization tips include tuning chunk size (300-600 tokens), experimenting with embedding models, using hybrid search, adjusting ANN parameters, and prompt engineering.",
                "Chunk size affects both retrieval accuracy and context quality. Too small chunks miss context, too large chunks dilute relevance. 300-600 tokens is typically optimal.",
                "Hybrid search combines keyword and semantic search, which is especially useful for rare terms that might not be well-represented in embedding space.",
                "Embedding models determine how well semantic similarity is captured. Larger, more sophisticated models can improve retrieval quality but at the cost of speed."
            ],
            "ground_truth_contexts": [
                ["In Information Retrieval (IR), three classic metrics are used: Precision@k, Recall@k, and Mean Reciprocal Rank (MRR)"],
                ["At its core, a RAG pipeline is only as strong as the documents it retrieves. Evaluation gives us a way to quantify retrieval quality instead of just relying on gut feeling."],
                ["Precision@k - Of the top k retrieved documents, how many are truly relevant? Recall@k - Out of all relevant documents in the corpus, how many did we retrieve?"],
                ["Mean Reciprocal Rank (MRR) - Looks at the position of the first relevant document in the ranked list. If the right document is always buried at rank 10, your LLM may never use it."],
                ["Tune Chunk Size, Experiment with Embedding Models, Hybrid Search, ANN Parameters, Prompt Engineering"],
                ["Tune Chunk Size - Try 300–600 tokens per chunk"],
                ["Hybrid Search - Combine keyword + semantic search for rare terms"],
                ["Experiment with Embedding Models - Larger models can improve retrieval"]
            ]
        }
    
    def load_custom_dataset(self, file_path: str) -> Dict[str, List[str]]:
        """
        Load a custom evaluation dataset from JSON file
        
        Expected format:
        {
            "question": ["Q1", "Q2", ...],
            "answer": ["A1", "A2", ...],  
            "ground_truth_contexts": [["context1", "context2"], ...]
        }
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Loaded dataset dictionary
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            required_keys = ['question', 'answer', 'ground_truth_contexts']
            missing_keys = [key for key in required_keys if key not in data]
            
            if missing_keys:
                raise ValueError(f"Missing required keys: {missing_keys}")
                
            print(f"Loaded {len(data['question'])} evaluation samples from {file_path}")
            return data
            
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            print("Using sample dataset instead")
            return self.create_sample_evaluation_dataset()
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Using sample dataset instead")
            return self.create_sample_evaluation_dataset()
    
    def mock_retrieval_function(self, query: str, top_k: int = 5) -> List[str]:
        """
        Mock retrieval function that simulates document retrieval
        Replace this with your actual RAG retrieval function
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve
            
        Returns:
            List of retrieved document texts
        """
        # This is a simplified mock - replace with your actual retrieval logic
        document_corpus = [
            "In Information Retrieval (IR), three classic metrics are used: Precision@k, Recall@k, and Mean Reciprocal Rank (MRR)",
            "At its core, a RAG pipeline is only as strong as the documents it retrieves",
            "Evaluation gives us a way to quantify retrieval quality instead of just relying on gut feeling",
            "Precision@k - Of the top k retrieved documents, how many are truly relevant?",
            "Recall@k - Out of all relevant documents in the corpus, how many did we retrieve?", 
            "Mean Reciprocal Rank (MRR) - Looks at the position of the first relevant document in the ranked list",
            "Tune Chunk Size - Try 300–600 tokens per chunk",
            "Experiment with Embedding Models - Larger models can improve retrieval",
            "Hybrid Search - Combine keyword + semantic search for rare terms",
            "ANN Parameters - Adjust efSearch (HNSW) for accuracy-speed tradeoff",
            "Without evaluation, RAG is just guesswork",
            "The photovoltaic cell was invented by Charles Fritts",
            "Germany aimed for 80% renewable electricity by 2030"
        ]
        
        # Simple similarity-based retrieval
        query_embedding = self.embedding_model.encode([query])
        doc_embeddings = self.embedding_model.encode(document_corpus)
        
        similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [document_corpus[i] for i in top_indices]
    
    def calculate_precision_at_k(self, retrieved_docs: List[str], 
                                ground_truth: List[str], k: int) -> float:
        """
        Calculate Precision@k
        
        Args:
            retrieved_docs: List of retrieved documents
            ground_truth: List of ground truth relevant documents
            k: Number of top documents to consider
            
        Returns:
            Precision@k score
        """
        if not retrieved_docs or k == 0:
            return 0.0
            
        top_k_docs = retrieved_docs[:k]
        relevant_count = 0
        
        for doc in top_k_docs:
            for gt_doc in ground_truth:
                # Simple overlap check - you might want to use semantic similarity
                if self._text_similarity(doc, gt_doc) > 0.7:
                    relevant_count += 1
                    break
                    
        return relevant_count / min(k, len(top_k_docs))
    
    def calculate_recall_at_k(self, retrieved_docs: List[str],
                             ground_truth: List[str], k: int) -> float:
        """
        Calculate Recall@k
        
        Args:
            retrieved_docs: List of retrieved documents
            ground_truth: List of ground truth relevant documents
            k: Number of top documents to consider
            
        Returns:
            Recall@k score
        """
        if not ground_truth:
            return 0.0
            
        top_k_docs = retrieved_docs[:k]
        relevant_found = 0
        
        for gt_doc in ground_truth:
            for doc in top_k_docs:
                if self._text_similarity(doc, gt_doc) > 0.7:
                    relevant_found += 1
                    break
                    
        return relevant_found / len(ground_truth)
    
    def calculate_mrr(self, retrieved_docs: List[str], ground_truth: List[str]) -> float:
        """
        Calculate Mean Reciprocal Rank
        
        Args:
            retrieved_docs: List of retrieved documents
            ground_truth: List of ground truth relevant documents
            
        Returns:
            MRR score
        """
        if not retrieved_docs or not ground_truth:
            return 0.0
            
        for rank, doc in enumerate(retrieved_docs, 1):
            for gt_doc in ground_truth:
                if self._text_similarity(doc, gt_doc) > 0.7:
                    return 1.0 / rank
                    
        return 0.0
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        embeddings = self.embedding_model.encode([text1, text2])
        return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    
    def evaluate_rag_system(self, dataset: Dict[str, List[str]], 
                           retrieval_function: callable = None,
                           k: int = 3) -> EvaluationResult:
        """
        Evaluate the RAG system comprehensively
        
        Args:
            dataset: Evaluation dataset
            retrieval_function: Function to retrieve documents (uses mock if None)
            k: Number of documents to retrieve and evaluate
            
        Returns:
            EvaluationResult object with all metrics
        """
        if retrieval_function is None:
            retrieval_function = self.mock_retrieval_function
            
        questions = dataset['question']
        answers = dataset['answer']
        ground_truths = dataset['ground_truth_contexts']
        
        precision_scores = []
        recall_scores = []
        mrr_scores = []
        retrieval_times = []
        
        print(f"\n🔄 Evaluating RAG system on {len(questions)} queries...")
        print("=" * 60)
        
        for i, (question, answer, ground_truth) in enumerate(zip(questions, answers, ground_truths)):
            print(f"Query {i+1}/{len(questions)}: {question[:50]}...")
            
            # Measure retrieval time
            start_time = time.time()
            retrieved_docs = retrieval_function(question, top_k=k)
            retrieval_time = time.time() - start_time
            retrieval_times.append(retrieval_time)
            
            # Calculate metrics
            precision = self.calculate_precision_at_k(retrieved_docs, ground_truth, k)
            recall = self.calculate_recall_at_k(retrieved_docs, ground_truth, k)
            mrr = self.calculate_mrr(retrieved_docs, ground_truth)
            
            precision_scores.append(precision)
            recall_scores.append(recall)
            mrr_scores.append(mrr)
            
            print(f"  Precision@{k}: {precision:.3f} | Recall@{k}: {recall:.3f} | MRR: {mrr:.3f} | Time: {retrieval_time:.3f}s")
        
        # Calculate averages
        avg_precision = np.mean(precision_scores)
        avg_recall = np.mean(recall_scores)
        avg_mrr = np.mean(mrr_scores)
        avg_retrieval_time = np.mean(retrieval_times)
        f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
        
        # RAGAS evaluation (if available)
        ragas_scores = None
        if RAGAS_AVAILABLE:
            try:
                ragas_scores = self._evaluate_with_ragas(dataset, retrieval_function, k)
            except Exception as e:
                print(f"RAGAS evaluation failed: {e}")
        
        return EvaluationResult(
            precision_at_k=avg_precision,
            recall_at_k=avg_recall,
            mrr=avg_mrr,
            f1_score=f1_score,
            retrieval_time=avg_retrieval_time,
            total_queries=len(questions),
            ragas_scores=ragas_scores
        )
    
    def _evaluate_with_ragas(self, dataset: Dict[str, List[str]], 
                            retrieval_function: callable, k: int) -> Dict[str, float]:
        """
        Evaluate using RAGAS metrics
        
        Args:
            dataset: Evaluation dataset
            retrieval_function: Function to retrieve documents
            k: Number of documents to retrieve
            
        Returns:
            Dictionary of RAGAS scores
        """
        print("\nRunning RAGAS evaluation...")
        
        # Prepare data for RAGAS
        contexts = []
        for question in dataset['question']:
            retrieved_docs = retrieval_function(question, top_k=k)
            contexts.append(retrieved_docs)
        
        ragas_dataset = Dataset.from_dict({
            'question': dataset['question'],
            'answer': dataset['answer'],
            'contexts': contexts,
            'ground_truth': dataset['answer']  # Using answers as ground truth for RAGAS
        })
        
        # Run evaluation
        metrics = [context_precision, context_recall, faithfulness, answer_relevancy]
        result = evaluate(ragas_dataset, metrics=metrics)
        
        return {
            'context_precision': result['context_precision'],
            'context_recall': result['context_recall'], 
            'faithfulness': result['faithfulness'],
            'answer_relevancy': result['answer_relevancy']
        }
    
    def print_evaluation_report(self, result: EvaluationResult):
        """
        Print a comprehensive evaluation report
        
        Args:
            result: EvaluationResult object
        """
        print("\n" + "=" * 80)
        print("RAG SYSTEM EVALUATION REPORT")
        print("=" * 80)
        
        print(f"\nBASIC METRICS")
        print(f"├── Total Queries Evaluated: {result.total_queries}")
        print(f"├── Average Retrieval Time: {result.retrieval_time:.4f} seconds")
        print(f"├── Precision@k: {result.precision_at_k:.4f}")
        print(f"├── Recall@k: {result.recall_at_k:.4f}")
        print(f"├── F1 Score: {result.f1_score:.4f}")
        print(f"└── Mean Reciprocal Rank: {result.mrr:.4f}")
        
        if result.ragas_scores:
            print(f"\n🎭 RAGAS METRICS")
            for metric, score in result.ragas_scores.items():
                print(f"├── {metric.replace('_', ' ').title()}: {score:.4f}")
        
        print(f"\nINTERPRETATION")
        self._interpret_results(result)
        
        print(f"\n🔧 OPTIMIZATION SUGGESTIONS")
        self._suggest_optimizations(result)
        
        print("\n" + "=" * 80)
    
    def _interpret_results(self, result: EvaluationResult):
        """Interpret and explain the evaluation results"""
        
        if result.precision_at_k > 0.7:
            print("High Precision: Your system returns mostly relevant documents")
        elif result.precision_at_k > 0.5:
            print("Medium Precision: Some irrelevant documents in results")
        else:
            print("Low Precision: Many irrelevant documents retrieved")
            
        if result.recall_at_k > 0.7:
            print("High Recall: Your system finds most relevant documents")
        elif result.recall_at_k > 0.5:
            print("Medium Recall: Missing some relevant documents")
        else:
            print("Low Recall: Many relevant documents not found")
            
        if result.mrr > 0.7:
            print("High MRR: Relevant documents appear early in results")
        elif result.mrr > 0.5:
            print("Medium MRR: Relevant documents sometimes buried")
        else:
            print("Low MRR: Relevant documents appear late in rankings")
    
    def _suggest_optimizations(self, result: EvaluationResult):
        """Provide optimization suggestions based on results"""
        
        if result.precision_at_k < 0.6:
            print("• Consider improving embedding quality or adding reranking")
            print("• Try hybrid search to balance keyword and semantic matching")
            
        if result.recall_at_k < 0.6:
            print("• Increase the number of retrieved documents (k)")
            print("• Experiment with different chunking strategies")
            print("• Consider expanding your document corpus")
            
        if result.mrr < 0.6:
            print("• Implement reranking to improve document ordering")
            print("• Fine-tune your embedding model on domain-specific data")
            
        if result.retrieval_time > 0.1:
            print("• Consider using approximate nearest neighbor (ANN) search")
            print("• Optimize your vector database configuration")
    
    def export_results(self, result: EvaluationResult, filename: str = "rag_evaluation_results.json"):
        """
        Export evaluation results to JSON file
        
        Args:
            result: EvaluationResult object
            filename: Output filename
        """
        export_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": {
                "precision_at_k": result.precision_at_k,
                "recall_at_k": result.recall_at_k,
                "mrr": result.mrr,
                "f1_score": result.f1_score,
                "average_retrieval_time": result.retrieval_time,
                "total_queries": result.total_queries
            }
        }
        
        if result.ragas_scores:
            export_data["ragas_metrics"] = result.ragas_scores
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Results exported to {filename}")

def main():
    """
    Main execution function
    """
    print("RAG Evaluation System Starting...")
    print("=" * 50)
    
    # Initialize evaluator
    evaluator = RAGEvaluator()
    
    # Load evaluation dataset
    # Try to load custom dataset, fallback to sample
    dataset_path = "evaluation_dataset.json"
    if os.path.exists(dataset_path):
        print(f"Found custom dataset: {dataset_path}")
        dataset = evaluator.load_custom_dataset(dataset_path)
    else:
        print("Using built-in sample dataset")
        print(f"To use your own data, create '{dataset_path}' with format:")
        print("   {\"question\": [...], \"answer\": [...], \"ground_truth_contexts\": [...]}")
        dataset = evaluator.create_sample_evaluation_dataset()
    
    # Run evaluation
    print(f"\n🔬 Starting evaluation with {len(dataset['question'])} test cases...")
    
    # You can replace this with your actual retrieval function
    # For example: from your_rag_system import retrieve_documents
    # result = evaluator.evaluate_rag_system(dataset, retrieve_documents, k=5)
    
    result = evaluator.evaluate_rag_system(dataset, k=3)
    
    # Print comprehensive report
    evaluator.print_evaluation_report(result)
    
    # Export results
    evaluator.export_results(result)
    
    print("\n✅ Evaluation completed successfully!")
    print("\n📚 Next steps:")
    print("1. Replace mock_retrieval_function with your actual RAG retrieval")
    print("2. Create custom evaluation datasets for your domain")
    print("3. Run iterative optimization based on the results")
    print("4. Set up automated evaluation in your CI/CD pipeline")

if __name__ == "__main__":
    main()