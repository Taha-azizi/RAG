#!/usr/bin/env python3
"""
Advanced RAG with ANN (FAISS) and Reranking
Demonstrates high-performance semantic search with cross-encoder reranking
"""

import argparse
import logging
import time
from pathlib import Path
import sys
from typing import List


# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from utils.product_data import PRODUCT_CATALOG, get_products_by_category
from utils.ann_retriever import FAISSANNRetriever
from utils.reranker import CrossEncoderReranker, HybridReranker
from utils.advanced_rag_utils import AdvancedRAGPipeline, PerformanceBenchmark, create_test_queries, format_search_results

# Try to import existing utilities for compatibility
try:
    from utils.generation import OllamaClient
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("⚠️  Ollama client not available. Answer generation will be limited.")

def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def create_pipeline(args) -> AdvancedRAGPipeline:
    """Create and initialize the advanced RAG pipeline."""
    
    print("🔧 Setting up Advanced RAG Pipeline...")
    
    # 1. Initialize ANN Retriever
    print(f"📊 Creating {args.index_type.upper()} index with {len(PRODUCT_CATALOG)} products...")
    ann_retriever = FAISSANNRetriever(
        model_name=args.embedding_model,
        index_type=args.index_type,
        m=args.hnsw_m,
        ef_construction=args.ef_construction,
        ef_search=args.ef_search
    )
    
    # Build index
    start_time = time.time()
    ann_retriever.build_index(PRODUCT_CATALOG, show_progress=True)
    build_time = time.time() - start_time
    print(f"✅ Index built in {build_time:.2f} seconds")
    
    # 2. Initialize Reranker
    print(f"🧠 Loading reranker model: {args.reranker_model}")
    if args.use_hybrid_reranker:
        reranker = HybridReranker(
            cross_encoder_model=args.reranker_model,
            ann_weight=args.ann_weight,
            cross_encoder_weight=args.ce_weight
        )
    else:
        reranker = CrossEncoderReranker(args.reranker_model)
    print("✅ Reranker loaded")
    
    # 3. Initialize LLM client if available
    llm_client = None
    if OLLAMA_AVAILABLE and args.use_ollama:
        try:
            llm_client = OllamaClient(model=args.ollama_model)
            print(f"✅ Ollama client initialized with model: {args.ollama_model}")
        except Exception as e:
            print(f"⚠️  Could not initialize Ollama: {e}")
    
    # 4. Create pipeline
    pipeline = AdvancedRAGPipeline(
        ann_retriever=ann_retriever,
        reranker=reranker,
        llm_client=llm_client,
        ann_top_k=args.ann_top_k,
        final_top_k=args.final_top_k,
        min_relevance_score=args.min_score
    )
    
    print("🚀 Advanced RAG Pipeline ready!")
    return pipeline

def demo_basic_search(pipeline: AdvancedRAGPipeline):
    """Demonstrate basic ANN + reranking search."""
    print("\n" + "="*60)
    print("🔍 DEMO: Basic ANN + Reranking Search")
    print("="*60)
    
    test_queries = [
        "best laptop for gaming",
        "wireless noise cancelling headphones", 
        "smartphone with good camera",
        "ergonomic office chair",
        "fitness tracker with heart rate monitor"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: '{query}'")
        print("-" * 40)
        
        # Time the search
        start_time = time.time()
        results = pipeline.search(query, return_scores=True)
        search_time = time.time() - start_time
        
        print(f"⏱️  Search time: {search_time*1000:.1f}ms")
        print(f"📊 Results: {len(results)}")
        
        if results:
            print("\n🏆 Top Results:")
            formatted = format_search_results(results, show_scores=True, max_results=3)
            print(formatted)
        
        input("\nPress Enter to continue...")

def demo_comparison(pipeline: AdvancedRAGPipeline):
    """Compare ANN vs exact search performance."""
    print("\n" + "="*60)
    print("⚡ DEMO: Performance Comparison")
    print("="*60)
    
    # Create test queries
    test_queries = create_test_queries()[:10]  # Use first 10 for demo
    
    print(f"🧪 Testing with {len(test_queries)} queries...")
    
    # Simple keyword baseline for comparison
    def baseline_search(query: str) -> List[str]:
        query_lower = query.lower()
        matches = []
        for product in PRODUCT_CATALOG:
            if any(word in product.lower() for word in query_lower.split()):
                matches.append(product)
        return matches[:5]  # Return top 5
    
    # Run comparison
    start_time = time.time()
    comparison = PerformanceBenchmark.compare_with_baseline(
        pipeline, baseline_search, test_queries
    )
    comparison_time = time.time() - start_time
    
    print(f"⏱️  Comparison completed in {comparison_time:.2f}s")
    print("\n📊 Results:")
    print(f"   Advanced Pipeline: {comparison['advanced']['total_avg_ms']:.1f}ms avg")
    print(f"   Baseline Search:   {comparison['baseline']['avg_time_ms']:.1f}ms avg")
    print(f"   🚀 Speedup: {comparison['speedup']:.1f}x")
    print(f"   📈 Throughput improvement: {comparison['throughput_improvement']:.1f}x")

def demo_answer_generation(pipeline: AdvancedRAGPipeline):
    """Demonstrate RAG answer generation."""
    print("\n" + "="*60)
    print("🤖 DEMO: RAG Answer Generation")
    print("="*60)
    
    if pipeline.llm_client is None:
        print("⚠️  LLM not available. Skipping answer generation demo.")
        return
    
    questions = [
        "What's the best gaming laptop with RTX GPU?",
        "I need wireless headphones with noise cancellation",
        "What fitness trackers have heart rate monitoring?",
        "Recommend a good smartphone for photography"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n{i}. Question: {question}")
        print("-" * 50)
        
        start_time = time.time()
        answer = pipeline.answer(question)
        answer_time = time.time() - start_time
        
        print(f"⏱️  Generation time: {answer_time:.2f}s")
        print(f"🤖 Answer:\n{answer}")
        
        input("\nPress Enter for next question...")

def interactive_mode(pipeline: AdvancedRAGPipeline):
    """Run interactive search and Q&A mode."""
    print("\n" + "="*60)
    print("💬 Interactive Mode")
    print("="*60)
    print("Commands:")
    print("  search <query>  - Search products")
    print("  ask <question>  - Get AI answer (requires Ollama)")
    print("  stats          - Show pipeline statistics")
    print("  benchmark      - Run performance benchmark")
    print("  help           - Show this help")
    print("  quit           - Exit")
    print("="*60)
    
    while True:
        try:
            user_input = input("\n💭 Enter command: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            parts = user_input.split(None, 1)
            command = parts[0].lower()
            query = parts[1] if len(parts) > 1 else ""
            
            if command == 'help':
                print("Available commands: search, ask, stats, benchmark, help, quit")
                
            elif command == 'search':
                if not query:
                    print("❌ Please provide a search query")
                    continue
                
                print(f"🔍 Searching for: '{query}'")
                start_time = time.time()
                results = pipeline.search(query, return_scores=True)
                search_time = time.time() - start_time
                
                print(f"⏱️  Search time: {search_time*1000:.1f}ms")
                
                if results:
                    formatted = format_search_results(results, show_scores=True)
                    print(f"\n📊 Results ({len(results)} found):")
                    print(formatted)
                else:
                    print("❌ No results found")
            
            elif command == 'ask':
                if not query:
                    print("❌ Please provide a question")
                    continue
                
                if pipeline.llm_client is None:
                    print("❌ LLM not available. Use 'search' command instead.")
                    continue
                
                print(f"🤖 Answering: '{query}'")
                start_time = time.time()
                answer = pipeline.answer(query)
                answer_time = time.time() - start_time
                
                print(f"⏱️  Generation time: {answer_time:.2f}s")
                print(f"💬 Answer:\n{answer}")
            
            elif command == 'stats':
                stats = pipeline.get_pipeline_stats()
                print("\n📊 Pipeline Statistics:")
                print(f"   ANN Top-K: {stats['ann_top_k']}")
                print(f"   Final Top-K: {stats['final_top_k']}")
                print(f"   Min Score: {stats['min_relevance_score']}")
                print(f"   Index Type: {stats['ann_retriever']['index_type']}")
                print(f"   Total Vectors: {stats['ann_retriever']['total_vectors']}")
                print(f"   LLM Available: {stats['llm_available']}")
            
            elif command == 'benchmark':
                print("🧪 Running benchmark...")
                test_queries = create_test_queries()[:5]  # Quick benchmark
                
                start_time = time.time()
                benchmark_results = PerformanceBenchmark.benchmark_retrieval(
                    pipeline, test_queries
                )
                benchmark_time = time.time() - start_time
                
                print(f"⏱️  Benchmark completed in {benchmark_time:.2f}s")
                print(f"📊 Average search time: {benchmark_results['total_avg_ms']:.1f}ms")
                print(f"🚀 Throughput: {benchmark_results['throughput_qps']:.1f} queries/sec")
            
            else:
                print(f"❌ Unknown command: {command}")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(description="Advanced RAG with ANN and Reranking")
    
    # Model configuration
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2",
                       help="Sentence transformer model for embeddings")
    parser.add_argument("--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2",
                       help="Cross-encoder model for reranking")
    parser.add_argument("--ollama-model", default="mistral", 
                       help="Ollama model for answer generation")
    
    # Index configuration
    parser.add_argument("--index-type", choices=["hnsw", "ivf", "flat"], default="hnsw",
                       help="FAISS index type")
    parser.add_argument("--hnsw-m", type=int, default=32,
                       help="HNSW parameter: number of neighbors per node")
    parser.add_argument("--ef-construction", type=int, default=200,
                       help="HNSW parameter: construction time")
    parser.add_argument("--ef-search", type=int, default=50,
                       help="HNSW parameter: search time")
    
    # Pipeline configuration
    parser.add_argument("--ann-top-k", type=int, default=20,
                       help="Number of candidates from ANN retrieval")
    parser.add_argument("--final-top-k", type=int, default=5,
                       help="Number of final results after reranking")
    parser.add_argument("--min-score", type=float, default=0.1,
                       help="Minimum relevance score threshold")
    
    # Hybrid reranking
    parser.add_argument("--use-hybrid-reranker", action="store_true",
                       help="Use hybrid reranker combining ANN and cross-encoder scores")
    parser.add_argument("--ann-weight", type=float, default=0.3,
                       help="Weight for ANN scores in hybrid reranking")
    parser.add_argument("--ce-weight", type=float, default=0.7,
                       help="Weight for cross-encoder scores in hybrid reranking")
    
    # Runtime options
    parser.add_argument("--mode", choices=["demo", "interactive", "benchmark"], 
                       default="demo", help="Run mode")
    parser.add_argument("--use-ollama", action="store_true", default=True,
                       help="Enable Ollama for answer generation")
    parser.add_argument("--log-level", default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup
    setup_logging(args.log_level)
    print("🚀 Advanced RAG with ANN + Reranking")
    print("="*60)
    
    # Create pipeline
    pipeline = create_pipeline(args)
    
    # Run selected mode
    if args.mode == "demo":
        demo_basic_search(pipeline)
        demo_comparison(pipeline)
        if OLLAMA_AVAILABLE and args.use_ollama:
            demo_answer_generation(pipeline)
        
        # Ask if user wants interactive mode
        response = input("\n🎮 Switch to interactive mode? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            interactive_mode(pipeline)
    
    elif args.mode == "interactive":
        interactive_mode(pipeline)
    
    elif args.mode == "benchmark":
        print("🧪 Running comprehensive benchmark...")
        test_queries = create_test_queries()
        results = PerformanceBenchmark.benchmark_retrieval(pipeline, test_queries)
        
        print("\n📊 Benchmark Results:")
        print(f"   Queries tested: {results['num_queries']}")
        print(f"   Average ANN time: {results['ann_avg_ms']:.1f}ms")
        print(f"   Average rerank time: {results['rerank_avg_ms']:.1f}ms")
        print(f"   Average total time: {results['total_avg_ms']:.1f}ms")
        print(f"   Throughput: {results['throughput_qps']:.1f} queries/second")

if __name__ == "__main__":
    main()