#!/usr/bin/env python3
"""
Simple MCP RAG Demo Script
Demonstrates the MCP RAG system capabilities.
"""

import asyncio
import time
import subprocess
import sys
import threading
import signal
from pathlib import Path

# Add utils to path for client import
sys.path.append(str(Path(__file__).parent / "utils"))

try:
    from utils.mcp_client import MCPRAGClient
except ImportError:
    print("❌ Could not import mcp_client. Make sure it's in the utils/ directory.")
    sys.exit(1)

def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"🎯 {text}")
    print('='*60)

def print_step(step: int, description: str):
    """Print a step in the demo."""
    print(f"\n{step}️⃣ {description}")
    print("-" * 40)

def wait_for_input(message: str = "Press Enter to continue..."):
    """Wait for user input with a message."""
    print(f"\n⏸️  {message}")
    input()

def check_ollama():
    """Check if Ollama is running and has models."""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            models = result.stdout.strip()
            if models and len(models.split('\n')) > 1:  # More than just header
                print("✅ Ollama is running with models available")
                return True
            else:
                print("⚠️  Ollama is running but no models found")
                print("   Run: ollama pull mistral")
                return False
        else:
            print("❌ Ollama not responding")
            return False
    except Exception as e:
        print(f"❌ Could not check Ollama: {e}")
        print("   Make sure Ollama is installed and running")
        return False

def start_mcp_server():
    """Start the MCP server in background."""
    print("🚀 Starting MCP RAG server in background...")
    try:
        # Start server with auto-init
        process = subprocess.Popen(
            [sys.executable, "mcp_rag.py", "--auto-init", "--port", "8001"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Give server time to start
        time.sleep(3)
        
        # Check if server is still running
        if process.poll() is None:
            print("✅ MCP server started successfully!")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Server failed to start:")
            print(f"   stdout: {stdout}")
            print(f"   stderr: {stderr}")
            return None
            
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return None

def demo_basic_functionality():
    """Demonstrate basic MCP RAG functionality."""
    print_header("MCP RAG Demo - Basic Functionality")
    
    # Connect to server
    client = MCPRAGClient("http://localhost:8001")
    
    print_step(1, "Checking Server Connection & Status")
    status = client.get_status()
    print(status)
    wait_for_input()
    
    print_step(2, "Listing Available Ollama Models")
    models = client.list_models()
    print(models)
    wait_for_input()
    
    print_step(3, "Testing Calculator Tool")
    calc_expressions = ["25 * 8 + 150", "2 ** 10", "math.pi * 2"]
    for expr in calc_expressions:
        print(f"\n🧮 Calculating: {expr}")
        result = client.calculate(expr)
        print(result)
    wait_for_input()
    
    print_step(4, "Testing RAG Search - Semantic Mode")
    search_query = "renewable energy sources"
    print(f"🔍 Searching for: '{search_query}'")
    result = client.search(search_query, mode="semantic", top_k=2)
    print(result)
    wait_for_input()
    
    print_step(5, "Testing RAG Search - Hybrid Mode")
    print(f"🔍 Searching for: '{search_query}' (hybrid mode)")
    result = client.search(search_query, mode="hybrid", top_k=3)
    print(result)
    wait_for_input()
    
    print_step(6, "Testing AI-Generated Answers")
    questions = [
        "What are the main types of renewable energy?",
        "What are the benefits of solar panels?",
        "Tell me about ecommerce platforms"
    ]
    
    for question in questions:
        print(f"\n❓ Question: {question}")
        print("🤖 Generating answer...")
        result = client.answer(question, mode="hybrid", top_k=3)
        print(result)
        wait_for_input()

def demo_advanced_features():
    """Demonstrate advanced MCP RAG features."""
    print_header("MCP RAG Demo - Advanced Features")
    
    client = MCPRAGClient("http://localhost:8001")
    
    print_step(1, "Comparing Search Modes")
    query = "solar power efficiency"
    
    for mode in ["semantic", "keyword", "hybrid"]:
        print(f"\n🔍 {mode.title()} Search for: '{query}'")
        result = client.search(query, mode=mode, top_k=2)
        print(result)
        print("-" * 30)
    wait_for_input()
    
    print_step(2, "Testing Different Chunk Retrieval Sizes")
    question = "What are wind turbines?"
    
    for top_k in [1, 3, 5]:
        print(f"\n🎯 Retrieving top {top_k} chunks")
        result = client.answer(question, top_k=top_k, mode="hybrid")
        print(result)
        print("-" * 30)
    wait_for_input()
    
    print_step(3, "RAG + Calculator Combination")
    print("🧠 Demonstrating tool combination...")
    
    # First get some numerical data from RAG
    data_query = "energy production statistics"
    print(f"📊 Searching for: '{data_query}'")
    search_result = client.search(data_query, top_k=2)
    print(search_result)
    
    # Then do some calculations
    print(f"\n🧮 Performing calculations...")
    calc_result = client.calculate("1000 * 0.15")  # 15% efficiency
    print(calc_result)
    wait_for_input()

def interactive_demo():
    """Run an interactive demo session."""
    print_header("Interactive MCP RAG Demo")
    
    client = MCPRAGClient("http://localhost:8001")
    
    print("🎮 Interactive Mode - You can now test the system!")
    print("\nSuggested commands to try:")
    print("  • Ask questions about renewable energy")
    print("  • Ask questions about ecommerce/laptops") 
    print("  • Try calculations like '50 * 3 + 25'")
    print("  • Type 'help' for available options")
    print("  • Type 'quit' to end demo")
    
    while True:
        try:
            user_input = input("\n💬 Your input: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Exiting interactive demo...")
                break
                
            if user_input.lower() == 'help':
                print("\n🛠️ Available demo commands:")
                print("  • Any question → Get AI answer with RAG")
                print("  • Math expression → Calculate result")
                print("  • 'status' → Check system status")
                print("  • 'models' → List Ollama models")
                print("  • 'quit' → Exit")
                continue
                
            if user_input.lower() == 'status':
                result = client.get_status()
                print(result)
                continue
                
            if user_input.lower() == 'models':
                result = client.list_models()
                print(result)
                continue
            
            # Check if it's a math expression
            if any(op in user_input for op in ['+', '-', '*', '/', '^', '%', '**']) and not user_input.lower().startswith(('what', 'how', 'why', 'tell', 'explain')):
                print("🧮 Calculating...")
                result = client.calculate(user_input)
                print(result)
            else:
                print("🤖 Generating AI answer...")
                result = client.answer(user_input, mode="hybrid", top_k=3)
                print(result)
                
        except KeyboardInterrupt:
            print("\n👋 Exiting interactive demo...")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def cleanup_server(process):
    """Clean up the server process."""
    if process and process.poll() is None:
        print("\n🧹 Cleaning up server...")
        process.terminate()
        try:
            process.wait(timeout=5)
            print("✅ Server stopped cleanly")
        except subprocess.TimeoutExpired:
            process.kill()
            print("🔥 Server force killed")

def main():
    """Run the complete MCP RAG demo."""
    print_header("Welcome to MCP RAG Demo!")
    print("This demo shows how to use MCP (Model Context Protocol) for RAG applications.")
    
    # Check prerequisites
    print("\n🔍 Checking Prerequisites...")
    if not check_ollama():
        print("\n❌ Please install Ollama and pull a model first:")
        print("   1. Install Ollama: https://ollama.com")
        print("   2. Pull a model: ollama pull mistral")
        return
    
    # Check if data files exist
    data_dir = Path("data")
    if not data_dir.exists() or not list(data_dir.glob("*.txt")):
        print("⚠️  No data files found in 'data/' directory")
        print("   Make sure you have .txt files in the data/ folder")
        wait_for_input("Continue anyway?")
    
    wait_for_input("Ready to start demo?")
    
    # Start MCP server
    server_process = start_mcp_server()
    if not server_process:
        print("❌ Could not start MCP server. Exiting...")
        return
    
    try:
        # Wait a bit more for server to fully initialize
        print("⏳ Waiting for server to fully initialize...")
        time.sleep(5)
        
        # Run demo sections
        demo_basic_functionality()
        
        wait_for_input("Continue to advanced features demo?")
        demo_advanced_features()
        
        wait_for_input("Continue to interactive demo?")
        interactive_demo()
        
        print_header("Demo Complete!")
        print("🎉 You've successfully demonstrated MCP RAG capabilities!")
        print("  ✅ Local MCP server with FastMCP")
        print("  ✅ Multiple RAG search modes (semantic, keyword, hybrid)")
        print("  ✅ AI-powered answers grounded in documents")
        print("  ✅ Tool composition (RAG + calculator)")
        print("  ✅ 100 percent local and free implementation")
        print("  ✅ Standard MCP protocol compatibility")
        
    except KeyboardInterrupt:
        print("\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
    finally:
        cleanup_server(server_process)

if __name__ == "__main__":
    main()