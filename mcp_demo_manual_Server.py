#!/usr/bin/env python3
"""
Fixed MCP RAG Demo Script
Uses the existing MCP server and client setup.
"""

import time
import subprocess
import sys
from pathlib import Path

# Use the existing mcp_client.py from the same directory
try:
    from utils.mcp_client import MCPRAGClient
except ImportError:
    print("❌ Could not import mcp_client from current directory.")
    print("   Make sure mcp_client.py is in the same folder as this demo.")
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

def check_existing_server():
    """Check if MCP server is already running on port 8000."""
    try:
        client = MCPRAGClient("http://localhost:8000")
        status = client.get_status()
        if "error" not in status.lower():
            print("✅ Found existing MCP server on port 8000")
            return True
        else:
            print("⚠️  MCP server found but not initialized")
            # Try to initialize
            print("🔄 Attempting to initialize...")
            init_result = client.initialize_rag()
            if "error" not in init_result.lower():
                print("✅ Server initialized successfully")
                return True
            else:
                print(f"❌ Failed to initialize: {init_result}")
                return False
    except Exception as e:
        print(f"❌ No MCP server found on port 8000: {e}")
        return False

def demo_basic_functionality():
    """Demonstrate basic MCP RAG functionality."""
    print_header("MCP RAG Demo - Basic Functionality")
    
    # Connect to existing server on port 8000
    client = MCPRAGClient("http://localhost:8000")
    
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
    result = client.search(search_query)
    print(result)
    wait_for_input()
    
    print_step(5, "Testing AI-Generated Answers")
    questions = [
        "What are the main types of renewable energy?",
        "What are the benefits of solar panels?",
        "Tell me about ecommerce platforms"
    ]
    
    for question in questions:
        print(f"\n❓ Question: {question}")
        print("🤖 Generating answer...")
        result = client.answer(question)
        print(result)
        wait_for_input()

def demo_advanced_features():
    """Demonstrate advanced MCP RAG features."""
    print_header("MCP RAG Demo - Advanced Features")
    
    client = MCPRAGClient("http://localhost:8000")
    
    print_step(1, "Testing Different Questions")
    questions = [
        "solar power efficiency",
        "What are wind turbines?",
        "energy production statistics"
    ]
    
    for question in questions:
        print(f"\n🔍 Question: '{question}'")
        result = client.answer(question)
        print(result)
        print("-" * 30)
    wait_for_input()
    
    print_step(2, "RAG + Calculator Combination")
    print("🧠 Demonstrating tool combination...")
    
    # First get some information from RAG
    data_query = "What percentage efficiency do solar panels have?"
    print(f"📊 Question: '{data_query}'")
    search_result = client.answer(data_query)
    print(search_result)
    
    # Then do some calculations
    print(f"\n🧮 Performing calculations...")
    calc_result = client.calculate("1000 * 0.15")  # 15% efficiency calculation
    print(f"If solar panel is 15% efficient: {calc_result}")
    
    calc_result2 = client.calculate("1000 * 0.20")  # 20% efficiency calculation
    print(f"If solar panel is 20% efficient: {calc_result2}")
    wait_for_input()

def interactive_demo():
    """Run an interactive demo session."""
    print_header("Interactive MCP RAG Demo")
    
    client = MCPRAGClient("http://localhost:8000")
    
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
                result = client.answer(user_input)
                print(result)
                
        except KeyboardInterrupt:
            print("\n👋 Exiting interactive demo...")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Run the complete MCP RAG demo."""
    print_header("Welcome to MCP RAG Demo!")
    print("This demo shows how to use MCP (Model Context Protocol) for RAG applications.")
    
    # Check prerequisites
    print("\n🔍 Checking Prerequisites...")
    
    # Check Ollama
    if not check_ollama():
        print("\n❌ Please install Ollama and pull a model first:")
        print("   1. Install Ollama: https://ollama.com")
        print("   2. Pull a model: ollama pull mistral")
        return
    
    # Check existing MCP server
    if not check_existing_server():
        print("\n❌ Please start your MCP server first:")
        print("   1. Open another terminal")
        print("   2. Run your MCP server (usually mcp_rag.py or similar)")
        print("   3. Make sure it's running on http://localhost:8000")
        return
    
    # Check if data files exist
    data_dir = Path("data")
    if not data_dir.exists() or not list(data_dir.glob("*.txt")):
        print("⚠️  No data files found in 'data/' directory")
        print("   Make sure you have .txt files in the data/ folder")
        wait_for_input("Continue anyway?")
    
    wait_for_input("Ready to start demo?")
    
    try:
        # Run demo sections
        demo_basic_functionality()
        
        wait_for_input("Continue to advanced features demo?")
        demo_advanced_features()
        
        wait_for_input("Continue to interactive demo?")
        interactive_demo()
        
        print_header("Demo Complete!")
        print("🎉 You've successfully demonstrated MCP RAG capabilities!")
        print("  ✅ Existing MCP server on localhost:8000")
        print("  ✅ RAG search and answer generation")
        print("  ✅ AI-powered answers grounded in documents")
        print("  ✅ Tool composition (RAG + calculator)")
        print("  ✅ 100% local and free implementation")
        print("  ✅ Standard MCP protocol compatibility")
        
    except KeyboardInterrupt:
        print("\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")

if __name__ == "__main__":
    main()