"""
MCP RAG Client
A simple client to interact with the MCP RAG server for testing and demonstration.
"""

import asyncio
import json
import argparse
from typing import Dict, Any

try:
    import httpx
except ImportError:
    print("httpx not installed. Run: pip install httpx")
    import sys
    sys.exit(1)

class MCPRAGClient:
    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url.rstrip('/')
        self.client = httpx.Client(timeout=30.0)
    
    def call_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Call an MCP tool on the server."""
        try:
            response = self.client.post(
                f"{self.server_url}/call/{tool_name}",
                json=kwargs
            )
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as e:
            return {"error": f"Request failed: {e}"}
        except httpx.HTTPStatusError as e:
            return {"error": f"HTTP {e.response.status_code}: {e.response.text}"}
        except json.JSONDecodeError:
            return {"error": "Invalid JSON response"}
    
    def initialize_rag(self, **kwargs) -> str:
        """Initialize the RAG system."""
        result = self.call_tool("initialize_rag", **kwargs)
        return result.get("result", result.get("error", "Unknown error"))
    
    def search(self, query: str, **kwargs) -> str:
        """Search the RAG knowledge base."""
        result = self.call_tool("rag_search", query=query, **kwargs)
        return result.get("result", result.get("error", "Unknown error"))
    
    def answer(self, query: str, **kwargs) -> str:
        """Get an AI answer using RAG."""
        result = self.call_tool("rag_answer", query=query, **kwargs)
        return result.get("result", result.get("error", "Unknown error"))
    
    def calculate(self, expression: str) -> str:
        """Perform a calculation."""
        result = self.call_tool("calculator", expression=expression)
        return result.get("result", result.get("error", "Unknown error"))
    
    def get_status(self) -> str:
        """Get RAG system status."""
        result = self.call_tool("rag_status")
        return result.get("result", result.get("error", "Unknown error"))
    
    def list_models(self) -> str:
        """List available Ollama models."""
        result = self.call_tool("list_available_models")
        return result.get("result", result.get("error", "Unknown error"))

def interactive_mode(client: MCPRAGClient):
    """Run interactive mode for testing."""
    print("🤖 MCP RAG Interactive Client")
    print("=" * 50)
    print("Commands:")
    print("  init [args...]    - Initialize RAG system")
    print("  search <query>    - Search knowledge base")
    print("  answer <query>    - Get AI answer")
    print("  calc <expr>       - Calculate expression")
    print("  status           - Show system status")
    print("  models           - List Ollama models")
    print("  help             - Show this help")
    print("  quit             - Exit")
    print("=" * 50)
    
    # Try to auto-initialize
    print("\n🔄 Attempting to auto-initialize RAG system...")
    result = client.initialize_rag()
    print(result)
    
    while True:
        try:
            command = input("\n💬 Enter command: ").strip()
            
            if not command:
                continue
                
            if command.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
                
            parts = command.split(None, 1)
            cmd = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""
            
            if cmd == 'help':
                print("Commands: init, search, answer, calc, status, models, help, quit")
                
            elif cmd == 'init':
                # Parse simple key=value pairs
                kwargs = {}
                if args:
                    for pair in args.split():
                        if '=' in pair:
                            key, value = pair.split('=', 1)
                            # Try to convert to appropriate type
                            if value.lower() in ['true', 'false']:
                                value = value.lower() == 'true'
                            elif value.isdigit():
                                value = int(value)
                            elif value.replace('.', '').isdigit():
                                value = float(value)
                            kwargs[key] = value
                
                result = client.initialize_rag(**kwargs)
                print(result)
                
            elif cmd == 'search':
                if not args:
                    print("❌ Please provide a search query")
                    continue
                result = client.search(args)
                print(result)
                
            elif cmd == 'answer':
                if not args:
                    print("❌ Please provide a question")
                    continue
                result = client.answer(args)
                print(result)
                
            elif cmd == 'calc':
                if not args:
                    print("❌ Please provide a mathematical expression")
                    continue
                result = client.calculate(args)
                print(result)
                
            elif cmd == 'status':
                result = client.get_status()
                print(result)
                
            elif cmd == 'models':
                result = client.list_models()
                print(result)
                
            else:
                print(f"❌ Unknown command: {cmd}. Type 'help' for available commands.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def batch_demo(client: MCPRAGClient):
    """Run a batch demonstration of MCP RAG capabilities."""
    print("🚀 MCP RAG Demonstration")
    print("=" * 50)
    
    # 1. Initialize
    print("\n1️⃣ Initializing RAG system...")
    result = client.initialize_rag()
    print(result)
    
    # 2. Check status
    print("\n2️⃣ Checking system status...")
    result = client.get_status()
    print(result)
    
    # 3. Demo searches
    queries = [
        "What are renewable energy sources?",
        "Tell me about solar panels",
        "What laptops are good for gaming?",
        "Calculate 15 * 23 + 100"
    ]
    
    for i, query in enumerate(queries, 3):
        print(f"\n{i}️⃣ Query: {query}")
        
        if any(op in query for op in ['+', '-', '*', '/', '^', '%']):
            # Use calculator for math
            result = client.calculate(query.replace('Calculate ', ''))
        else:
            # Use RAG for knowledge queries
            result = client.answer(query)
        
        print(result)
    
    print("\n✅ Demonstration complete!")

def main():
    parser = argparse.ArgumentParser(description="MCP RAG Client")
    parser.add_argument("--server", type=str, default="http://localhost:8000", 
                       help="MCP server URL")
    parser.add_argument("--mode", choices=["interactive", "demo"], default="interactive",
                       help="Run mode")
    
    args = parser.parse_args()
    
    print(f"🔗 Connecting to MCP RAG server at {args.server}")
    client = MCPRAGClient(args.server)
    
    if args.mode == "interactive":
        interactive_mode(client)
    else:
        batch_demo(client)

if __name__ == "__main__":
    main()