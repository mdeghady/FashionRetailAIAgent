"""
StylistAI Agent - Usage Examples and Configuration
=================================================

This file demonstrates various ways to use the StylistAI agent, including
configuration options, environment setup, and integration patterns.
"""

import os
import json
from dotenv import load_dotenv
from stylist_ai_agent import StylistAIAgent, AgentConfig

# Load environment variables
load_dotenv()


# =============================================================================
# Environment Configuration Examples
# =============================================================================

def setup_environment():
    """Example of setting up environment variables."""
    
    # Required environment variables
    required_env = {
        "GOOGLE_API_KEY": "your_google_api_key_here",  # Required for Gemini
    }
    
    # Optional environment variables with defaults
    optional_env = {
        "CHROMA_COLLECTION": "fashion_products",
        "CHROMA_PERSIST_DIR": "./chroma_db",
        "RETRIEVAL_K": "4",
        "SIMILARITY_THRESHOLD": "0.7",
        "MAX_CONTEXT_LENGTH": "3000",
        "GEMINI_MODEL": "gemini-pro",
        "GEMINI_TEMPERATURE": "0.7",
        "GEMINI_MAX_TOKENS": "1024",
    }
    
    # Check required environment variables
    missing_vars = []
    for var, description in required_env.items():
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set these in your .env file or environment")
        return False
    
    # Set optional variables if not present
    for var, default in optional_env.items():
        if not os.getenv(var):
            os.environ[var] = default
            print(f"✅ Set {var} to default: {default}")
    
    return True


# =============================================================================
# Configuration Examples
# =============================================================================

def create_custom_config():
    """Example of creating a custom configuration."""
    
    config = AgentConfig(
        # ChromaDB settings
        chroma_collection_name="fashion_products",
        chroma_persist_directory="./data/chroma_db",
        
        # Retrieval settings
        retrieval_k=5,  # Retrieve more documents
        similarity_threshold=0.6,  # Lower threshold for more results
        max_context_length=4000,  # Longer context window
        
        # LLM settings
        gemini_model="gemini-pro",
        gemini_temperature=0.8,  # More creative responses
        gemini_max_tokens=1500,  # Longer responses
        
        # Agent settings
        agent_name="FashionGuru",
        max_retries=2,
    )
    
    return config


def create_production_config():
    """Example of production-ready configuration."""
    
    config = AgentConfig(
        # Optimized for production
        retrieval_k=3,  # Fewer docs for faster response
        similarity_threshold=0.75,  # Higher threshold for quality
        max_context_length=2500,  # Controlled context size
        
        # Conservative LLM settings
        gemini_temperature=0.6,  # Less randomness
        gemini_max_tokens=800,  # Concise responses
        
        # Reliability settings
        max_retries=3,
    )
    
    return config


# =============================================================================
# Basic Usage Examples
# =============================================================================

def basic_usage_example():
    """Demonstrate basic agent usage."""
    
    print("🚀 Basic Usage Example")
    print("=" * 50)
    
    # Initialize agent with default config
    agent = StylistAIAgent()
    
    # Simple query
    query = "What are the latest trends in women's sneakers?"
    result = agent.query(query)
    
    print(f"Query: {query}")
    print(f"Response: {result['response']}")
    print(f"Success: {result['success']}")
    
    if not result['success']:
        print(f"Error: {result.get('error', 'Unknown error')}")
    
    return agent


def advanced_usage_example():
    """Demonstrate advanced agent usage with custom config."""
    
    print("\n🔧 Advanced Usage Example")
    print("=" * 50)
    
    # Custom configuration
    config = create_custom_config()
    agent = StylistAIAgent(config)
    
    # Multiple queries with session tracking
    queries = [
        "I'm looking for a professional outfit for a tech conference.",
        "What colors would work best with my previous request?",
        "Can you suggest specific brands within a $300 budget?"
    ]
    
    session_id = "advanced_session_001"
    
    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: {query}")
        result = agent.query(query, session_id=session_id)
        print(f"Response: {result['response']}")
        
        # Show metadata
        metadata = result['metadata']
        print(f"Retrieved docs: {metadata.get('retrieved_count', 0)}")
        print(f"Context length: {metadata.get('context_length', 0)} chars")


# =============================================================================
# Integration Examples
# =============================================================================

class FashionChatbot:
    """Example chatbot integration with conversation history."""
    
    def __init__(self, config: AgentConfig = None):
        self.agent = StylistAIAgent(config)
        self.conversation_history = {}
    
    def chat(self, user_id: str, message: str) -> dict:
        """Process a chat message with user context."""
        
        # Get or create user session
        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = []
        
        # Add user message to history
        self.conversation_history[user_id].append({
            "role": "user",
            "content": message,
            "timestamp": "2024-01-01T12:00:00Z"  # In real app, use actual timestamp
        })
        
        # Process with agent
        result = self.agent.query(message, session_id=user_id)
        
        # Add agent response to history
        self.conversation_history[user_id].append({
            "role": "assistant",
            "content": result['response'],
            "timestamp": "2024-01-01T12:00:01Z",
            "metadata": result['metadata']
        })
        
        return {
            "response": result['response'],
            "success": result['success'],
            "conversation_length": len(self.conversation_history[user_id]),
            "user_id": user_id
        }
    
    def get_conversation(self, user_id: str) -> list:
        """Get conversation history for a user."""
        return self.conversation_history.get(user_id, [])


def chatbot_integration_example():
    """Demonstrate chatbot integration."""
    
    print("\n💬 Chatbot Integration Example")
    print("=" * 50)
    
    chatbot = FashionChatbot()
    user_id = "user_123"
    
    # Simulate conversation
    messages = [
        "Hi! I need help choosing an outfit for a wedding.",
        "It's an outdoor summer wedding.",
        "What about shoes to go with a floral dress?"
    ]
    
    for message in messages:
        response = chatbot.chat(user_id, message)
        print(f"\nUser: {message}")
        print(f"Bot: {response['response']}")
        print(f"Conversation length: {response['conversation_length']}")
    
    # Show full conversation
    print(f"\nFull conversation history:")
    history = chatbot.get_conversation(user_id)
    for i, msg in enumerate(history, 1):
        role = "👤" if msg['role'] == 'user' else "🤖"
        print(f"{i}. {role} {msg['content'][:100]}...")


# =============================================================================
# API Integration Example
# =============================================================================

async def create_fastapi_integration():
    """Example FastAPI integration (async version)."""
    
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    
    app = FastAPI(title="StylistAI API", version="1.0.0")
    
    # Global agent instance
    agent = StylistAIAgent()
    
    class QueryRequest(BaseModel):
        query: str
        session_id: str = "default"
        config_overrides: dict = {}
    
    class QueryResponse(BaseModel):
        query: str
        response: str
        success: bool
        metadata: dict
        error: str = None
    
    @app.post("/query", response_model=QueryResponse)
    async def process_query(request: QueryRequest):
        """Process a fashion query."""
        try:
            result = agent.query(request.query, request.session_id)
            return QueryResponse(**result)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "agent": "StylistAI"}
    
    @app.get("/config")
    async def get_config():
        """Get current agent configuration."""
        return {
            "collection_name": agent.config.chroma_collection_name,
            "retrieval_k": agent.config.retrieval_k,
            "model": agent.config.gemini_model,
            "agent_name": agent.config.agent_name
        }
    
    return app


# =============================================================================
# Testing and Debugging Examples
# =============================================================================

def debug_workflow_example():
    """Example of debugging the agent workflow."""
    
    print("\n🔍 Debug Workflow Example")
    print("=" * 50)
    
    # Enable debug logging
    import logging
    logging.getLogger().setLevel(logging.DEBUG)
    
    agent = StylistAIAgent()
    query = "Show me budget-friendly winter coats"
    session_id = "debug_session"
    
    # Process query
    result = agent.query(query, session_id)
    
    # Get detailed workflow state
    workflow_state = agent.get_workflow_state(session_id)
    
    print(f"Query: {query}")
    print(f"Final Response: {result['response']}")
    print(f"\nWorkflow State:")
    print(json.dumps(workflow_state, indent=2, default=str))
    
    print(f"\nMetadata:")
    print(json.dumps(result['metadata'], indent=2))


def performance_testing_example():
    """Example of performance testing."""
    
    import time
    
    print("\n⚡ Performance Testing Example")
    print("=" * 50)
    
    agent = StylistAIAgent()
    
    test_queries = [
        "What are trending sneaker colors?",
        "Best formal shoes under $200?",
        "Summer dress recommendations?",
        "Sustainable fashion brands?",
        "Color matching tips for accessories?"
    ]
    
    total_time = 0
    results = []
    
    for query in test_queries:
        start_time = time.time()
        result = agent.query(query)
        end_time = time.time()
        
        query_time = end_time - start_time
        total_time += query_time
        
        results.append({
            "query": query,
            "response_time": query_time,
            "success": result['success'],
            "retrieved_docs": result['metadata'].get('retrieved_count', 0),
            "response_length": len(result['response'])
        })
        
        print(f"Query: {query[:30]}... | Time: {query_time:.2f}s | Success: {result['success']}")
    
    avg_time = total_time / len(test_queries)
    print(f"\nAverage response time: {avg_time:.2f}s")
    print(f"Total time: {total_time:.2f}s")
    
    return results


# =============================================================================
# Error Handling Examples
# =============================================================================

def error_handling_example():
    """Demonstrate error handling patterns."""
    
    print("\n🚨 Error Handling Example")
    print("=" * 50)
    
    # Test with invalid configuration
    try:
        # This might fail if ChromaDB collection doesn't exist
        config = AgentConfig(chroma_collection_name="non_existent_collection")
        agent = StylistAIAgent(config)
    except Exception as e:
        print(f"Expected configuration error: {e}")
    
    # Test with valid agent but problematic query
    agent = StylistAIAgent()  # Use default config
    
    # Test various edge cases
    edge_cases = [
        "",  # Empty query
        "x" * 10000,  # Very long query
        "🎨👗💄",  # Emoji-only query
        "What is 2+2?",  # Non-fashion query
    ]
    
    for query in edge_cases:
        result = agent.query(query)
        print(f"\nEdge case: '{query[:30]}{'...' if len(query) > 30 else ''}'")
        print(f"Success: {result['success']}")
        print(f"Response length: {len(result['response'])}")
        if not result['success']:
            print(f"Error: {result.get('error', 'Unknown')}")


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Run all examples."""
    
    print("🎨 StylistAI Agent - Usage Examples")
    print("=" * 60)
    
    # Setup environment
    if not setup_environment():
        print("❌ Environment setup failed. Please check your configuration.")
        return
    
    try:
        # Run examples
        basic_usage_example()
        advanced_usage_example()
        chatbot_integration_example()
        debug_workflow_example()
        performance_results = performance_testing_example()
        error_handling_example()
        
        print("\n✅ All examples completed successfully!")
        
        # Summary
        print(f"\n📊 Performance Summary:")
        avg_time = sum(r['response_time'] for r in performance_results) / len(performance_results)
        success_rate = sum(1 for r in performance_results if r['success']) / len(performance_results) * 100
        
        print(f"Average Response Time: {avg_time:.2f}s")
        print(f"Success Rate: {success_rate:.1f}%")
        
    except Exception as e:
        print(f"❌ Example execution failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()