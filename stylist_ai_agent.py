"""
StylistAI Agent - Production-Quality Fashion RAG System
======================================================

A modular AI agent that provides fashion advice, pricing trends, and style recommendations
using ChromaDB retrieval and Gemini LLM integration via LangChain and LangGraph.

Architecture:
- Retrieval Layer: ChromaDB integration with configurable search parameters
- Prompt Engineering: Template-based context construction with token management
- LLM Integration: Gemini via LangChain with error handling and retries
- Workflow Orchestration: LangGraph for transparent, extensible agent flow
- Logging & Monitoring: Comprehensive logging for debugging and performance tracking
"""

import os
import logging
from typing import Dict, List, Optional, Any, TypedDict
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json

import chromadb
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate


from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration and Data Models
# =============================================================================

@dataclass
class AgentConfig:
    """Configuration settings for the StylistAI agent."""
    
    # ChromaDB settings
    chroma_collection_name: str = "fashion_products"
    chroma_persist_directory: Optional[str] = None
    
    # Retrieval settings
    retrieval_k: int = 4
    similarity_threshold: float = 0.7
    max_context_length: int = 3000
    
    # LLM settings
    gemini_model: str = "gemini-pro"
    gemini_temperature: float = 0.7
    gemini_max_tokens: int = 1024
    
    # Agent settings
    agent_name: str = "StylistAI"
    max_retries: int = 3
    
    @classmethod
    def from_env(cls) -> 'AgentConfig':
        """Create configuration from environment variables."""
        return cls(
            chroma_collection_name=os.getenv('CHROMA_COLLECTION', 'fashion_products'),
            chroma_persist_directory=os.getenv('CHROMA_PERSIST_DIR'),
            retrieval_k=int(os.getenv('RETRIEVAL_K', '4')),
            similarity_threshold=float(os.getenv('SIMILARITY_THRESHOLD', '0.7')),
            max_context_length=int(os.getenv('MAX_CONTEXT_LENGTH', '3000')),
            gemini_model=os.getenv('GEMINI_MODEL', 'gemini-pro'),
            gemini_temperature=float(os.getenv('GEMINI_TEMPERATURE', '0.7')),
            gemini_max_tokens=int(os.getenv('GEMINI_MAX_TOKENS', '1024')),
        )


class AgentState(TypedDict):
    """State definition for the LangGraph agent workflow."""
    query: str
    retrieved_docs: List[Document]
    context: str
    prompt: str
    response: str
    metadata: Dict[str, Any]
    error: Optional[str]


# =============================================================================
# Abstract Base Classes for Extensibility
# =============================================================================

class BaseRetriever(ABC):
    """Abstract base class for document retrievers."""
    
    @abstractmethod
    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        """Retrieve relevant documents for the given query."""
        pass


class BasePromptBuilder(ABC):
    """Abstract base class for prompt builders."""
    
    @abstractmethod
    def build_prompt(self, query: str, context: str) -> str:
        """Build a prompt from query and retrieved context."""
        pass


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a response from the given prompt."""
        pass


# =============================================================================
# Concrete Implementations
# =============================================================================

class ChromaRetriever(BaseRetriever):
    """ChromaDB-based document retriever with embedding support."""
    
    def __init__(self, config: AgentConfig):
        """Initialize ChromaDB retriever."""
        self.config = config
        self.embeddings = SentenceTransformerEmbeddings(model_name="multi-qa-MiniLM-L6-cos-v1")
        self._setup_chroma_client()
        logger.info(f"Initialized ChromaRetriever for collection: {config.chroma_collection_name}")
    
    def _setup_chroma_client(self):
        """Set up ChromaDB client and collection."""
        try:
            # Initialize Chroma client
            logger.info(f"Setting up ChromaDB Persistent client in directory : {self.config.chroma_persist_directory}...")
            if self.config.chroma_persist_directory:
                self.chroma_client = chromadb.PersistentClient(path=self.config.chroma_persist_directory)
            else:
                self.chroma_client = chromadb.Client()
            
            # Get existing collection
            self.collection = self.chroma_client.get_collection(name=self.config.chroma_collection_name)
            
            # Create LangChain Chroma wrapper
            self.vectorstore = Chroma(
                client=self.chroma_client,
                collection_name=self.config.chroma_collection_name,
                embedding_function=self.embeddings,
            )
            
            logger.info(f"Connected to ChromaDB collection: {self.config.chroma_collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to setup ChromaDB client: {e}")
            raise
    
    def retrieve(self, query: str, k: int = None) -> List[Document]:
        """Retrieve relevant documents using similarity search."""
        if k is None:
            k = self.config.retrieval_k
            
        try:
            logger.info(f"Retrieving {k} documents for query: '{query[:50]}...'")
            
            # Perform similarity search
            docs = self.vectorstore.similarity_search_with_score(
                query=query,
                k=k
            )
            
            # Filter by similarity threshold and convert to Document objects
            filtered_docs = []
            for doc, score in docs:
                # Convert distance to similarity (assuming lower distance = higher similarity)
                similarity = 1 - score if score <= 1 else 1 / (1 + score)
                
                if similarity >= self.config.similarity_threshold:
                    # Add similarity score to metadata
                    doc.metadata['similarity_score'] = similarity
                    filtered_docs.append(doc)
                    logger.debug(f"Retrieved doc with similarity {similarity:.3f}")
                else:
                    logger.debug(f"Filtered out doc with low similarity {similarity:.3f}")
            
            logger.info(f"Retrieved {len(filtered_docs)} documents above threshold")
            return filtered_docs
            
        except Exception as e:
            logger.error(f"Failed to retrieve documents: {e}")
            return []


class StylistPromptBuilder(BasePromptBuilder):
    """Fashion-focused prompt builder with context management."""
    
    def __init__(self, config: AgentConfig):
        """Initialize prompt builder with configuration."""
        self.config = config
        self.prompt_template = self._create_prompt_template()
        logger.info("Initialized StylistPromptBuilder")
    
    def _create_prompt_template(self) -> PromptTemplate:
        """Create the main prompt template for fashion queries."""
        template = """You are {agent_name}, a knowledgeable and friendly fashion stylist with expertise in:
- Fashion trends and style advice
- Product pricing and market analysis  
- Brand comparisons and recommendations
- Seasonal fashion guidance
- Personal styling tips

Your personality: Professional yet approachable, fashion-savvy, and always up-to-date with current trends and pricing.

CONTEXT FROM FASHION DATABASE:
{context}

USER QUESTION: {query}

INSTRUCTIONS:
1. Use the provided context to give accurate, data-driven insights
2. If discussing prices, mention specific figures when available in the context
3. Provide actionable fashion advice and styling tips
4. Compare different options when relevant
5. Be conversational and engaging while maintaining professionalism
6. If the context doesn't contain sufficient information, acknowledge this and provide general fashion guidance

RESPONSE:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["agent_name", "context", "query"]
        )
    
    def build_prompt(self, query: str, context: str) -> str:
        """Build a complete prompt from query and context."""
        try:
            # Truncate context if it's too long
            if len(context) > self.config.max_context_length:
                context = context[:self.config.max_context_length] + "...\n[Context truncated for length]"
                logger.warning(f"Context truncated to {self.config.max_context_length} characters")
            
            prompt = self.prompt_template.format(
                agent_name=self.config.agent_name,
                context=context,
                query=query
            )
            
            logger.debug(f"Built prompt with {len(prompt)} characters")
            return prompt
            
        except Exception as e:
            logger.error(f"Failed to build prompt: {e}")
            return f"Error building prompt. Query: {query}"
    
    def format_context(self, documents: List[Document]) -> str:
        """Format retrieved documents into context string."""
        if not documents:
            return "No relevant fashion data found in the database."
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            # Extract key metadata
            metadata = doc.metadata
            similarity = metadata.get('similarity_score', 'N/A')
            
            # Format document content
            doc_context = f"--- Product {i} (Similarity: {similarity:.3f}) ---\n"
            doc_context += doc.page_content
            
            # Add relevant metadata
            if 'brand' in metadata:
                doc_context += f"\nBrand: {metadata['brand']}"
            if 'current_price' in metadata:
                doc_context += f"\nPrice: {metadata['currency']} {metadata['current_price']}"
            if 'category' in metadata:
                doc_context += f"\nCategory: {metadata['category']}"
            
            context_parts.append(doc_context)
        
        return "\n\n".join(context_parts)


class GeminiClient(BaseLLMClient):
    """Gemini LLM client with error handling and retries."""
    
    def __init__(self, config: AgentConfig):
        """Initialize Gemini client."""
        self.config = config
        self._setup_llm()
        logger.info(f"Initialized GeminiClient with model: {config.gemini_model}")
    
    def _setup_llm(self):
        """Set up Gemini LLM instance."""
        try:
            # Ensure Google API key is available
            if not os.getenv('GOOGLE_API_KEY'):
                raise ValueError("GOOGLE_API_KEY environment variable is required")
            
            self.llm = ChatGoogleGenerativeAI(
                model=self.config.gemini_model,
                temperature=self.config.gemini_temperature,
                max_tokens=self.config.gemini_max_tokens,
                convert_system_message_to_human=True  # Required for Gemini
            )
            
            logger.info("Successfully initialized Gemini LLM")
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini LLM: {e}")
            raise
    
    def generate(self, prompt: str) -> str:
        """Generate response from Gemini with retry logic."""
        for attempt in range(self.config.max_retries):
            try:
                logger.info(f"Generating response (attempt {attempt + 1}/{self.config.max_retries})")
                
                response = self.llm.invoke(prompt)
                
                # Extract content from response
                if hasattr(response, 'content'):
                    result = response.content
                else:
                    result = str(response)
                
                logger.info(f"Successfully generated response ({len(result)} characters)")
                return result
                
            except Exception as e:
                logger.warning(f"LLM generation attempt {attempt + 1} failed: {e}")
                if attempt == self.config.max_retries - 1:
                    logger.error("All LLM generation attempts failed")
                    return f"I apologize, but I'm having trouble generating a response right now. Please try again later. (Error: {str(e)})"
        
        return "Unable to generate response after multiple attempts."


# =============================================================================
# Main Agent Implementation
# =============================================================================

class StylistAIAgent:
    """Main StylistAI agent class orchestrating the RAG workflow."""
    
    def __init__(self, config: AgentConfig = None):
        """Initialize the StylistAI agent."""
        self.config = config or AgentConfig.from_env()
        
        # Initialize components
        self.retriever = ChromaRetriever(self.config)
        self.prompt_builder = StylistPromptBuilder(self.config)
        self.llm_client = GeminiClient(self.config)
        
        # Initialize LangGraph workflow
        self.workflow = self._create_workflow()
        self.app = self.workflow.compile(checkpointer=MemorySaver())
        
        logger.info("StylistAI Agent initialized successfully")
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow for the agent."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("retrieve", self._retrieve_step)
        workflow.add_node("build_context", self._build_context_step)
        workflow.add_node("generate_response", self._generate_response_step)
        
        # Define edges
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "build_context")
        workflow.add_edge("build_context", "generate_response")
        workflow.add_edge("generate_response", END)
        
        return workflow
    
    def _retrieve_step(self, state: AgentState) -> AgentState:
        """Retrieve relevant documents for the query."""
        logger.info("Executing retrieval step")
        
        try:
            query = state["query"]
            docs = self.retriever.retrieve(query)
            
            state["retrieved_docs"] = docs
            state["metadata"] = {
                **state.get("metadata", {}),
                "retrieved_count": len(docs),
                "retrieval_successful": True
            }
            
            logger.info(f"Retrieval step completed: {len(docs)} documents retrieved")
            
        except Exception as e:
            logger.error(f"Retrieval step failed: {e}")
            state["retrieved_docs"] = []
            state["error"] = f"Retrieval failed: {str(e)}"
            state["metadata"] = {
                **state.get("metadata", {}),
                "retrieved_count": 0,
                "retrieval_successful": False
            }
        
        return state
    
    def _build_context_step(self, state: AgentState) -> AgentState:
        """Build context and prompt from retrieved documents."""
        logger.info("Executing context building step")
        
        try:
            query = state["query"]
            docs = state["retrieved_docs"]
            
            # Format context from documents
            context = self.prompt_builder.format_context(docs)
            
            # Build final prompt
            prompt = self.prompt_builder.build_prompt(query, context)
            
            state["context"] = context
            state["prompt"] = prompt
            state["metadata"] = {
                **state.get("metadata", {}),
                "context_length": len(context),
                "prompt_length": len(prompt)
            }
            
            logger.info("Context building step completed")
            
        except Exception as e:
            logger.error(f"Context building step failed: {e}")
            state["context"] = "Error building context"
            state["prompt"] = f"Error: {str(e)}"
            state["error"] = f"Context building failed: {str(e)}"
        
        return state
    
    def _generate_response_step(self, state: AgentState) -> AgentState:
        """Generate final response using Gemini."""
        logger.info("Executing response generation step")
        
        try:
            prompt = state["prompt"]
            response = self.llm_client.generate(prompt)
            
            state["response"] = response
            state["metadata"] = {
                **state.get("metadata", {}),
                "response_length": len(response),
                "generation_successful": True
            }
            
            logger.info("Response generation step completed")
            
        except Exception as e:
            logger.error(f"Response generation step failed: {e}")
            state["response"] = f"I apologize, but I encountered an error while generating your response: {str(e)}"
            state["error"] = f"Generation failed: {str(e)}"
            state["metadata"] = {
                **state.get("metadata", {}),
                "generation_successful": False
            }
        
        return state
    
    def query(self, user_query: str, session_id: str = "default") -> Dict[str, Any]:
        """Process a user query and return the agent's response."""
        logger.info(f"Processing query: '{user_query[:50]}...'")
        
        # Initialize state
        initial_state = AgentState(
            query=user_query,
            retrieved_docs=[],
            context="",
            prompt="",
            response="",
            metadata={"session_id": session_id},
            error=None
        )
        
        try:
            # Execute workflow
            final_state = self.app.invoke(initial_state, config={"configurable": {"thread_id": session_id}})
            
            # Prepare response
            result = {
                "query": user_query,
                "response": final_state["response"],
                "metadata": final_state["metadata"],
                "success": final_state.get("error") is None
            }
            
            if final_state.get("error"):
                result["error"] = final_state["error"]
            
            logger.info("Query processing completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return {
                "query": user_query,
                "response": f"I apologize, but I encountered an unexpected error: {str(e)}",
                "metadata": {"session_id": session_id},
                "success": False,
                "error": str(e)
            }
    
    def get_workflow_state(self, session_id: str = "default") -> Dict[str, Any]:
        """Get the current workflow state for debugging."""
        try:
            state = self.app.get_state(config={"configurable": {"thread_id": session_id}})
            return state.values if state else {}
        except Exception as e:
            logger.error(f"Failed to get workflow state: {e}")
            return {}


# =============================================================================
# CLI Interface and Usage Examples
# =============================================================================

def create_cli_interface():
    """Create a simple CLI interface for testing the agent."""
    import argparse
    
    parser = argparse.ArgumentParser(description="StylistAI Fashion Assistant")
    parser.add_argument("--query", "-q", type=str, help="Fashion query to ask")
    parser.add_argument("--interactive", "-i", action="store_true", help="Start interactive mode")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug logging")
    
    return parser


def interactive_mode(agent: StylistAIAgent):
    """Run the agent in interactive mode."""
    print(f"\n🎨 Welcome to {agent.config.agent_name}!")
    print("Your personal fashion assistant. Ask me about styles, trends, and prices!")
    print("Type 'quit', 'exit', or 'bye' to end the conversation.\n")
    
    session_id = "interactive_session"
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                print(f"\n👗 {agent.config.agent_name}: Thanks for chatting! Stay stylish! ✨")
                break
            
            if not user_input:
                continue
            
            print(f"\n🤔 {agent.config.agent_name} is thinking...")
            
            # Process query
            result = agent.query(user_input, session_id)
            
            # Display response
            print(f"\n👗 {agent.config.agent_name}: {result['response']}")
            
            # Show metadata in debug mode
            if logging.getLogger().level <= logging.DEBUG:
                print(f"\n📊 Debug Info: {json.dumps(result['metadata'], indent=2)}")
                
        except KeyboardInterrupt:
            print(f"\n\n👗 {agent.config.agent_name}: Goodbye! Stay fashionable! 👋")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            logger.error(f"Interactive mode error: {e}")


def main():
    """Main entry point for CLI usage."""
    parser = create_cli_interface()
    args = parser.parse_args()
    
    # Set up logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize agent
        print("🔄 Initializing StylistAI Agent...")
        config = AgentConfig.from_env()
        agent = StylistAIAgent(config)
        print("✅ Agent initialized successfully!\n")
        
        if args.interactive or not args.query:
            # Interactive mode
            interactive_mode(agent)
        else:
            # Single query mode
            print(f"🤔 Processing query: {args.query}")
            result = agent.query(args.query)
            print(f"\n👗 {config.agent_name}: {result['response']}")
            
            if args.debug:
                print(f"\n📊 Debug Info:")
                print(json.dumps(result['metadata'], indent=2))
    
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        logger.error(f"Main execution failed: {e}")
        return 1
    
    return 0


# Example usage and testing functions
def run_example_queries(agent: StylistAIAgent):
    """Run some example queries for testing."""
    example_queries = [
        "What are the current price trends for white sneakers?",
        "Can you recommend some stylish winter coats under $200?",
        "What's the difference between Nike and Adidas running shoes?",
        "I need a professional outfit for a job interview. Any suggestions?",
        "What colors are trending this season for handbags?"
    ]
    
    print("🧪 Running example queries...\n")
    
    for i, query in enumerate(example_queries, 1):
        print(f"Example {i}: {query}")
        result = agent.query(query, session_id=f"example_{i}")
        print(f"Response: {result['response']}\n")
        print("-" * 80 + "\n")


if __name__ == "__main__":
    exit(main())