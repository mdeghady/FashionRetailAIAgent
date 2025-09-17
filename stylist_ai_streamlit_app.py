"""
StylistAI Streamlit Web Application
==================================

A modern chat interface for the StylistAI fashion assistant agent.
Provides a ChatGPT-like UI with markdown support, session management,
and seamless integration with the LangChain + LangGraph RAG system.
"""

import streamlit as st
import time
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
import json
import traceback

# Import the existing agent
try:
    from stylist_ai_agent import StylistAIAgent, AgentConfig
    AGENT_AVAILABLE = True
except ImportError as e:
    st.error(f"⚠️ Could not import StylistAI Agent: {e}")
    st.error("Please ensure stylist_ai_agent.py is in the same directory or properly installed.")
    AGENT_AVAILABLE = False

# =============================================================================
# Configuration and Styling
# =============================================================================

# Page configuration
st.set_page_config(
    page_title="StylistAI - Fashion Assistant",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for chat interface
def inject_custom_css():
    """Inject custom CSS for better chat UI styling."""
    st.markdown("""
    <style>
    /* Main chat container */
    .stApp {
        background-color: #fafafa;
    }
    
    /* Custom chat bubbles */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 12px 16px;
        border-radius: 18px 18px 5px 18px;
        margin: 8px 0;
        margin-left: 20%;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        font-size: 14px;
        line-height: 1.4;
    }
    
    .assistant-message {
        background: white;
        color: #333;
        padding: 12px 16px;
        border-radius: 18px 18px 18px 5px;
        margin: 8px 0;
        margin-right: 20%;
        border-left: 4px solid #ff6b6b;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        font-size: 14px;
        line-height: 1.4;
    }
    
    /* System message styling */
    .system-message {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        color: #2c3e50;
        padding: 12px 16px;
        border-radius: 12px;
        margin: 8px 0;
        text-align: center;
        font-style: italic;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    }
    
    /* Loading message */
    .loading-message {
        background: #f8f9fa;
        color: #6c757d;
        padding: 12px 16px;
        border-radius: 18px 18px 18px 5px;
        margin: 8px 0;
        margin-right: 20%;
        border-left: 4px solid #6c757d;
        font-style: italic;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.6; }
        100% { opacity: 1; }
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    
    /* Chat input styling */
    .stTextInput > div > div > input {
        background-color: white;
        border-radius: 20px;
        border: 2px solid #e1e5e9;
        padding: 10px 15px;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.25);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Custom markdown in chat bubbles */
    .user-message h1, .user-message h2, .user-message h3 {
        color: white;
        margin-top: 0;
    }
    
    .assistant-message h1, .assistant-message h2, .assistant-message h3 {
        color: #2c3e50;
        margin-top: 0;
    }
    
    .assistant-message code {
        background: #f4f4f4;
        padding: 2px 4px;
        border-radius: 3px;
        font-family: 'Courier New', monospace;
    }
    
    .assistant-message pre {
        background: #f8f9fa;
        padding: 10px;
        border-radius: 5px;
        overflow-x: auto;
    }
    
    /* Metadata styling */
    .message-metadata {
        font-size: 11px;
        color: #888;
        text-align: right;
        margin-top: 4px;
    }
    </style>
    """, unsafe_allow_html=True)

# =============================================================================
# Message Management Classes
# =============================================================================

class ChatMessage:
    """Represents a single chat message with metadata."""
    
    def __init__(self, role: str, content: str, timestamp: Optional[datetime] = None, 
                 metadata: Optional[Dict[str, Any]] = None, message_id: Optional[str] = None):
        self.role = role  # 'user', 'assistant', 'system'
        self.content = content
        self.timestamp = timestamp or datetime.now()
        self.metadata = metadata or {}
        self.message_id = message_id or str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'role': self.role,
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata,
            'message_id': self.message_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatMessage':
        """Create from dictionary."""
        return cls(
            role=data['role'],
            content=data['content'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            metadata=data.get('metadata', {}),
            message_id=data.get('message_id', str(uuid.uuid4()))
        )

class ChatSession:
    """Manages a chat session with message history."""
    
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.messages: List[ChatMessage] = []
        self.created_at = datetime.now()
    
    def add_message(self, message: ChatMessage):
        """Add a message to the session."""
        self.messages.append(message)
    
    def get_messages(self) -> List[ChatMessage]:
        """Get all messages in chronological order."""
        return self.messages
    
    def clear_messages(self):
        """Clear all messages."""
        self.messages = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'session_id': self.session_id,
            'messages': [msg.to_dict() for msg in self.messages],
            'created_at': self.created_at.isoformat()
        }

# =============================================================================
# Agent Interface Wrapper
# =============================================================================

class StreamlitAgentInterface:
    """Wrapper for the StylistAI agent with Streamlit integration."""
    
    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the agent interface."""
        self.config = config or AgentConfig.from_env()
        self.agent = None
        self.last_error = None
        self._initialize_agent()
    
    def _initialize_agent(self) -> bool:
        """Initialize the StylistAI agent."""
        try:
            if not AGENT_AVAILABLE:
                raise ImportError("StylistAI agent not available")
            
            self.agent = StylistAIAgent(self.config)
            self.last_error = None
            return True
            
        except Exception as e:
            self.last_error = str(e)
            st.error(f"Failed to initialize agent: {e}")
            return False
    
    def is_ready(self) -> bool:
        """Check if agent is ready to process requests."""
        return self.agent is not None and self.last_error is None
    
    def chat(self, user_input: str, session_id: str) -> Dict[str, Any]:
        """
        Send a message to the agent and get response.
        
        Returns:
            Dict with keys: 'response', 'success', 'metadata', 'error'
        """
        if not self.is_ready():
            return {
                'response': f"I apologize, but I'm having trouble starting up. Error: {self.last_error}",
                'success': False,
                'metadata': {},
                'error': self.last_error
            }
        
        try:
            # Use the existing agent's query method
            result = self.agent.query(user_input, session_id=session_id)
            return result
            
        except Exception as e:
            error_msg = f"I encountered an error while processing your request: {str(e)}"
            return {
                'response': error_msg,
                'success': False,
                'metadata': {'error_details': traceback.format_exc()},
                'error': str(e)
            }
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about the current agent configuration."""
        if not self.is_ready():
            return {'status': 'not_ready', 'error': self.last_error}
        
        return {
            'status': 'ready',
            'agent_name': self.config.agent_name,
            'model': self.config.gemini_model,
            'collection': self.config.chroma_collection_name,
            'retrieval_k': self.config.retrieval_k
        }

# =============================================================================
# Streamlit UI Components
# =============================================================================

def render_message(message: ChatMessage, show_metadata: bool = False):
    """Render a single chat message with appropriate styling."""
    
    # Format timestamp
    time_str = message.timestamp.strftime("%H:%M")
    
    if message.role == "user":
        # User message bubble
        st.markdown(f"""
        <div class="user-message">
            {message.content}
            {f'<div class="message-metadata">{time_str}</div>' if show_metadata else ''}
        </div>
        """, unsafe_allow_html=True)
        
    elif message.role == "assistant":
        # Assistant message bubble with markdown rendering
        st.markdown(f"""
        <div class="assistant-message">
        """, unsafe_allow_html=True)
        
        # Render the markdown content
        st.markdown(message.content)
        
        if show_metadata:
            metadata_info = []
            if 'retrieved_count' in message.metadata:
                metadata_info.append(f"📚 {message.metadata['retrieved_count']} sources")
            if 'response_time' in message.metadata:
                metadata_info.append(f"⏱️ {message.metadata['response_time']:.1f}s")
            
            if metadata_info:
                st.markdown(f"""
                <div class="message-metadata">
                    {' • '.join(metadata_info)} • {time_str}
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
    elif message.role == "system":
        # System message
        st.markdown(f"""
        <div class="system-message">
            {message.content}
        </div>
        """, unsafe_allow_html=True)

def render_loading_message():
    """Render a loading message while agent is thinking."""
    st.markdown("""
    <div class="loading-message">
        🤔 StylistAI is thinking...
    </div>
    """, unsafe_allow_html=True)

def initialize_session():
    """Initialize the Streamlit session state."""
    
    # Initialize chat session
    if 'chat_session' not in st.session_state:
        st.session_state.chat_session = ChatSession()
        
        # Add welcome message
        welcome_msg = ChatMessage(
            role="system",
            content="👗 Welcome to StylistAI! I'm your personal fashion assistant. Ask me about trends, styling tips, product recommendations, or anything fashion-related!"
        )
        st.session_state.chat_session.add_message(welcome_msg)
    
    # Initialize agent interface
    if 'agent_interface' not in st.session_state:
        st.session_state.agent_interface = StreamlitAgentInterface()
    
    # Initialize UI state
    if 'show_metadata' not in st.session_state:
        st.session_state.show_metadata = False
    
    if 'processing' not in st.session_state:
        st.session_state.processing = False

def render_sidebar():
    """Render the sidebar with controls and information."""
    
    st.sidebar.title("👗 StylistAI")
    st.sidebar.markdown("*Your AI Fashion Assistant*")
    
    # Agent status
    agent_info = st.session_state.agent_interface.get_agent_info()
    
    if agent_info['status'] == 'ready':
        st.sidebar.success("🟢 Agent Ready")
        st.sidebar.info(f"**Model**: {agent_info['model']}")
        st.sidebar.info(f"**Collection**: {agent_info['collection']}")
    else:
        st.sidebar.error("🔴 Agent Error")
        st.sidebar.error(agent_info.get('error', 'Unknown error'))
    
    st.sidebar.markdown("---")
    
    # Chat controls
    st.sidebar.subheader("💬 Chat Controls")
    
    # Clear chat button
    if st.sidebar.button("🗑️ Clear Chat", help="Clear all messages"):
        st.session_state.chat_session.clear_messages()
        # Re-add welcome message
        welcome_msg = ChatMessage(
            role="system",
            content="👗 Chat cleared! How can I help you with fashion today?"
        )
        st.session_state.chat_session.add_message(welcome_msg)
        st.rerun()
    
    # Show metadata toggle
    st.session_state.show_metadata = st.sidebar.checkbox(
        "🔍 Show Message Metadata", 
        value=st.session_state.show_metadata,
        help="Show retrieval info and response times"
    )
    
    # Download chat history
    if st.sidebar.button("💾 Download Chat"):
        chat_data = st.session_state.chat_session.to_dict()
        st.sidebar.download_button(
            label="📁 Download JSON",
            data=json.dumps(chat_data, indent=2),
            file_name=f"stylist_ai_chat_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json"
        )
    
    st.sidebar.markdown("---")
    
    # Chat statistics
    st.sidebar.subheader("📊 Chat Stats")
    messages = st.session_state.chat_session.get_messages()
    user_messages = [m for m in messages if m.role == "user"]
    assistant_messages = [m for m in messages if m.role == "assistant"]
    
    st.sidebar.metric("Total Messages", len(messages))
    st.sidebar.metric("Your Messages", len(user_messages))
    st.sidebar.metric("AI Responses", len(assistant_messages))
    
    # Usage examples
    st.sidebar.markdown("---")
    st.sidebar.subheader("💡 Try These Examples")
    
    example_prompts = [
        "What are the latest sneaker trends?",
        "Help me choose a professional outfit",
        "What colors go well with navy blue?",
        "Sustainable fashion brands under $100",
        "Summer dress recommendations for petite women"
    ]
    
    for prompt in example_prompts:
        if st.sidebar.button(f"💬 {prompt[:30]}...", key=f"example_{hash(prompt)}"):
            # Add the example as a user message and process it
            user_msg = ChatMessage(role="user", content=prompt)
            st.session_state.chat_session.add_message(user_msg)
            st.session_state.processing = True
            st.rerun()

def process_user_input(user_input: str):
    """Process user input and get agent response."""
    
    # Add user message
    user_msg = ChatMessage(role="user", content=user_input)
    st.session_state.chat_session.add_message(user_msg)
    
    # Get agent response
    start_time = time.time()
    response_data = st.session_state.agent_interface.chat(
        user_input, 
        session_id=st.session_state.chat_session.session_id
    )
    response_time = time.time() - start_time
    
    # Add response metadata
    response_data['metadata']['response_time'] = response_time
    
    # Create assistant message
    assistant_msg = ChatMessage(
        role="assistant",
        content=response_data['response'],
        metadata=response_data['metadata']
    )
    
    st.session_state.chat_session.add_message(assistant_msg)
    
    # Reset processing state
    st.session_state.processing = False

def main():
    """Main Streamlit application."""
    
    # Inject custom CSS
    inject_custom_css()
    
    # Initialize session
    initialize_session()
    
    # Render sidebar
    render_sidebar()
    
    # Main chat interface
    st.title("👗 StylistAI Fashion Assistant")
    
    # Check if agent is ready
    if not st.session_state.agent_interface.is_ready():
        st.error("⚠️ The fashion assistant is not ready. Please check the configuration and try again.")
        st.info("Make sure you have:")
        st.markdown("""
        - Set your `GOOGLE_API_KEY` environment variable
        - Your ChromaDB collection `fashion_products` is accessible
        - All required dependencies are installed
        """)
        return
    
    # Chat container
    chat_container = st.container()
    
    # Display chat messages
    with chat_container:
        messages = st.session_state.chat_session.get_messages()
        
        for message in messages:
            render_message(message, show_metadata=st.session_state.show_metadata)
        
        # Show loading message if processing
        if st.session_state.processing:
            render_loading_message()
    
    # Chat input
    st.markdown("---")
    
    # Create columns for input layout
    input_col, button_col = st.columns([4, 1])
    
    with input_col:
        user_input = st.text_input(
            "Ask me anything about fashion...",
            placeholder="e.g., What are the best winter coats for under $200?",
            disabled=st.session_state.processing,
            key="user_input"
        )
    
    with button_col:
        send_button = st.button(
            "Send 🚀", 
            disabled=st.session_state.processing or not user_input.strip(),
            type="primary"
        )
    
    # Handle input submission
    if send_button and user_input.strip():
        st.session_state.processing = True
        st.rerun()
    
    # Process input if in processing state
    if st.session_state.processing and user_input.strip():
        process_user_input(user_input)
        st.rerun()
    
    # Keyboard shortcut info
    st.caption("💡 Tip: Press Ctrl+Enter to send your message quickly!")

# =============================================================================
# Error Handling and Fallbacks
# =============================================================================

def render_error_page():
    """Render an error page when the agent is not available."""
    
    st.title("⚠️ StylistAI - Setup Required")
    
    st.error("The StylistAI agent could not be initialized.")
    
    st.markdown("""
    ## 🔧 Setup Instructions
    
    To use this application, you need to:
    
    ### 1. Install Dependencies
    ```bash
    pip install -r requirements.txt
    ```
    
    ### 2. Set Environment Variables
    Create a `.env` file with:
    ```
    GOOGLE_API_KEY=your_google_api_key_here
    CHROMA_COLLECTION=fashion_products
    ```
    
    ### 3. Ensure ChromaDB Collection Exists
    Make sure you have a ChromaDB collection named `fashion_products` with your fashion data.
    
    ### 4. Place Agent Code
    Ensure `stylist_ai_agent.py` is in the same directory as this Streamlit app.
    
    ## 🚀 Running the App
    ```bash
    streamlit run stylist_ai_streamlit.py
    ```
    """)
    
    st.markdown("---")
    st.info("Once you've completed the setup, refresh this page to start chatting with StylistAI!")

# =============================================================================
# Application Entry Point
# =============================================================================

if __name__ == "__main__":
    try:
        if AGENT_AVAILABLE:
            main()
        else:
            render_error_page()
    except Exception as e:
        st.error(f"Application Error: {e}")
        st.code(traceback.format_exc())
        st.info("Please check the console for detailed error information.")
