# 👗 StylistAI - AI-Powered Fashion Assistant

> A production-ready Retrieval-Augmented Generation (RAG) system that provides personalized fashion recommendations through natural language conversation.


## 🎯 Overview

StylistAI transforms e-commerce product discovery by enabling customers to find products through natural conversation instead of keyword search. Ask questions like *"I need a professional outfit for a summer wedding under $200"* and receive personalized, context-aware recommendations.

### Key Features

- 🤖 **Natural Language Understanding** - Semantic search that understands intent, not just keywords
- 💬 **Conversational Interface** - ChatGPT-style web UI with conversation history
- 🎨 **Personalized Recommendations** - Context-aware styling advice with real product data
- ⚡ **Fast Response Times** - 2-3 second average query response
- 📊 **Scalable Architecture** - Handles 10,000+ products with room to scale
- 🔍 **Transparent Retrieval** - See which products influenced each recommendation
- 🛠️ **Production-Ready** - Comprehensive error handling, logging, and monitoring

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  User Interface Layer                    │
│              (Streamlit Chat Interface)                  │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│              Agent Orchestration Layer                   │
│                  (LangGraph Workflow)                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  ChromaDB    │→ │    Prompt    │→ │    Gemini    │ │
│  │  Retriever   │  │   Builder    │  │     LLM      │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│               Data Ingestion Layer                       │
│    (Preprocessing → Embedding → Vector Storage)          │
└──────────────────────────────────────────────────────────┘
```

### Tech Stack

- **Vector Database**: ChromaDB with persistent storage
- **Embeddings**: SentenceTransformers (multi-qa-MiniLM-L6-cos-v1)
- **LLM**: Google Gemini Pro via LangChain
- **Orchestration**: LangGraph for workflow management
- **Web Interface**: Streamlit with custom styling
- **Data Processing**: Pandas, NumPy

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Google API Key (for Gemini)
- 2GB+ RAM recommended

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/mdeghady/FashionRetailAIAgent.git
cd FashionRetailAIAgent
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**

Create a `.env` file in the project root:

```env
# Required
GOOGLE_API_KEY=your_google_api_key_here

# Optional (defaults provided)
CHROMA_COLLECTION=fashion_products
CHROMA_PERSIST_DIR=./chroma_db
RETRIEVAL_K=4
SIMILARITY_THRESHOLD=0.7
MAX_CONTEXT_LENGTH=3000
GEMINI_MODEL=gemini-pro
GEMINI_TEMPERATURE=0.7
GEMINI_MAX_TOKENS=1024
```

### Data Preparation

5. **Prepare your product data**

Your data should be in Parquet format with these columns:
- `Brand`, `ProductName`, `Description`
- `Category`, `Department`
- `CurrentPrice`, `OriginalPrice`, `PriceCurrency`
- `ProductColor`, `AvailableSizes`
- `StockAvailability`, `ProductURL`
- `sku`, `website`, `date`

6. **Run the ingestion pipeline**

```bash
python fashion_rag_system.py
```

This will:
- Preprocess and clean your product data
- Generate embeddings for all products
- Create and populate the ChromaDB vector database
- Save to `./chroma_db` directory

Expected output:
```
Loaded dataset with shape: (10000, 15)
Step 1: Preprocessing data...
Step 2: Creating product documents...
Step 3: Creating metadata...
Step 4: Generating embeddings...
Step 5: Setting up vector database...
Step 6: Inserting data...
✅ Fashion RAG system setup complete!
```

### Running the Application

7. **Launch the web interface**

```bash
streamlit run stylist_ai_streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

8. **Or use the CLI interface**

```bash
# Interactive mode
python stylist_ai_agent.py --interactive

# Single query mode
python stylist_ai_agent.py --query "Show me comfortable running shoes under $150"

# Debug mode
python stylist_ai_agent.py --interactive --debug
```

## 📖 Usage Examples

### Web Interface

```
User: I need a dress for a summer wedding

StylistAI: I'd recommend looking at floral midi dresses or elegant maxi 
dresses for a summer wedding! Based on your inventory, I found some 
beautiful options:

1. **Zara Floral Midi Dress** - $89.99
   - Perfect floral pattern in soft pastels
   - Breathable cotton blend
   - Available in sizes S-XL

2. **H&M Elegant Maxi Dress** - $79.99
   - Flowing silhouette ideal for outdoor weddings
   - Navy blue with subtle print
   - Sizes XS-L in stock

Would you like me to suggest accessories to complete the look?
```

### Python API

```python
from stylist_ai_agent import StylistAIAgent, AgentConfig

# Initialize agent
config = AgentConfig.from_env()
agent = StylistAIAgent(config)

# Query the agent
result = agent.query(
    "What are trending sneaker colors this season?",
    session_id="user_123"
)

print(result['response'])
print(f"Retrieved {result['metadata']['retrieved_count']} products")
```

### Custom Configuration

```python
from stylist_ai_agent import AgentConfig

config = AgentConfig(
    retrieval_k=5,              # Retrieve more products
    similarity_threshold=0.6,    # Lower threshold for more results
    gemini_temperature=0.8,      # More creative responses
    max_context_length=4000      # Longer context window
)

agent = StylistAIAgent(config)
```


## 🔧 Configuration Options

### Retrieval Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `retrieval_k` | 4 | Number of products to retrieve |
| `similarity_threshold` | 0.7 | Minimum similarity score (0-1) |
| `max_context_length` | 3000 | Maximum context characters for LLM |

### LLM Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gemini_model` | gemini-pro | Gemini model version |
| `gemini_temperature` | 0.7 | Response creativity (0-1) |
| `gemini_max_tokens` | 1024 | Maximum response length |
| `max_retries` | 3 | API retry attempts |

### ChromaDB Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chroma_collection_name` | fashion_products | Collection name |
| `chroma_persist_directory` | ./chroma_db | Storage location |


## 🐛 Troubleshooting

### Common Issues

**Issue**: `Failed to initialize ChromaDB client`
```
Solution: Ensure chroma_db directory exists and has write permissions
chmod 755 ./chroma_db
```

**Issue**: `GOOGLE_API_KEY not found`
```
Solution: Set environment variable or add to .env file
export GOOGLE_API_KEY=your_key_here
```

**Issue**: `No relevant fashion data found`
```
Solution: Run ingestion pipeline first
python fashion_rag_system.py
```

**Issue**: Slow response times
```
Solution: Reduce retrieval_k or max_context_length
config = AgentConfig(retrieval_k=3, max_context_length=2000)
```

### Debug Mode

Enable detailed logging:
```bash
python stylist_ai_agent.py --interactive --debug
```

Or in code:
```python
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

## 📊 Performance Benchmarks

Tested on MacBook Pro M1, 16GB RAM:

| Metric | Value |
|--------|-------|
| Average Response Time | 2.3 seconds |
| Embedding Generation (10K products) | ~45 seconds |
| Vector Search Latency | ~50ms |
| Memory Usage | ~500MB |
| Max Products Tested | 50,000 |

## 🔒 Security & Privacy

- **API Keys**: Never commit `.env` file - use environment variables
- **User Data**: Conversations stored in memory only (not persisted)
- **Product Data**: Ensure compliance with data usage rights
- **Rate Limiting**: Gemini API has rate limits - implement queuing for high traffic

## 🚢 Deployment

### Option 1: Streamlit Cloud (Easiest)

1. Push code to GitHub
2. Connect to [Streamlit Cloud](https://streamlit.io/cloud)
3. Add `GOOGLE_API_KEY` as a secret
4. Deploy!

### Option 2: Docker

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "stylist_ai_streamlit_app.py"]
```

```bash
docker build -t stylist-ai .
docker run -p 8501:8501 -e GOOGLE_API_KEY=your_key stylist-ai
```

### Option 3: Production API

Use FastAPI wrapper (see `stylist_ai_usage_example.py`):

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```


## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Add docstrings to all functions
- Include unit tests for new features
- Update README with new functionality



## 🙏 Acknowledgments

- **LangChain** for the excellent LLM orchestration framework
- **ChromaDB** for the performant vector database
- **Sentence Transformers** for quality embeddings
- **Streamlit** for the rapid UI development



## 🌟 Star History

If you find this project helpful, please consider giving it a star! ⭐

