# AegisIsle - Multi-Agent Collaborative RAG System

<div align="center">

![AegisIsle Logo](docs/images/logo.png)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Enterprise-grade Multi-Agent Collaborative Retrieval-Augmented Generation System**

English | [简体中文](README.md) | [Documentation](https://docs.aegisisle.com) | [API Docs](https://api.aegisisle.com/docs)

</div>

## 🚀 Introduction

AegisIsle is an advanced enterprise-grade multi-agent collaborative RAG (Retrieval-Augmented Generation) system that integrates cutting-edge large language model technology, intelligent agent orchestration, and retrieval-augmented generation capabilities. The system adopts a modular design, supporting multiple LLM providers, vector databases, and multimodal processing, providing enterprises with intelligent knowledge Q&A and decision support solutions.

### ✨ Key Features

- 🤖 **Multi-Agent Collaboration**: Specialized agents work together to improve task processing efficiency
- 📚 **Advanced RAG Technology**: Intelligent document chunking, vector retrieval, hybrid search
- 🔌 **Multi-Model Support**: Compatible with OpenAI, Anthropic, Hugging Face, and other LLMs
- 🗄️ **Multiple Vector Databases**: Support for Qdrant, ChromaDB, Pinecone, FAISS
- 🖼️ **Multimodal Processing**: Support for text, images, tables, and other data types
- 🌐 **RESTful API**: Complete API interface, easy to integrate
- 🐳 **Containerized Deployment**: Docker support, one-click deployment
- 📊 **Observability**: Complete logging, monitoring, and metrics collection
- 🔒 **Enterprise Security**: Access control, audit logs, data encryption

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker & Docker Compose (recommended)
- 8GB+ RAM
- OpenAI API Key or other LLM service credentials

### One-Click Deployment

```bash
# Clone the project
git clone https://github.com/your-org/aegis-isle.git
cd aegis-isle

# Quick setup
./scripts/setup.sh

# Start development environment
./scripts/deploy.sh --env development
```

### Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env file and add your API keys

# Start development server
uvicorn src.aegis_isle.api.main:app --reload
```

Visit http://localhost:8000/docs to view API documentation.

## 🏗️ System Architecture

```mermaid
graph TB
    subgraph "Client Layer"
        WEB[Web Interface]
        API[REST API]
        SDK[Python SDK]
    end

    subgraph "Service Layer"
        GATEWAY[API Gateway]
        AUTH[Authentication]
        RATE[Rate Limiting]
    end

    subgraph "Core Layer"
        ORCHESTRATOR[Agent Orchestrator]
        RAG[RAG Engine]
        AGENT_ROUTER[Agent Router]
    end

    subgraph "Agent Layer"
        RESEARCHER[Researcher Agent]
        RETRIEVER[Retriever Agent]
        SUMMARIZER[Summarizer Agent]
        CHART_GEN[Chart Generator]
        TOOL_CALLER[Tool Caller]
    end

    subgraph "Data Layer"
        VECTOR_DB[Vector Database]
        POSTGRES[PostgreSQL]
        REDIS[Redis Cache]
        FILES[File Storage]
    end

    subgraph "Model Layer"
        OPENAI[OpenAI GPT]
        ANTHROPIC[Claude]
        HUGGINGFACE[HuggingFace]
        LOCAL[Local Models]
    end

    WEB --> GATEWAY
    API --> GATEWAY
    SDK --> GATEWAY

    GATEWAY --> AUTH
    GATEWAY --> RATE
    GATEWAY --> ORCHESTRATOR
    GATEWAY --> RAG

    ORCHESTRATOR --> AGENT_ROUTER
    AGENT_ROUTER --> RESEARCHER
    AGENT_ROUTER --> RETRIEVER
    AGENT_ROUTER --> SUMMARIZER
    AGENT_ROUTER --> CHART_GEN
    AGENT_ROUTER --> TOOL_CALLER

    RAG --> VECTOR_DB
    RAG --> POSTGRES

    ORCHESTRATOR --> REDIS
    AGENT_ROUTER --> REDIS

    RESEARCHER --> OPENAI
    RETRIEVER --> ANTHROPIC
    SUMMARIZER --> HUGGINGFACE
    CHART_GEN --> LOCAL

    VECTOR_DB --> FILES
    POSTGRES --> FILES
```

## 🎯 Features

### Multi-Agent Collaboration

- **Role Specialization**: Different agents focus on specific tasks
- **Parallel Processing**: Support for parallel agent execution
- **Workflow Orchestration**: Automated process management for complex tasks
- **Message Routing**: Intelligent message distribution and task allocation
- **Memory Management**: Context sharing and memory between agents

### RAG Engine

- **Multiple Chunking Strategies**: Recursive, semantic, fixed-size chunking
- **Hybrid Retrieval**: Vector retrieval + keyword retrieval
- **Multimodal Support**: Unified processing of text, images, tables
- **Multiple Format Support**: PDF, DOCX, HTML, Markdown, etc.
- **Intelligent Ranking**: Result ranking based on relevance and importance

### Document Processing

- **OCR Recognition**: Text extraction from images and scanned documents
- **Table Parsing**: Intelligent table content extraction
- **Multi-language Support**: Chinese and English mixed processing
- **Batch Processing**: Parallel processing of large-scale documents
- **Incremental Updates**: Support for incremental document indexing

### Model Integration

- **Multi-Provider Support**: OpenAI, Anthropic, HuggingFace, etc.
- **Model Switching**: Dynamic model selection and load balancing
- **Local Deployment**: Support for privately deployed local models
- **Fine-tuning Support**: Support for domain adaptation and fine-tuning
- **Cost Optimization**: Intelligent model selection balancing performance and cost

## 🛠️ Installation & Deployment

### Docker Deployment (Recommended)

#### Development Environment

```bash
# Start development environment (includes Jupyter, PGAdmin, etc.)
docker-compose -f docker-compose.dev.yml up -d

# Check service status
docker-compose -f docker-compose.dev.yml ps
```

#### Production Environment

```bash
# Configure production environment variables
cp .env.example .env
# Edit .env file

# Start production environment
docker-compose -f docker-compose.prod.yml up -d

# View logs
docker-compose -f docker-compose.prod.yml logs -f aegis-isle
```

### Local Installation

#### System Dependencies

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y \
    python3.10 python3.10-venv python3-pip \
    tesseract-ocr tesseract-ocr-eng tesseract-ocr-chi-sim \
    poppler-utils \
    build-essential

# macOS
brew install python@3.10 tesseract tesseract-lang poppler

# Windows
# Install Python 3.10+
# Download and install Tesseract OCR
# Install Visual Studio Build Tools
```

#### Python Environment

```bash
# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install package manager
pip install --upgrade pip poetry

# Install dependencies with Poetry
poetry install

# Or use pip
pip install -r requirements.txt
```

## 📖 Usage Guide

### Basic Usage

#### Upload Documents

```python
import requests

# Upload PDF document
files = {'file': open('document.pdf', 'rb')}
response = requests.post(
    'http://localhost:8000/api/v1/documents/upload',
    files=files
)
print(response.json())
```

#### Text Queries

```python
# Basic query
query_data = {
    "query": "What is artificial intelligence?",
    "max_docs": 5
}
response = requests.post(
    'http://localhost:8000/api/v1/query/',
    json=query_data
)
print(response.json())

# Query using agents
query_data = {
    "query": "Analyze recent market trends and generate a report",
    "use_agents": True,
    "agent_workflow": "rag_query"
}
response = requests.post(
    'http://localhost:8000/api/v1/query/',
    json=query_data
)
```

#### Batch Queries

```python
batch_data = {
    "queries": [
        "What is machine learning?",
        "What are the applications of deep learning?",
        "How to choose the right algorithm?"
    ],
    "use_agents": True
}
response = requests.post(
    'http://localhost:8000/api/v1/query/batch',
    json=batch_data
)
```

### Advanced Features

#### Agent Management

```python
# View all agents
response = requests.get('http://localhost:8000/api/v1/agents/')

# Send message to agents
message_data = {
    "message": "Please research the latest AI development trends",
    "target_agents": ["researcher_123"]
}
response = requests.post(
    'http://localhost:8000/api/v1/agents/message',
    json=message_data
)
```

#### Workflow Execution

```python
# Execute workflow
workflow_data = {
    "workflow_name": "rag_query",
    "input_data": "Analyze company financial statements"
}
response = requests.post(
    'http://localhost:8000/api/v1/agents/workflows/execute',
    json=workflow_data
)

# Check workflow status
workflow_id = response.json()["workflow_id"]
status = requests.get(
    f'http://localhost:8000/api/v1/agents/workflows/{workflow_id}/status'
)
```

## 🔧 Configuration

### Environment Variables

```bash
# Basic configuration
ENVIRONMENT=development
DEBUG=True
LOG_LEVEL=INFO

# API configuration
API_HOST=0.0.0.0
API_PORT=8000

# Database configuration
DATABASE_URL=postgresql://user:password@localhost:5432/aegis_isle
REDIS_URL=redis://localhost:6379/0

# Vector database
VECTOR_DB_TYPE=qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333

# LLM configuration
LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
DEFAULT_LLM_MODEL=gpt-4-1106-preview

# RAG configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_RETRIEVED_DOCS=5
SIMILARITY_THRESHOLD=0.7

# Agent configuration
MAX_AGENT_ITERATIONS=10
AGENT_TIMEOUT=300
ENABLE_MEMORY=True

# Multimodal support
ENABLE_MULTIMODAL=True
OCR_ENABLED=True
IMAGE_PROCESSING_ENABLED=True

# Security configuration
SECRET_KEY=your-secret-key
ALLOWED_HOSTS=localhost,127.0.0.1
```

## 🏃‍♂️ Development Guide

### Project Structure

```
src/aegis_isle/
├── core/           # Core configuration and utilities
├── agents/         # Agent system
├── rag/           # RAG engine
├── api/           # API interfaces
├── models/        # Model integration
├── embeddings/    # Vector embeddings
├── database/      # Database operations
├── utils/         # Utility functions
└── multimodal/    # Multimodal processing
```

### Code Standards

```bash
# Code formatting
black src/ tests/
flake8 src/ tests/

# Type checking
mypy src/

# Run tests
pytest tests/ -v

# Test coverage
pytest tests/ --cov=src --cov-report=html
```

### Adding New Agents

```python
from aegis_isle.agents.base import BaseAgent, AgentConfig, AgentRole

class CustomAgent(BaseAgent):
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        # Initialize custom logic

    async def process(self, message):
        # Message processing logic
        return AgentResponse(
            agent_id=self.id,
            content="Processing result",
            success=True
        )

    async def initialize(self):
        # Initialize resources
        return True

    async def cleanup(self):
        # Cleanup resources
        return True
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Service Startup Failure

```bash
# Check port usage
netstat -tulpn | grep 8000

# Check logs
docker-compose logs aegis-isle

# Check environment variables
docker-compose config
```

#### 2. Vector Database Connection Issues

```bash
# Check Qdrant status
curl http://localhost:6333/cluster

# Check Qdrant logs
docker-compose logs qdrant
```

#### 3. High Memory Usage

```bash
# Check container resource usage
docker stats

# Adjust memory limits
# Set memory limits in docker-compose.yml
```

## 🤝 Contributing

We welcome all forms of contributions!

### How to Contribute

1. **Fork** the project
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Create a **Pull Request**

### Development Workflow

1. Check [Issues](https://github.com/your-org/aegis-isle/issues) or create a new issue
2. Assign to yourself and start development
3. Ensure all tests pass
4. Update documentation
5. Submit PR

### Code Standards

- Follow PEP 8 coding standards
- Add comprehensive test cases
- Write clear documentation and comments
- Use meaningful commit messages

## 📚 Related Documentation

- [API Reference](docs/api-reference.md)
- [Architecture Design](docs/architecture.md)
- [Deployment Guide](docs/deployment.md)
- [Performance Tuning](docs/performance.md)
- [Security Configuration](docs/security.md)
- [Troubleshooting Manual](docs/troubleshooting.md)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Thanks to the following open source projects and contributors:

- [FastAPI](https://fastapi.tiangolo.com/) - Modern, fast web framework
- [LangChain](https://github.com/langchain-ai/langchain) - LLM application development framework
- [Qdrant](https://qdrant.tech/) - Vector database
- [OpenAI](https://openai.com/) - GPT model API
- [Anthropic](https://www.anthropic.com/) - Claude model
- [Hugging Face](https://huggingface.co/) - Open source models and tools

## 📧 Contact Us

- Email: contact@aegisisle.com
- GitHub: https://github.com/your-org/aegis-isle
- Forum: https://forum.aegisisle.com

---

<div align="center">

**If this project helps you, please give us a ⭐**

Made with ❤️ by AegisIsle Team

</div>