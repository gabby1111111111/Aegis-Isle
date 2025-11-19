# AegisIsle - 多智能体协同RAG系统

<div align="center">

![AegisIsle Logo](docs/images/logo.png)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**企业级多智能体协同检索增强生成系统**

[English](README.en.md) | 简体中文 | [在线文档](https://docs.aegisisle.com) | [API文档](https://api.aegisisle.com/docs)

</div>

## 🚀 项目简介

AegisIsle 是一个先进的企业级多智能体协同RAG（检索增强生成）系统，融合了最新的大语言模型技术、智能体编排和检索增强生成能力。系统采用模块化设计，支持多种LLM提供商、向量数据库和多模态处理，为企业提供智能化的知识问答和决策支持解决方案。

### ✨ 核心特性

- 🤖 **多智能体协同**: 专业化智能体分工合作，提高任务处理效率
- 📚 **先进RAG技术**: 智能文档分块、向量检索、混合搜索
- 🔌 **多模型支持**: 兼容 OpenAI、Anthropic、Hugging Face 等多种LLM
- 🗄️ **多种向量数据库**: 支持 Qdrant、ChromaDB、Pinecone、FAISS
- 🖼️ **多模态处理**: 支持文本、图像、表格等多种数据类型
- 🌐 **RESTful API**: 完整的API接口，易于集成
- 🐳 **容器化部署**: Docker支持，一键部署
- 📊 **可观测性**: 完整的日志、监控和指标收集
- 🔒 **企业级安全**: 权限控制、审计日志、数据加密

## 📋 目录

- [快速开始](#-快速开始)
- [系统架构](#-系统架构)
- [功能特性](#-功能特性)
- [安装部署](#-安装部署)
- [使用指南](#-使用指南)
- [API文档](#-api文档)
- [配置说明](#-配置说明)
- [开发指南](#-开发指南)
- [最佳实践](#-最佳实践)
- [故障排查](#-故障排查)
- [贡献指南](#-贡献指南)
- [许可证](#-许可证)

## 🚀 快速开始

### 前置要求

- Python 3.10+
- Docker & Docker Compose (推荐)
- 8GB+ RAM
- OpenAI API Key 或其他LLM服务密钥

### 一键部署

```bash
# 克隆项目
git clone https://github.com/your-org/aegis-isle.git
cd aegis-isle

# 快速设置
./scripts/setup.sh

# 启动开发环境
./scripts/deploy.sh --env development
```

### 本地开发

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 文件，添加你的API密钥

# 启动开发服务器
uvicorn src.aegis_isle.api.main:app --reload
```

访问 http://localhost:8000/docs 查看API文档。

## 🏗️ 系统架构

```mermaid
graph TB
    subgraph "客户端层"
        WEB[Web界面]
        API[REST API]
        SDK[Python SDK]
    end

    subgraph "服务层"
        GATEWAY[API网关]
        AUTH[认证服务]
        RATE[限流服务]
    end

    subgraph "核心层"
        ORCHESTRATOR[智能体编排器]
        RAG[RAG引擎]
        AGENT_ROUTER[智能体路由器]
    end

    subgraph "智能体层"
        RESEARCHER[研究员智能体]
        RETRIEVER[检索智能体]
        SUMMARIZER[总结智能体]
        CHART_GEN[图表生成智能体]
        TOOL_CALLER[工具调用智能体]
    end

    subgraph "数据层"
        VECTOR_DB[向量数据库]
        POSTGRES[PostgreSQL]
        REDIS[Redis缓存]
        FILES[文件存储]
    end

    subgraph "模型层"
        OPENAI[OpenAI GPT]
        ANTHROPIC[Claude]
        HUGGINGFACE[HuggingFace]
        LOCAL[本地模型]
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

## 🎯 功能特性

### 多智能体协同

- **专业分工**: 不同角色的智能体专注特定任务
- **并行处理**: 支持智能体并行执行，提高效率
- **工作流编排**: 复杂任务的自动化流程管理
- **消息路由**: 智能消息分发和任务分配
- **内存管理**: 智能体间的上下文共享和记忆

### RAG引擎

- **多种分块策略**: 递归分块、语义分块、固定大小分块
- **混合检索**: 向量检索 + 关键词检索
- **多模态支持**: 文本、图像、表格的统一处理
- **多格式支持**: PDF、DOCX、HTML、Markdown等
- **智能排序**: 基于相关性和重要性的结果排序

### 文档处理

- **OCR识别**: 图像和扫描文档的文字提取
- **表格解析**: 智能表格内容提取
- **多语言支持**: 中英文混合处理
- **批量处理**: 大规模文档的并行处理
- **增量更新**: 支持文档的增量索引

### 模型集成

- **多提供商支持**: OpenAI、Anthropic、HuggingFace等
- **模型切换**: 动态模型选择和负载均衡
- **本地部署**: 支持私有化部署的本地模型
- **微调支持**: 支持模型的领域适配和微调
- **成本优化**: 智能模型选择，平衡性能和成本

## 🛠️ 安装部署

### Docker部署（推荐）

#### 开发环境

```bash
# 启动开发环境（包含Jupyter、PGAdmin等工具）
docker-compose -f docker-compose.dev.yml up -d

# 查看服务状态
docker-compose -f docker-compose.dev.yml ps
```

#### 生产环境

```bash
# 配置生产环境变量
cp .env.example .env
# 编辑.env文件

# 启动生产环境
docker-compose -f docker-compose.prod.yml up -d

# 查看日志
docker-compose -f docker-compose.prod.yml logs -f aegis-isle
```

### 本地安装

#### 系统依赖

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
# 安装 Python 3.10+
# 下载并安装 Tesseract OCR
# 安装 Visual Studio Build Tools
```

#### Python环境

```bash
# 创建虚拟环境
python3.10 -m venv venv
source venv/bin/activate

# 安装包管理器
pip install --upgrade pip poetry

# 使用Poetry安装依赖
poetry install

# 或使用pip
pip install -r requirements.txt
```

#### 数据库设置

```bash
# 安装PostgreSQL
sudo apt-get install postgresql postgresql-contrib

# 创建数据库
sudo -u postgres createdb aegis_isle
sudo -u postgres createuser -P aegis_user

# 安装Redis
sudo apt-get install redis-server

# 安装并启动Qdrant
docker run -p 6333:6333 qdrant/qdrant:v1.7.0
```

## 📖 使用指南

### 基本使用

#### 上传文档

```python
import requests

# 上传PDF文档
files = {'file': open('document.pdf', 'rb')}
response = requests.post(
    'http://localhost:8000/api/v1/documents/upload',
    files=files
)
print(response.json())
```

#### 文本查询

```python
# 基本查询
query_data = {
    "query": "什么是人工智能？",
    "max_docs": 5
}
response = requests.post(
    'http://localhost:8000/api/v1/query/',
    json=query_data
)
print(response.json())

# 使用智能体进行查询
query_data = {
    "query": "分析最近的市场趋势并生成报告",
    "use_agents": True,
    "agent_workflow": "rag_query"
}
response = requests.post(
    'http://localhost:8000/api/v1/query/',
    json=query_data
)
```

#### 批量查询

```python
batch_data = {
    "queries": [
        "什么是机器学习？",
        "深度学习的应用领域有哪些？",
        "如何选择合适的算法？"
    ],
    "use_agents": True
}
response = requests.post(
    'http://localhost:8000/api/v1/query/batch',
    json=batch_data
)
```

### 高级功能

#### 智能体管理

```python
# 查看所有智能体
response = requests.get('http://localhost:8000/api/v1/agents/')

# 发送消息给智能体
message_data = {
    "message": "请研究最新的AI发展趋势",
    "target_agents": ["researcher_123"]
}
response = requests.post(
    'http://localhost:8000/api/v1/agents/message',
    json=message_data
)
```

#### 工作流执行

```python
# 执行工作流
workflow_data = {
    "workflow_name": "rag_query",
    "input_data": "分析公司财务报表"
}
response = requests.post(
    'http://localhost:8000/api/v1/agents/workflows/execute',
    json=workflow_data
)

# 查看工作流状态
workflow_id = response.json()["workflow_id"]
status = requests.get(
    f'http://localhost:8000/api/v1/agents/workflows/{workflow_id}/status'
)
```

## 🔧 配置说明

### 环境变量

```bash
# 基本配置
ENVIRONMENT=development
DEBUG=True
LOG_LEVEL=INFO

# API配置
API_HOST=0.0.0.0
API_PORT=8000

# 数据库配置
DATABASE_URL=postgresql://user:password@localhost:5432/aegis_isle
REDIS_URL=redis://localhost:6379/0

# 向量数据库
VECTOR_DB_TYPE=qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333

# LLM配置
LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
DEFAULT_LLM_MODEL=gpt-4-1106-preview

# RAG配置
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_RETRIEVED_DOCS=5
SIMILARITY_THRESHOLD=0.7

# 智能体配置
MAX_AGENT_ITERATIONS=10
AGENT_TIMEOUT=300
ENABLE_MEMORY=True

# 多模态支持
ENABLE_MULTIMODAL=True
OCR_ENABLED=True
IMAGE_PROCESSING_ENABLED=True

# 安全配置
SECRET_KEY=your-secret-key
ALLOWED_HOSTS=localhost,127.0.0.1
```

### 高级配置

详细配置说明请参考 [配置文档](docs/configuration.md)。

## 🏃‍♂️ 开发指南

### 项目结构

```
src/aegis_isle/
├── core/           # 核心配置和工具
├── agents/         # 智能体系统
├── rag/           # RAG引擎
├── api/           # API接口
├── models/        # 模型集成
├── embeddings/    # 向量嵌入
├── database/      # 数据库操作
├── utils/         # 工具函数
└── multimodal/    # 多模态处理
```

### 代码规范

```bash
# 代码格式化
black src/ tests/
flake8 src/ tests/

# 类型检查
mypy src/

# 运行测试
pytest tests/ -v

# 测试覆盖率
pytest tests/ --cov=src --cov-report=html
```

### 添加新智能体

```python
from aegis_isle.agents.base import BaseAgent, AgentConfig, AgentRole

class CustomAgent(BaseAgent):
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        # 初始化自定义逻辑

    async def process(self, message):
        # 处理消息的逻辑
        return AgentResponse(
            agent_id=self.id,
            content="处理结果",
            success=True
        )

    async def initialize(self):
        # 初始化资源
        return True

    async def cleanup(self):
        # 清理资源
        return True
```

### 扩展RAG功能

```python
from aegis_isle.rag.chunker import BaseChunker

class CustomChunker(BaseChunker):
    def chunk_document(self, document):
        # 自定义分块逻辑
        chunks = []
        # ...实现分块逻辑
        return chunks
```

## 📊 性能优化

### 系统调优

- **并发设置**: 根据硬件资源调整worker数量
- **缓存策略**: 使用Redis缓存频繁查询结果
- **批处理**: 使用批量处理提高吞吐量
- **模型选择**: 根据场景选择合适的模型大小

### 监控指标

```python
# 查看系统状态
response = requests.get('http://localhost:8000/api/v1/health/detailed')

# 查看性能指标
response = requests.get('http://localhost:8000/api/v1/admin/stats')
```

## 🔍 故障排查

### 常见问题

#### 1. 服务启动失败

```bash
# 检查端口占用
netstat -tulpn | grep 8000

# 检查日志
docker-compose logs aegis-isle

# 检查环境变量
docker-compose config
```

#### 2. 向量数据库连接问题

```bash
# 检查Qdrant状态
curl http://localhost:6333/cluster

# 检查Qdrant日志
docker-compose logs qdrant
```

#### 3. 内存使用过高

```bash
# 检查容器资源使用
docker stats

# 调整内存限制
# 在docker-compose.yml中设置memory limits
```

#### 4. API响应慢

- 检查数据库查询性能
- 优化向量检索参数
- 调整并发设置
- 使用缓存

### 调试模式

```bash
# 启用调试模式
export DEBUG=True
export LOG_LEVEL=DEBUG

# 查看详细日志
tail -f logs/aegis_isle.log
```

## 🤝 贡献指南

我们欢迎所有形式的贡献！

### 如何贡献

1. **Fork** 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 **Pull Request**

### 开发流程

1. 查看 [Issues](https://github.com/your-org/aegis-isle/issues) 或创建新issue
2. 分配给自己并开始开发
3. 确保所有测试通过
4. 更新文档
5. 提交PR

### 代码标准

- 遵循 PEP 8 代码规范
- 添加充分的测试用例
- 编写清晰的文档和注释
- 使用有意义的提交信息

## 📚 相关文档

- [API参考文档](docs/api-reference.md)
- [架构设计文档](docs/architecture.md)
- [部署运维指南](docs/deployment.md)
- [性能调优指南](docs/performance.md)
- [安全配置指南](docs/security.md)
- [故障排查手册](docs/troubleshooting.md)

## 🔗 相关链接

- [项目官网](https://aegisisle.com)
- [在线文档](https://docs.aegisisle.com)
- [API文档](https://api.aegisisle.com/docs)
- [Github仓库](https://github.com/your-org/aegis-isle)
- [Docker Hub](https://hub.docker.com/r/aegisisle/aegis-isle)

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

感谢以下开源项目和贡献者：

- [FastAPI](https://fastapi.tiangolo.com/) - 现代高性能Web框架
- [LangChain](https://github.com/langchain-ai/langchain) - LLM应用开发框架
- [Qdrant](https://qdrant.tech/) - 向量数据库
- [OpenAI](https://openai.com/) - GPT模型API
- [Anthropic](https://www.anthropic.com/) - Claude模型
- [Hugging Face](https://huggingface.co/) - 开源模型和工具

## 📧 联系我们

- 邮箱: contact@aegisisle.com
- 微信群: 扫描下方二维码加入
- QQ群: 123456789
- 论坛: https://forum.aegisisle.com

---

<div align="center">

**如果这个项目对你有帮助，请给我们一个 ⭐**

Made with ❤️ by AegisIsle Team

</div>