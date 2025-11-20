# AegisIsle - 多智能体协作 RAG 系统

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-20.10+-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

AegisIsle 是一个企业级的多智能体协作检索增强生成 (RAG) 系统，具备完整的 OAuth2 + RBAC 权限控制、结构化审计日志和容器化部署能力。

## 🌟 主要特性

### 🤖 多智能体系统
- **协作式 RAG**: 多个专业化智能体协同工作
- **工具集成**: Python REPL、网络搜索、数据可视化
- **LangGraph 工作流**: 状态管理和智能体编排
- **自适应路由**: 智能任务分发和负载均衡

### 🔒 企业级安全
- **OAuth2 认证**: JWT 令牌管理和刷新
- **RBAC 权限控制**: 角色基础访问控制
- **审计日志**: ELK 堆栈兼容的结构化日志
- **API 安全**: 端点级权限保护

### 📚 先进的 RAG 技术
- **混合文档处理**: PDF、Word、图片 OCR 解析
- **多模态嵌入**: 文本、图像统一向量空间
- **智能分块**: 表格感知的语义分割
- **增强检索**: 查询扩展和结果重排

### 🏗️ 生产就绪架构
- **Docker 容器化**: 一键部署和扩展
- **微服务架构**: 松耦合、高可用
- **监控集成**: Prometheus + Grafana
- **负载均衡**: Nginx 反向代理

## 🚀 快速开始

### 先决条件
- Docker 20.10+ & Docker Compose v2.0+
- 8GB+ RAM, 4+ CPU 核心
- 50GB+ 可用磁盘空间

### 一分钟部署

```bash
# 克隆项目
git clone https://github.com/your-org/aegis-isle.git
cd aegis-isle

# 配置环境
cp .env.example .env
# 编辑 .env 文件，设置 API 密钥和密码

# 启动服务
docker-compose up -d

# 验证部署
curl http://localhost:8000/api/v1/health
```

### 获取访问令牌

```bash
curl -X POST "http://localhost:8000/api/v1/auth/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=admin123"
```

## 📖 使用示例

### 1. 文档上传和处理

```python
import requests

# 上传文档
files = {'file': open('document.pdf', 'rb')}
headers = {'Authorization': 'Bearer YOUR_TOKEN'}
response = requests.post(
    'http://localhost:8000/api/v1/documents/upload',
    files=files,
    headers=headers
)
```

### 2. RAG 查询

```python
query_data = {
    "question": "什么是量子计算？",
    "max_docs": 5,
    "use_reranking": True
}

response = requests.post(
    'http://localhost:8000/api/v1/query',
    json=query_data,
    headers={'Authorization': 'Bearer YOUR_TOKEN'}
)

print(response.json()['answer'])
```

### 3. 智能体执行

```python
agent_task = {
    "agent_type": "researcher",
    "task": "分析人工智能的发展趋势",
    "params": {
        "use_web_search": True,
        "generate_chart": True
    }
}

response = requests.post(
    'http://localhost:8000/api/v1/agents/execute',
    json=agent_task,
    headers={'Authorization': 'Bearer YOUR_TOKEN'}
)
```

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                        前端层                                │
├─────────────────────────────────────────────────────────────┤
│ 🌐 Web UI  │ 🔧 Admin Panel │ 📱 API Client │ 📊 Monitoring│
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                      API Gateway                            │
├─────────────────────────────────────────────────────────────┤
│ 🔒 OAuth2/JWT  │ 🛡️ RBAC  │ 📝 Audit Log │ ⚡ Rate Limit│
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                      业务逻辑层                             │
├─────────────────────────────────────────────────────────────┤
│ 🤖 Agent Router │ 📚 RAG Pipeline │ 🔍 Query Engine │ 🛠️ Tools│
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                      数据和存储层                           │
├─────────────────────────────────────────────────────────────┤
│ 🗃️ PostgreSQL │ 🔍 Qdrant │ ⚡ Redis │ 📁 Object Storage│
└─────────────────────────────────────────────────────────────┘
```

## 🔧 核心组件

### RAG 管道
- **文档处理器**: 支持 PDF、DOCX、图片 OCR
- **嵌入器**: OpenAI、Sentence Transformers、CLIP
- **分块器**: 语义分割、表格保留、重叠策略
- **检索器**: 向量搜索、查询扩展、结果重排

### 智能体系统
- **ChartAgent**: 数据可视化和图表生成
- **ResearcherAgent**: 网络搜索和信息聚合
- **CodeAgent**: 代码执行和调试支持
- **OrchestrationAgent**: 多智能体协调管理

### 工具集成
- **PythonREPL**: 安全的代码执行环境
- **WebSearch**: 多搜索引擎聚合（DuckDuckGo、Google、Bing）
- **ChartGenerator**: Plotly 图表生成和导出
- **DocumentParser**: 多格式文档解析

## 📊 监控和日志

### 审计日志格式 (ELK 兼容)

```json
{
  "@timestamp": "2024-01-20T10:30:45.123Z",
  "@version": "1",
  "level": "info",
  "logger": "aegis-isle-audit",
  "service": "aegis-isle",
  "environment": "production",
  "event_type": "authentication",
  "action": "login_success",
  "outcome": "success",
  "username": "admin",
  "ip_address": "192.168.1.100",
  "user_agent": "Mozilla/5.0...",
  "request_id": "req-123456789"
}
```

### 性能指标

| 指标 | 描述 | 目标值 |
|------|------|--------|
| API 响应时间 | 平均响应延迟 | < 500ms |
| 文档处理时间 | PDF/DOCX 解析时间 | < 30s/MB |
| RAG 查询时间 | 检索到回答生成 | < 5s |
| 并发用户数 | 同时在线用户 | 100+ |

## 🔐 安全特性

### 认证和授权
- **多因素认证**: 支持 TOTP、短信验证
- **会话管理**: JWT 令牌轮换和黑名单
- **角色权限**: 细粒度的 RBAC 控制
- **API 限流**: 防止暴力攻击和滥用

### 数据安全
- **传输加密**: TLS 1.3 端到端加密
- **存储加密**: 静态数据 AES-256 加密
- **敏感信息**: 自动脱敏和掩码
- **合规审计**: SOC 2、ISO 27001 标准

## 📈 扩展性

### 水平扩展
- **无状态设计**: 支持多实例负载均衡
- **数据库分片**: PostgreSQL 读写分离
- **缓存策略**: Redis 集群和多级缓存
- **CDN 集成**: 静态资源全球分发

### 垂直扩展
- **GPU 支持**: CUDA 加速的模型推理
- **内存优化**: 大型模型的量化和剪枝
- **存储层**: 对象存储和分布式文件系统
- **网络优化**: HTTP/2、gRPC 协议支持

## 🛠️ 开发指南

### 本地开发环境

```bash
# 克隆项目
git clone https://github.com/your-org/aegis-isle.git
cd aegis-isle

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate

# 安装依赖
pip install -r requirements.txt

# 启动开发服务
uvicorn src.aegis_isle.api.main:app --reload --host 0.0.0.0 --port 8000
```

### 添加新的智能体

```python
from src.aegis_isle.agents.base import BaseAgent

class CustomAgent(BaseAgent):
    def __init__(self, name: str = "custom"):
        super().__init__(name)
        self.description = "自定义智能体描述"

    async def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        # 实现自定义逻辑
        result = await self.execute_custom_logic(task)
        return {
            "result": result,
            "agent": self.name,
            "timestamp": datetime.utcnow().isoformat()
        }
```

### 添加新的工具

```python
from src.aegis_isle.tools.base import BaseTool

class CustomTool(BaseTool):
    name = "custom_tool"
    description = "执行自定义操作的工具"

    async def run(self, tool_input: str) -> ToolResult:
        # 实现工具逻辑
        result = await self.execute_operation(tool_input)

        return ToolResult(
            success=True,
            result=result,
            metadata={"execution_time": time.time()}
        )
```

## 🧪 测试

### 运行测试套件

```bash
# 单元测试
pytest tests/unit/ -v

# 集成测试
pytest tests/integration/ -v

# API 测试
pytest tests/api/ -v

# 性能测试
pytest tests/performance/ -v --benchmark-only

# 覆盖率报告
pytest --cov=src/aegis_isle --cov-report=html
```

### 压力测试

```bash
# 使用 locust 进行负载测试
cd tests/load
locust -f locustfile.py --host=http://localhost:8000

# 并发 RAG 查询测试
python tests/performance/rag_benchmark.py --concurrent=10 --queries=100
```

## 📋 API 文档

完整的 API 文档可在以下地址访问：
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### 主要端点

| 分类 | 端点 | 方法 | 权限 |
|------|------|------|------|
| **认证** |
| | `/api/v1/auth/token` | POST | 公开 |
| | `/api/v1/auth/me` | GET | 用户 |
| | `/api/v1/auth/refresh` | POST | 用户 |
| **文档** |
| | `/api/v1/documents/upload` | POST | 用户 |
| | `/api/v1/documents/list` | GET | 用户 |
| | `/api/v1/documents/{id}` | DELETE | 用户 |
| **查询** |
| | `/api/v1/query` | POST | 用户 |
| | `/api/v1/query/history` | GET | 用户 |
| **智能体** |
| | `/api/v1/agents/execute` | POST | 用户 |
| | `/api/v1/agents/status` | GET | 用户 |
| **管理** |
| | `/api/v1/admin/config` | GET | 管理员 |
| | `/api/v1/admin/stats` | GET | 管理员 |
| | `/api/v1/admin/logs` | GET | 管理员 |

## 🤝 贡献指南

我们欢迎社区贡献！请遵循以下步骤：

### 开发流程

1. **Fork** 项目仓库
2. **创建**特性分支: `git checkout -b feature/amazing-feature`
3. **提交**更改: `git commit -m 'Add amazing feature'`
4. **推送**分支: `git push origin feature/amazing-feature`
5. **创建** Pull Request

### 代码规范

```bash
# 代码格式化
black src/ tests/

# 类型检查
mypy src/

# 代码质量检查
flake8 src/ tests/

# 安全检查
bandit -r src/
```

### 提交信息规范

```
type(scope): description

feat(auth): add OAuth2 refresh token support
fix(rag): resolve document parsing encoding issue
docs(api): update authentication examples
test(agents): add unit tests for chart generation
```

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

特别感谢以下开源项目：

- [FastAPI](https://fastapi.tiangolo.com/) - 现代化的 Python Web 框架
- [LangChain](https://langchain.com/) - 构建 LLM 应用的框架
- [LangGraph](https://langchain-ai.github.io/langgraph/) - 智能体工作流编排
- [Qdrant](https://qdrant.tech/) - 高性能向量数据库
- [Loguru](https://loguru.readthedocs.io/) - 现代化的日志系统

## 📞 支持

- 📧 邮箱: support@aegisisle.dev
- 💬 Discord: [AegisIsle Community](https://discord.gg/aegisisle)
- 📚 文档: [docs.aegisisle.dev](https://docs.aegisisle.dev)
- 🐛 问题报告: [GitHub Issues](https://github.com/your-org/aegis-isle/issues)

---

<div align="center">
  <p>由 ❤️ 和 ☕ 驱动</p>
  <p>© 2024 AegisIsle Team. All rights reserved.</p>
</div>
