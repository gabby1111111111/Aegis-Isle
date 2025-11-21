# AegisIsle 本地开发环境指南

## 🚀 快速开始

### 前置要求

- **Python 3.9+**
- **Git**
- **至少 4GB 内存** (用于AI模型)
- **至少 10GB 磁盘空间** (用于依赖和模型)

### 1. 克隆项目

```bash
git clone https://github.com/your-org/aegis-isle.git
cd aegis-isle
```

### 2. 设置开发环境

#### Windows
```cmd
setup_dev_env.bat
```

#### Linux/Mac
```bash
chmod +x setup_dev_env.sh
./setup_dev_env.sh
```

### 3. 激活虚拟环境

#### Windows
```cmd
venv\Scripts\activate
```

#### Linux/Mac
```bash
source venv/bin/activate
```

### 4. 启动开发服务

#### 方式一：使用开发脚本（推荐）
```bash
# 启动简化认证服务器（推荐，适合开发调试）
python run_dev.py --mode auth

# 启动完整服务器（包含RAG、Agent等功能）
python run_dev.py --mode full

# 自定义端口
python run_dev.py --mode auth --port 8080
```

#### 方式二：直接使用uvicorn
```bash
# 简化版本
uvicorn auth_server_simple:app --reload --host 0.0.0.0 --port 8000

# 完整版本（需要解决依赖问题）
uvicorn src.aegis_isle.api.main:app --reload --host 0.0.0.0 --port 8000
```

## 📱 访问服务

- **API文档 (Swagger)**: http://localhost:8000/docs
- **API文档 (ReDoc)**: http://localhost:8000/redoc
- **根端点**: http://localhost:8000/
- **健康检查**: http://localhost:8000/api/v1/health

## 👥 默认账户

| 用户名 | 密码 | 角色 | 权限 |
|--------|------|------|------|
| admin | admin123 | super_admin | 所有权限 |
| testuser | testpass123 | user | 基础权限 |

## 🔧 开发模式说明

### 简化认证模式 (`--mode auth`)

- **适用场景**: 前端开发、认证测试、API调试
- **包含功能**: OAuth2、JWT、RBAC、审计日志
- **优点**: 启动快、依赖少、稳定
- **推荐**: 用于日常开发

### 完整功能模式 (`--mode full`)

- **适用场景**: 完整功能测试、AI功能开发
- **包含功能**: 所有模块（RAG、Agent、Tools等）
- **注意**: 需要解决嵌入模型等依赖问题
- **首次启动**: 可能需要下载AI模型

## 🧪 API测试示例

### 1. 登录获取Token

```bash
curl -X POST "http://localhost:8000/api/v1/auth/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=admin123"
```

### 2. 访问受保护资源

```bash
# 使用返回的token
curl -X GET "http://localhost:8000/api/v1/auth/me" \
  -H "Authorization: Bearer YOUR_TOKEN_HERE"
```

### 3. 测试权限控制

```bash
# 管理员端点（需要admin权限）
curl -X GET "http://localhost:8000/api/v1/auth/admin-test" \
  -H "Authorization: Bearer YOUR_TOKEN_HERE"
```

## 📁 项目结构

```
aegis-isle/
├── src/aegis_isle/           # 主要源码
│   ├── api/                  # FastAPI应用
│   ├── core/                 # 核心配置
│   ├── rag/                  # RAG管道
│   ├── agents/               # Agent系统
│   └── tools/                # 工具系统
├── logs/                     # 日志目录
│   ├── audit/                # 审计日志
│   ├── application/          # 应用日志
│   └── errors/               # 错误日志
├── data/                     # 数据目录
├── uploads/                  # 上传文件
├── .env                      # 环境配置
├── requirements.txt          # Python依赖
├── auth_server_simple.py     # 简化认证服务器
└── run_dev.py               # 开发启动脚本
```

## ⚙️ 环境配置

主要配置文件：`.env`

### 重要配置项

```env
# 环境
ENVIRONMENT=development
DEBUG=True

# API
API_HOST=0.0.0.0
API_PORT=8000

# 安全
SECRET_KEY=your-secret-key-here
ACCESS_TOKEN_EXPIRE_MINUTES=30

# OAuth2 + RBAC
ADMIN_USERNAME=admin
ADMIN_PASSWORD=admin123

# 审计日志
AUDIT_LOG_ENABLED=True
STRUCTURED_LOGGING=True
ELK_COMPATIBLE=True

# AI模型（完整模式）
LLM_PROVIDER=huggingface
EMBEDDING_MODEL=all-MiniLM-L6-v2
VECTOR_DB_TYPE=faiss
```

## 🐛 常见问题

### 1. 模块导入错误

**问题**: `ImportError: cannot import name 'xxx'`

**解决**:
```bash
pip install -r requirements.txt
```

### 2. 端口被占用

**问题**: `Address already in use`

**解决**:
```bash
# 查找占用端口的进程
netstat -ano | findstr :8000   # Windows
lsof -i :8000                  # Linux/Mac

# 使用不同端口
python run_dev.py --port 8080
```

### 3. 虚拟环境问题

**问题**: 依赖冲突或版本问题

**解决**:
```bash
# 删除并重建虚拟环境
rm -rf venv              # Linux/Mac
rmdir /s venv           # Windows

# 重新运行设置脚本
./setup_dev_env.sh      # Linux/Mac
setup_dev_env.bat       # Windows
```

### 4. AI模型下载问题

**问题**: 网络连接或模型下载失败

**解决**:
```bash
# 使用简化模式开发
python run_dev.py --mode auth

# 或配置代理后重试完整模式
```

## 🔍 调试技巧

### 1. 查看日志

```bash
# 应用日志
tail -f logs/application/app_*.log

# 审计日志
tail -f logs/audit/audit_*.jsonl

# 错误日志
tail -f logs/errors/error_*.log
```

### 2. 调试模式

在`.env`中设置：
```env
DEBUG=True
LOG_LEVEL=DEBUG
```

### 3. 数据库调试

```bash
# 查看SQLite数据库（如果使用）
sqlite3 aegis_isle.db
.tables
.schema agent_memory
```

## 🚀 部署准备

### 开发 → 生产清单

- [ ] 更新SECRET_KEY为安全随机值
- [ ] 设置强密码策略
- [ ] 配置真实数据库（PostgreSQL）
- [ ] 配置Redis缓存
- [ ] 设置ELK日志聚合
- [ ] 配置HTTPS/TLS
- [ ] 设置防火墙规则
- [ ] 配置监控告警

## 📚 更多资源

- [FastAPI 官方文档](https://fastapi.tiangolo.com/)
- [OAuth2 规范](https://oauth.net/2/)
- [JWT 标准](https://jwt.io/)
- [项目Wiki](https://github.com/your-org/aegis-isle/wiki)

---

**💡 提示**: 建议在开发阶段使用简化模式(`--mode auth`)，在需要测试完整AI功能时再切换到完整模式(`--mode full`)。