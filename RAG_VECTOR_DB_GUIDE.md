# RAG向量数据库配置指南

## 概述

AegisIsle支持多种向量数据库用于RAG文档检索：
- **FAISS** - Facebook AI Similarity Search (本地内存)
- **Qdrant** - 高性能向量搜索引擎
- **ChromaDB** - 简单易用的向量数据库

## 最新改进

### ✅ 增强的错误日志

所有向量数据库操作现在都包含详细的诊断日志：
- 步骤级别的操作跟踪
- 维度验证和不匹配检测
- 完整的异常堆栈跟踪
- 操作结果确认

### ✅ 修复的问题

1. **FAISS维度不匹配** - 现在自动检测embedding维度
2. **Qdrant错误处理** - 增强的错误捕获和日志记录
3. **审计日志格式** - 修复了ELK兼容的JSON格式错误
4. **时区处理** - 使用timezone-aware datetime替代弃用的utcnow()
5. **自定义OpenAI API端点** - 现在支持使用第三方OpenAI兼容API（如SiliconFlow、智谱AI等）

## 配置指南

### OpenAI兼容API配置

如果您使用第三方OpenAI兼容API服务，可以配置自定义端点：

**SiliconFlow**:
```env
OPENAI_API_KEY="sk-your-siliconflow-key"
OPENAI_BASE_URL="https://api.siliconflow.cn/v1"
LLM_PROVIDER=openai
DEFAULT_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
```

**智谱AI (GLM)**:
```env
OPENAI_API_KEY="your-zhipu-api-key"
OPENAI_BASE_URL="https://open.bigmodel.cn/api/paas/v4"
LLM_PROVIDER=openai
DEFAULT_LLM_MODEL=glm-4
```

**DeepSeek**:
```env
OPENAI_API_KEY="your-deepseek-key"
OPENAI_BASE_URL="https://api.deepseek.com/v1"
LLM_PROVIDER=openai
DEFAULT_LLM_MODEL=deepseek-chat
```

**注意**:
- 确保API密钥格式正确
- 模型名称必须是服务商支持的模型
- 不需要修改代码，系统会自动使用配置的base_url

### 选项1: 使用FAISS (推荐开发使用)

**优点**:
- 无需额外服务
- 快速启动
- 适合开发和测试

**配置** (.env):
```env
VECTOR_DB_TYPE=faiss
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

**注意**: FAISS现在自动使用embedding模型的实际维度(384 for all-MiniLM-L6-v2)

### 选项2: 使用Qdrant (推荐生产使用)

**优点**:
- 高性能
- 支持分布式
- 持久化存储
- 适合生产环境

**步骤1: 启动Qdrant服务**

使用Docker:
```bash
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
```

**步骤2: 配置** (.env):
```env
VECTOR_DB_TYPE=qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=aegis_isle_collection
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

**注意**: Qdrant collection会自动创建，维度设置为384 (all-MiniLM-L6-v2)

### 选项3: 使用ChromaDB

**优点**:
- 简单易用
- 持久化存储
- 适合中小规模应用

**配置** (.env):
```env
VECTOR_DB_TYPE=chromadb
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

## 测试文档上传

### 使用测试脚本

```bash
python test_rag_upload.py
```

### 通过API上传

```bash
# 获取token
TOKEN=$(curl -X POST "http://localhost:8000/api/v1/auth/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=admin123" \
  | jq -r '.access_token')

# 上传文档
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@rag_test_data.txt"
```

## 查看日志

### 应用日志
```bash
tail -f logs/application_$(date +%Y-%m-%d).log
```

### 错误日志
```bash
tail -f logs/errors_$(date +%Y-%m-%d).log
```

### 审计日志 (JSON格式)
```bash
tail -f logs/audit/audit_$(date +%Y-%m-%d).jsonl | jq
```

## 常见问题排查

### 1. OpenAI API错误 401 (Incorrect API key)

**症状**: `Error code: 401 - Incorrect API key provided`

**可能原因**:
1. API密钥格式错误或已过期
2. 使用第三方API但未配置`OPENAI_BASE_URL`
3. API密钥与服务商不匹配

**解决方案**:

**检查1: 验证API密钥**
```bash
# 测试SiliconFlow API
curl -H "Authorization: Bearer YOUR_API_KEY" \
  https://api.siliconflow.cn/v1/models

# 如果返回401，说明API密钥无效
```

**检查2: 确认.env配置**
```env
# SiliconFlow示例
OPENAI_API_KEY="sk-your-actual-key-here"  # 确保没有多余空格
OPENAI_BASE_URL="https://api.siliconflow.cn/v1"  # 必须配置
LLM_PROVIDER=openai
DEFAULT_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct  # 确认模型存在
```

**检查3: 重启服务**
```bash
# 修改.env后必须重启服务才能生效
# Ctrl+C停止当前服务，然后重新启动
python run_dev.py --mode full
```

**检查4: 查看详细日志**
```bash
# 查看错误日志中的完整错误信息
tail -f logs/errors_$(date +%Y-%m-%d).log

# 应该能看到 "Using custom OpenAI base URL: ..." 的日志
# 如果没有这条日志，说明base_url未生效
```

### 2. FAISS维度不匹配

**症状**: `Embedding dimension mismatch: got 384, expected 1536`

**解决**:
- 已修复！FAISS现在自动使用正确的维度
- 如果还有问题，删除旧的FAISS索引重新创建

### 3. Qdrant连接失败

**症状**: `Failed to initialize Qdrant: Connection refused`

**解决**:
```bash
# 检查Qdrant是否运行
curl http://localhost:6333/collections

# 如果未运行，启动Qdrant
docker run -p 6333:6333 qdrant/qdrant
```

### 3. Qdrant维度不匹配

**症状**: `Embedding dimension mismatch: got 384, collection expects 1536`

**解决**:
```bash
# 删除旧collection
curl -X DELETE "http://localhost:6333/collections/aegis_isle_collection"

# 重启服务，会自动创建正确维度的collection
```

### 4. 内存不足

**症状**: `页面文件太小，无法完成操作`

**解决**:
- 增加系统虚拟内存
- 使用更小的embedding模型
- 或者使用Qdrant替代FAISS

## 嵌入模型选择

### all-MiniLM-L6-v2 (默认)
- **维度**: 384
- **速度**: 快
- **质量**: 良好
- **适用**: 一般用途

### text-embedding-ada-002 (OpenAI)
- **维度**: 1536
- **速度**: 需要API调用
- **质量**: 优秀
- **适用**: 生产环境

### 切换模型

修改 `.env`:
```env
# 使用本地模型
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# 或使用OpenAI
EMBEDDING_MODEL=text-embedding-ada-002
OPENAI_API_KEY=your_api_key_here
```

**注意**: 切换模型后需要重新索引所有文档！

## 性能优化

### FAISS优化
- 使用`faiss-gpu`替代`faiss-cpu`(如果有GPU)
- 调整chunk_size和chunk_overlap

### Qdrant优化
- 使用SSD存储
- 增加Qdrant内存限制
- 使用Qdrant集群(生产环境)

### 通用优化
```env
# 调整分块参数
CHUNK_SIZE=1000          # 减小可提高精度，增加可提高速度
CHUNK_OVERLAP=200        # 重叠区域保证上下文连贯性

# 调整检索参数
MAX_RETRIEVED_DOCS=5     # 返回的文档数量
SIMILARITY_THRESHOLD=0.7 # 相似度阈值
```

## 监控和维护

### 查看向量数据库统计

**FAISS**:
- 索引保存在内存中
- 重启服务后需要重新索引

**Qdrant**:
```bash
# 查看collection信息
curl http://localhost:6333/collections/aegis_isle_collection

# 查看向量数量
curl http://localhost:6333/collections/aegis_isle_collection | jq '.result.points_count'
```

### 备份和恢复

**FAISS**: 实现持久化需要修改代码保存索引

**Qdrant**: 自动持久化到`qdrant_storage`目录

## 迁移指南

### 从FAISS迁移到Qdrant

1. 启动Qdrant服务
2. 修改`.env`中的`VECTOR_DB_TYPE=qdrant`
3. 重启服务
4. 重新上传所有文档

### 从Qdrant迁移到FAISS

1. 修改`.env`中的`VECTOR_DB_TYPE=faiss`
2. 重启服务
3. 重新上传所有文档

**注意**: 不同向量数据库之间无法直接迁移索引，需要重新索引文档。

## 技术支持

如遇到问题:
1. 查看详细错误日志 (`logs/errors_*.log`)
2. 检查配置是否正确 (`.env`)
3. 确认向量数据库服务正在运行
4. 查看本指南的"常见问题排查"部分

---

**最后更新**: 2025-11-27
**版本**: v1.1.0
**改进**: 增强错误日志、修复维度问题、修复审计日志格式
