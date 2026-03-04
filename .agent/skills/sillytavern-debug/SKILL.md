---
name: sillytavern-debug
description: SillyTavern 与 Aegis-Isle 连接问题的诊断与修复流程
---

# SillyTavern 调试技能

## 概述
当 SillyTavern 无法正常连接 Aegis-Isle 后端，或对话出现异常时，按照以下检查清单逐步排查。

## 快速诊断清单

### 1. 服务器是否在运行？
```bash
# 检查端口 8001 是否被占用
netstat -ano | findstr :8001

# 如果没有输出，启动服务器
uvicorn test_server:app --host 0.0.0.0 --port 8001
```

### 2. 端口冲突？
```bash
# 强制释放端口（杀掉所有 Python 进程）
taskkill /F /IM python.exe /T

# 重新启动
uvicorn test_server:app --host 0.0.0.0 --port 8001
```

### 3. SillyTavern 连接配置
在 SillyTavern 的 API 设置中确认:
- **API Type**: `Chat Completion (OpenAI)`
- **Custom Endpoint**: `http://localhost:8001/v1`
- **API Key**: 任意非空字符串 (如 `sk-test`)
- **Model**: 任意 (后端会强制替换为 `Qwen/Qwen2.5-7B-Instruct`)

### 4. 常见错误排查

| 错误 | 原因 | 解决 |
|:---|:---|:---|
| 404 Not Found | 路由不匹配 | 确认 URL 为 `/v1/chat/completions` |
| 400 Bad Request | 参数不兼容 | 检查 `test_server.py` 的消息清洗逻辑 |
| 500 Internal Error | 后端异常 | 查看终端日志，检查 API Key |
| Connection Refused | 服务器未启动 | `netstat -ano \| findstr :8001` |
| Errno 10048 | 端口被占用 | `taskkill /F /IM python.exe /T` |

### 5. API Key 检查
```bash
# 确认 .env 中有正确的 SiliconFlow API Key
type .env | findstr OPENAI_API_KEY
# 应显示: OPENAI_API_KEY=sk-xxx...
```

### 6. 手动测试端点
```bash
# 测试健康检查
curl http://127.0.0.1:8001/health

# 测试聊天
curl -X POST http://127.0.0.1:8001/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"test\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"stream\":false}"
```

### 7. 状态文件检查
```bash
# 查看用户状态是否正常
type data\state\default.json | python -m json.tool
```

## SillyTavern 启动
```bash
# 启动 SillyTavern
Start-Process "E:\SillyTaven\SillyTavern\Start.bat"
```

## 已知问题
- SillyTavern 发送 `gpt-4` 作为 model，后端需要强制替换为实际模型名
- SillyTavern 的 messages 中可能包含 `name` 字段，需要清洗
- 使用 `127.0.0.1` 而非 `localhost` 以避免 DNS 解析问题
