---
name: fastapi-observability
description: 为 Aegis-Isle FastAPI 后端添加 Token 统计、延迟监控和审计日志
---

# FastAPI 可观测性技能

## 概述
为 Aegis-Isle 的 FastAPI 后端添加生产级可观测性，包括 Token 使用统计、响应延迟监控和 LLM 调用审计日志。

## 现有基础设施

### 已有组件 (直接增强)
| 组件 | 文件 | 现状 |
|:---|:---|:---|
| AuditLogger | `src/aegis_isle/core/logging.py` | ELK 兼容 JSON，支持 `log_api_access()` |
| MetricsMiddleware | `src/aegis_isle/api/middleware.py` | 请求计数、总耗时、错误率 |
| Settings | `src/aegis_isle/core/config.py` | `enable_metrics`, `audit_log_enabled` |
| GenerationResult | `src/aegis_isle/rag/generator.py` | `usage` 字段 (可存 token 数据) |

## 实施指南

### 1. Token 统计收集器
在 `src/aegis_isle/core/metrics.py` 中创建:

```python
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class TokenRecord:
    request_id: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    timestamp: datetime
    user_id: str = "default"

class TokenMetrics:
    def __init__(self, max_history: int = 100):
        self.history: deque[TokenRecord] = deque(maxlen=max_history)
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
    
    def record(self, record: TokenRecord):
        self.history.append(record)
        self.total_prompt_tokens += record.prompt_tokens
        self.total_completion_tokens += record.completion_tokens
    
    def get_dashboard(self) -> dict:
        return {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_requests": len(self.history),
            "avg_latency_ms": self._avg_latency(),
            "p95_latency_ms": self._percentile_latency(95),
        }
```

### 2. LLM 审计日志
在 `AuditLogger` 中添加 `log_llm_call()`:

```python
def log_llm_call(self, model, prompt_tokens, completion_tokens,
                 latency_ms, user_id, request_id, outcome="success"):
    self.log_event(
        event_type="model_inference",
        action="llm_completion",
        user_id=user_id,
        metadata={
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "latency_ms": latency_ms,
        },
        outcome=outcome,
    )
```

### 3. 延迟分位数
在 `MetricsMiddleware` 中添加:

```python
import statistics

class MetricsMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.latencies: deque = deque(maxlen=1000)
    
    def get_percentile(self, p: int) -> float:
        if not self.latencies:
            return 0.0
        sorted_lat = sorted(self.latencies)
        idx = int(len(sorted_lat) * p / 100)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]
```

### 4. Dashboard API 端点
在 `src/aegis_isle/api/routers/metrics.py` 中:

```python
from fastapi import APIRouter
router = APIRouter()

@router.get("/dashboard")
async def get_dashboard():
    return token_metrics.get_dashboard()

@router.get("/token-usage")
async def get_token_usage():
    return [asdict(r) for r in token_metrics.history]

@router.get("/export")
async def export_csv():
    # 导出为 CSV 文件下载
```

### 5. 在 test_server.py 中集成
在 `call_llm_streaming()` 返回后记录:
```python
token_metrics.record(TokenRecord(
    request_id=request_id,
    model="Qwen/Qwen2.5-7B-Instruct",
    prompt_tokens=response.usage.prompt_tokens,
    completion_tokens=response.usage.completion_tokens,
    ...
))
```

## 验证步骤
1. 在 SillyTavern 中对话 5 轮
2. 访问 `http://localhost:8001/api/v1/metrics/dashboard`
3. 确认 Token 统计递增、延迟数据正常
4. 查看 `logs/audit.log` 确认 LLM 调用日志
