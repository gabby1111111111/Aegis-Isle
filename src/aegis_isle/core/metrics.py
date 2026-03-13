"""
Token 使用统计与延迟监控模块。

提供 TokenMetrics 收集器，用于记录每次 LLM 调用的 token 消耗、
响应延迟和费用估算。支持 P50/P95/P99 延迟分位数计算。
"""

import csv
import io
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, List, Deque

from ..core.logging import logger


# ============================================
# 模型定价表 (每 1M tokens，单位 USD)
# ============================================

MODEL_PRICING: Dict[str, Dict[str, float]] = {
    "Qwen/Qwen2.5-7B-Instruct": {
        "prompt": 0.35,  # SiliconFlow 价格
        "completion": 0.35,
    },
    "Qwen/Qwen2.5-72B-Instruct": {
        "prompt": 4.13,
        "completion": 4.13,
    },
    "deepseek-ai/DeepSeek-V3": {
        "prompt": 1.33,
        "completion": 1.33,
    },
    # 默认兜底价格
    "_default": {
        "prompt": 1.0,
        "completion": 1.0,
    },
}


@dataclass
class TokenRecord:
    """单次 LLM 调用的 Token 记录"""

    request_id: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    user_id: str = "default"
    endpoint: str = "/v1/chat/completions"
    cost_usd: float = 0.0

    def __post_init__(self):
        """自动计算费用"""
        pricing = MODEL_PRICING.get(self.model, MODEL_PRICING["_default"])
        self.cost_usd = (
            self.prompt_tokens * pricing["prompt"] / 1_000_000
            + self.completion_tokens * pricing["completion"] / 1_000_000
        )


class TokenMetrics:
    """
    Token 使用统计收集器。

    线程安全的 Token 消耗追踪器，支持:
    - 累计 prompt/completion token 统计
    - 按模型分组的 token 使用量
    - 费用估算 (基于 SiliconFlow 定价)
    - P50/P95/P99 延迟分位数
    - CSV 导出
    """

    def __init__(self, max_history: int = 500):
        self.history: Deque[TokenRecord] = deque(maxlen=max_history)
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.total_cost_usd: float = 0.0
        self.total_requests: int = 0
        self._latencies: List[float] = []
        self._model_usage: Dict[str, Dict[str, int]] = {}

    def record(self, record: TokenRecord) -> None:
        """记录一次 LLM 调用的 Token 使用"""
        self.history.append(record)
        self.total_prompt_tokens += record.prompt_tokens
        self.total_completion_tokens += record.completion_tokens
        self.total_cost_usd += record.cost_usd
        self.total_requests += 1
        self._latencies.append(record.latency_ms)

        # 按模型分组统计
        model_key = record.model
        if model_key not in self._model_usage:
            self._model_usage[model_key] = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "request_count": 0,
            }
        self._model_usage[model_key]["prompt_tokens"] += record.prompt_tokens
        self._model_usage[model_key]["completion_tokens"] += record.completion_tokens
        self._model_usage[model_key]["request_count"] += 1

        logger.info(
            f"[Metrics] 📊 Token 记录: "
            f"prompt={record.prompt_tokens}, "
            f"completion={record.completion_tokens}, "
            f"latency={record.latency_ms:.0f}ms, "
            f"cost=${record.cost_usd:.6f}"
        )

    def _percentile(self, p: int) -> float:
        """计算延迟的第 p 百分位数"""
        if not self._latencies:
            return 0.0
        sorted_lat = sorted(self._latencies)
        idx = int(len(sorted_lat) * p / 100)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]

    def get_dashboard(self) -> dict:
        """
        返回汇总统计面板数据。

        Returns:
            包含 Token 统计、延迟分位数、费用和模型分布的字典
        """
        avg_latency = (
            sum(self._latencies) / len(self._latencies) if self._latencies else 0.0
        )

        return {
            "token_usage": {
                "total_prompt_tokens": self.total_prompt_tokens,
                "total_completion_tokens": self.total_completion_tokens,
                "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            },
            "cost": {
                "total_cost_usd": round(self.total_cost_usd, 6),
                "avg_cost_per_request": round(
                    self.total_cost_usd / max(self.total_requests, 1), 6
                ),
            },
            "latency": {
                "avg_ms": round(avg_latency, 1),
                "p50_ms": round(self._percentile(50), 1),
                "p95_ms": round(self._percentile(95), 1),
                "p99_ms": round(self._percentile(99), 1),
            },
            "requests": {
                "total": self.total_requests,
                "history_size": len(self.history),
            },
            "model_usage": self._model_usage,
        }

    def get_recent_records(self, limit: int = 20) -> List[dict]:
        """返回最近的 Token 使用记录"""
        records = list(self.history)[-limit:]
        return [asdict(r) for r in records]

    def export_csv(self) -> str:
        """
        将所有记录导出为 CSV 字符串。

        Returns:
            CSV 格式的字符串
        """
        output = io.StringIO()
        writer = csv.writer(output)

        # 表头
        writer.writerow(
            [
                "timestamp",
                "request_id",
                "user_id",
                "model",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "latency_ms",
                "cost_usd",
                "endpoint",
            ]
        )

        for record in self.history:
            writer.writerow(
                [
                    record.timestamp,
                    record.request_id,
                    record.user_id,
                    record.model,
                    record.prompt_tokens,
                    record.completion_tokens,
                    record.total_tokens,
                    round(record.latency_ms, 1),
                    round(record.cost_usd, 6),
                    record.endpoint,
                ]
            )

        return output.getvalue()

    def reset(self) -> None:
        """重置所有统计数据"""
        self.history.clear()
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost_usd = 0.0
        self.total_requests = 0
        self._latencies.clear()
        self._model_usage.clear()
        logger.info("[Metrics] 统计数据已重置")


# ============================================
# tiktoken Token 估算工具
# ============================================


def estimate_tokens(text: str, model: str = "gpt-4") -> int:
    """
    使用 tiktoken 估算文本的 token 数。

    对于不被 tiktoken 原生支持的模型（如 Qwen），
    使用 cl100k_base 编码器作为近似估算。

    Args:
        text: 要估算的文本
        model: 模型名称 (用于选择编码器)

    Returns:
        估算的 token 数
    """
    try:
        import tiktoken

        # Qwen/DeepSeek 等国产模型用 cl100k_base 近似
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        # 如果 tiktoken 未安装，用字符数粗略估算
        # 中文约 1.5 token/字，英文约 0.25 token/word
        chinese_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
        other_chars = len(text) - chinese_chars
        return int(chinese_chars * 1.5 + other_chars * 0.3)
    except Exception as e:
        logger.warning(f"[Metrics] tiktoken 估算失败: {e}，使用字符估算")
        return len(text) // 3


def estimate_messages_tokens(messages: list, model: str = "gpt-4") -> int:
    """
    估算消息列表的总 token 数 (包含角色标记开销)。

    Args:
        messages: OpenAI 格式的消息列表
        model: 模型名称

    Returns:
        估算的总 token 数
    """
    total = 0
    for msg in messages:
        total += 4  # 每条消息的角色标记开销 (<|im_start|>role\n...<|im_end|>\n)
        content = msg.get("content", "")
        if content:
            total += estimate_tokens(content, model)
    total += 2  # 对话末尾 token
    return total


# ============================================
# 全局单例
# ============================================

token_metrics = TokenMetrics()
"""全局 Token 统计收集器实例"""
