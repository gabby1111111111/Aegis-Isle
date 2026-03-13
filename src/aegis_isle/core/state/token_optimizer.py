"""
Token 优化策略

策略:
1. 双模型配置(贵模型对话,便宜模型推理)
2. 增量状态注入(只注入最近变更)
3. 条件注入(智能判断是否需要完整状态)
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from ...core.logging import logger


class DualModelConfig:
    """双模型配置"""

    # 主对话模型(贵,质量高)
    CHAT_MODEL = "gpt-4"
    CHAT_MODEL_COST = 0.03  # per 1K tokens

    # 推理模型(便宜,用于状态提取)
    REASONING_MODEL = "gpt-4o-mini"
    REASONING_MODEL_COST = 0.001  # per 1K tokens

    @classmethod
    def get_cost_savings(cls) -> float:
        """计算成本节省比例"""
        return (
            (cls.CHAT_MODEL_COST - cls.REASONING_MODEL_COST) / cls.CHAT_MODEL_COST * 100
        )


class StateChangeTracker:
    """
    状态变更追踪器

    追踪最近的状态变更,用于增量注入
    """

    def __init__(self, max_changes: int = 5):
        """
        初始化

        Args:
            max_changes: 保留的最近变更数量
        """
        self._changes: List[Dict[str, Any]] = []
        self._max_changes = max_changes

    def record_change(
        self, change_type: str, description: str, timestamp: Optional[datetime] = None
    ):
        """
        记录一次变更

        Args:
            change_type: 变更类型(insert/update/delete)
            description: 变更描述
            timestamp: 时间戳
        """
        change = {
            "type": change_type,
            "description": description,
            "timestamp": timestamp or datetime.now(),
        }

        self._changes.append(change)

        # 只保留最近 N 条
        if len(self._changes) > self._max_changes:
            self._changes = self._changes[-self._max_changes :]

    def get_recent_changes(self, limit: int = 3) -> List[Dict[str, Any]]:
        """获取最近的变更"""
        return self._changes[-limit:] if self._changes else []

    def to_markdown(self, limit: int = 3) -> str:
        """
        转换为 Markdown 格式

        Returns:
            简洁的变更摘要
        """
        recent = self.get_recent_changes(limit)

        if not recent:
            return ""

        lines = ["## 📝 最近状态变更\n"]

        for change in reversed(recent):  # 最新的在前
            # 计算时间差
            time_ago = self._format_time_ago(change["timestamp"])

            # 格式化变更类型
            icon = {"insert": "➕", "update": "↻", "delete": "➖"}.get(
                change["type"], "•"
            )

            lines.append(f"{icon} [{time_ago}] {change['description']}")

        return "\n".join(lines) + "\n"

    def _format_time_ago(self, timestamp: datetime) -> str:
        """格式化时间差"""
        delta = datetime.now() - timestamp

        if delta < timedelta(minutes=1):
            return "刚刚"
        elif delta < timedelta(hours=1):
            minutes = int(delta.total_seconds() / 60)
            return f"{minutes}分钟前"
        elif delta < timedelta(days=1):
            hours = int(delta.total_seconds() / 3600)
            return f"{hours}小时前"
        else:
            days = delta.days
            return f"{days}天前"


class TokenOptimizer:
    """Token 优化器"""

    def __init__(self):
        self.change_tracker = StateChangeTracker(max_changes=10)
        self.turn_count = 0

    def should_inject_full_state(self, user_message: str) -> bool:
        """
        判断是否需要注入完整状态

        触发条件:
        1. 用户明确询问状态
        2. 每 N 轮对话注入一次(防止状态漂移)
        3. 用户执行重要操作

        Args:
            user_message: 用户消息

        Returns:
            是否注入完整状态
        """
        # 关键词检测
        query_keywords = ["背包", "物品", "任务", "状态", "属性", "有什么", "查看"]
        if any(kw in user_message for kw in query_keywords):
            logger.info("[TokenOptimizer] 检测到状态查询关键词,注入完整状态")
            return True

        # 每 5 轮对话注入一次完整状态
        self.turn_count += 1
        if self.turn_count % 5 == 0:
            logger.info(f"[TokenOptimizer] 第 {self.turn_count} 轮对话,注入完整状态")
            return True

        logger.info("[TokenOptimizer] 使用增量状态注入")
        return False

    def get_optimized_context(
        self, full_state_markdown: str, user_message: str, max_tokens: int = 2000
    ) -> str:
        """
        获取优化后的上下文

        Args:
            full_state_markdown: 完整状态
            user_message: 用户消息
            max_tokens: Token 限制

        Returns:
            优化后的状态字符串
        """
        if self.should_inject_full_state(user_message):
            # 注入完整状态
            return full_state_markdown
        else:
            # 注入增量变更
            incremental = self.change_tracker.to_markdown(limit=3)

            if incremental:
                return incremental
            else:
                # 如果没有最近变更,返回简化的状态摘要
                return self._create_summary(full_state_markdown)

    def _create_summary(self, full_state: str) -> str:
        """
        创建状态摘要

        只保留非空表格的标题
        """
        lines = []
        for line in full_state.split("\n"):
            if line.startswith("##"):
                lines.append(line)

        if lines:
            return "\n".join(lines) + "\n\n*(详细信息已省略,如需查看请询问)*"
        else:
            return ""

    def record_state_change(self, change_type: str, description: str):
        """记录状态变更"""
        self.change_tracker.record_change(change_type, description)

    def get_stats(self) -> Dict[str, Any]:
        """获取优化统计"""
        return {
            "turn_count": self.turn_count,
            "recent_changes": len(self.change_tracker.get_recent_changes(10)),
            "cost_savings": f"{DualModelConfig.get_cost_savings():.1f}%",
        }


# 全局实例
_global_optimizer = TokenOptimizer()


def get_optimizer() -> TokenOptimizer:
    """获取全局优化器实例"""
    return _global_optimizer


# 导出
__all__ = [
    "DualModelConfig",
    "StateChangeTracker",
    "TokenOptimizer",
    "get_optimizer",
]
