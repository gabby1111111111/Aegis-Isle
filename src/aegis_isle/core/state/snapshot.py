"""
状态快照与回滚系统

功能:
- 自动创建状态快照
- 支持回滚到指定快照
- 自动清理旧快照
"""

import json
from pathlib import Path
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field

from .models import UserState
from ...core.logging import logger


class StateSnapshot(BaseModel):
    """状态快照模型"""

    snapshot_id: str = Field(description="快照唯一标识,格式:snap_YYYYMMDD_HHMMSS")
    user_id: str = Field(description="用户ID")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="快照创建时间"
    )
    version: int = Field(description="状态版本号")
    change_summary: str = Field(default="", description="变更摘要说明")
    file_path: str = Field(description="快照文件路径")


class SnapshotManager:
    """
    快照管理器

    特性:
    - 自动创建快照(每次状态更新前)
    - 快照元数据管理
    - 自动清理旧快照
    """

    def __init__(self, base_dir: str = "data/snapshots"):
        """
        初始化快照管理器

        Args:
            base_dir: 快照存储根目录
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"SnapshotManager initialized at {self.base_dir}")

    def _get_user_snapshot_dir(self, user_id: str) -> Path:
        """获取用户快照目录"""
        user_dir = self.base_dir / user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir

    def _get_metadata_path(self, user_id: str) -> Path:
        """获取元数据文件路径"""
        return self._get_user_snapshot_dir(user_id) / "metadata.json"

    def _load_metadata(self, user_id: str) -> List[StateSnapshot]:
        """加载快照元数据"""
        metadata_path = self._get_metadata_path(user_id)

        if not metadata_path.exists():
            return []

        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            snapshots = []
            for item in data.get("snapshots", []):
                # 解析 timestamp
                if isinstance(item.get("timestamp"), str):
                    item["timestamp"] = datetime.fromisoformat(item["timestamp"])
                snapshots.append(StateSnapshot(**item))

            return snapshots
        except Exception as e:
            logger.error(f"Failed to load metadata for {user_id}: {e}")
            return []

    def _save_metadata(self, user_id: str, snapshots: List[StateSnapshot]) -> bool:
        """保存快照元数据"""
        metadata_path = self._get_metadata_path(user_id)

        try:
            data = {
                "user_id": user_id,
                "last_updated": datetime.now().isoformat(),
                "snapshot_count": len(snapshots),
                "snapshots": [
                    {
                        "snapshot_id": snap.snapshot_id,
                        "user_id": snap.user_id,
                        "timestamp": snap.timestamp.isoformat(),
                        "version": snap.version,
                        "change_summary": snap.change_summary,
                        "file_path": snap.file_path,
                    }
                    for snap in snapshots
                ],
            }

            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            return True
        except Exception as e:
            logger.error(f"Failed to save metadata for {user_id}: {e}")
            return False

    async def create_snapshot(
        self, user_state: UserState, change_summary: str = ""
    ) -> Optional[StateSnapshot]:
        """
        创建状态快照

        Args:
            user_state: 当前用户状态
            change_summary: 变更摘要

        Returns:
            StateSnapshot 对象,失败返回 None
        """
        try:
            user_id = user_state.user_id

            # 生成快照 ID (精确到微秒，防止高频覆盖)
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            snapshot_id = f"snap_{timestamp_str}"

            # 快照文件路径
            snapshot_dir = self._get_user_snapshot_dir(user_id)
            snapshot_file = snapshot_dir / f"{snapshot_id}.json"

            # 保存状态到快照文件
            with open(snapshot_file, "w", encoding="utf-8") as f:
                json.dump(
                    user_state.model_dump(by_alias=True),
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            # 创建快照对象
            snapshot = StateSnapshot(
                snapshot_id=snapshot_id,
                user_id=user_id,
                timestamp=datetime.now(),
                version=user_state.version,
                change_summary=change_summary,
                file_path=str(snapshot_file),
            )

            # 更新元数据
            snapshots = self._load_metadata(user_id)
            snapshots.append(snapshot)
            self._save_metadata(user_id, snapshots)

            logger.info(f"Created snapshot {snapshot_id} for user {user_id}")
            return snapshot

        except Exception as e:
            logger.error(f"Failed to create snapshot: {e}")
            return None

    async def rollback_to_snapshot(
        self, user_id: str, snapshot_id: str
    ) -> Optional[UserState]:
        """
        回滚到指定快照

        Args:
            user_id: 用户 ID
            snapshot_id: 快照 ID

        Returns:
            恢复的 UserState,失败返回 None
        """
        try:
            # 查找快照
            snapshots = self._load_metadata(user_id)
            target_snapshot = None

            for snap in snapshots:
                if snap.snapshot_id == snapshot_id:
                    target_snapshot = snap
                    break

            if not target_snapshot:
                logger.error(f"Snapshot {snapshot_id} not found for user {user_id}")
                return None

            # 加载快照文件
            snapshot_path = Path(target_snapshot.file_path)

            if not snapshot_path.exists():
                logger.error(f"Snapshot file not found: {snapshot_path}")
                return None

            with open(snapshot_path, "r", encoding="utf-8") as f:
                state_data = json.load(f)

            # 解析状态
            restored_state = UserState(**state_data)

            logger.info(f"Rolled back user {user_id} to snapshot {snapshot_id}")
            return restored_state

        except Exception as e:
            logger.error(f"Failed to rollback: {e}")
            return None

    async def list_snapshots(
        self, user_id: str, limit: int = 10
    ) -> List[StateSnapshot]:
        """
        列出用户的快照

        Args:
            user_id: 用户 ID
            limit: 返回数量限制

        Returns:
            快照列表(按时间倒序)
        """
        snapshots = self._load_metadata(user_id)

        # 按时间倒序排序
        snapshots.sort(key=lambda s: s.timestamp, reverse=True)

        return snapshots[:limit]

    async def cleanup_old_snapshots(self, user_id: str, keep_count: int = 10) -> int:
        """
        清理旧快照,保留最近 N 个

        Args:
            user_id: 用户 ID
            keep_count: 保留的快照数量

        Returns:
            删除的快照数量
        """
        try:
            snapshots = self._load_metadata(user_id)

            if len(snapshots) <= keep_count:
                return 0

            # 按时间排序
            snapshots.sort(key=lambda s: s.timestamp, reverse=True)

            # 保留最近的 N 个
            to_keep = snapshots[:keep_count]
            to_delete = snapshots[keep_count:]

            # 删除旧快照文件
            deleted_count = 0
            for snap in to_delete:
                try:
                    snapshot_path = Path(snap.file_path)
                    if snapshot_path.exists():
                        snapshot_path.unlink()
                        deleted_count += 1
                except Exception as e:
                    logger.error(f"Failed to delete snapshot file: {e}")

            # 更新元数据
            self._save_metadata(user_id, to_keep)

            logger.info(f"Cleaned up {deleted_count} old snapshots for user {user_id}")
            return deleted_count

        except Exception as e:
            logger.error(f"Failed to cleanup snapshots: {e}")
            return 0


# 导出
__all__ = [
    "StateSnapshot",
    "SnapshotManager",
]
