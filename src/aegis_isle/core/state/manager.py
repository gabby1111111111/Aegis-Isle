"""
State Manager: Handles loading, saving, and editing user state.

⚠️ WARNING: Current implementation uses asyncio.Lock which is NOT
safe for multi-process deployments. For MVP, run with `workers=1`.
TODO: Migrate to filelock for production.

Provides file-based storage for UserState objects with basic concurrency safety.
"""

import json
import asyncio
import shutil
from pathlib import Path
from typing import List

from .models import UserState, Sheet, TableEditCommand, EditType, SheetMetadata
from ...core.logging import logger


class StateManager:
    """
    Manages user state persistence and manipulation.

    Features:
    - JSON file-based storage
    - Basic file locking for concurrency safety
    - CRUD operations on sheets
    """

    def __init__(self, state_dir: str = "data/state"):
        """
        Initialize StateManager.

        Args:
            state_dir: Directory to store state JSON files
        """
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._locks = {}  # Per-user locks for concurrency control

        # WARNING: Only safe in single-process mode
        logger.warning(
            "StateManager initialized with asyncio.Lock. "
            "For production, use filelock for multi-process safety. "
            "Current setup only works with workers=1."
        )

    def _get_lock(self, user_id: str) -> asyncio.Lock:
        """Get or create a lock for a specific user."""
        if user_id not in self._locks:
            self._locks[user_id] = asyncio.Lock()
        return self._locks[user_id]

    def _get_state_path(self, user_id: str) -> Path:
        """Get the file path for a user's state."""
        return self.state_dir / f"{user_id}.json"

    async def load_state(self, user_id: str) -> UserState:
        """
        Load user state from disk.

        Args:
            user_id: User identifier

        Returns:
            UserState object (creates new if file doesn't exist)
        """
        async with self._get_lock(user_id):
            state_path = self._get_state_path(user_id)

            if not state_path.exists():
                logger.info(f"No existing state for user {user_id}, creating new state")
                return self._create_default_state(user_id)

            try:
                with open(state_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                state = UserState(**data)
                logger.info(
                    f"Loaded state for user {user_id} with {len(state.sheets)} sheets"
                )
                return state

            except Exception as e:
                logger.error(f"Failed to load state for {user_id}: {e}")
                logger.warning("Creating new default state as fallback")
                return self._create_default_state(user_id)

    async def save_state(self, state: UserState) -> bool:
        """
        Save user state to disk using atomic write.

        Args:
            state: UserState object to save

        Returns:
            True if successful, False otherwise

        Notes:
            Uses temp file + rename for atomic operation to prevent file corruption.
        """
        user_id = state.user_id
        async with self._get_lock(user_id):
            state_path = self._get_state_path(user_id)
            tmp_path = state_path.with_suffix(".tmp")  # Temporary file

            try:
                # Create backup of existing state (optional safety net)
                if state_path.exists():
                    backup_path = state_path.with_suffix(".json.bak")
                    try:
                        shutil.copy2(state_path, backup_path)
                    except Exception as backup_error:
                        logger.warning(f"Failed to create backup: {backup_error}")

                # Write to temporary file first
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(
                        state.model_dump(by_alias=True), f, ensure_ascii=False, indent=2
                    )
                    f.flush()  # Force write to disk
                    # Try to sync to disk (Unix/Linux only, graceful fail on Windows)
                    try:
                        import os

                        os.fsync(f.fileno())
                    except (AttributeError, OSError):
                        pass  # Windows or unsupported platform

                # Atomic rename (safe even if power fails during rename)
                tmp_path.replace(state_path)

                logger.info(f"Saved state for user {user_id} using atomic write")
                return True

            except Exception as e:
                logger.error(f"Failed to save state for {user_id}: {e}")
                # Cleanup temporary file if it exists
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except Exception as cleanup_error:
                        logger.error(f"Failed to cleanup temp file: {cleanup_error}")
                return False

    def _create_default_state(self, user_id: str) -> UserState:
        """
        Create a default UserState with empty sheets based on Shujuku template.

        Args:
            user_id: User identifier

        Returns:
            UserState with initialized sheets
        """
        # Define default sheets based on Shujuku's DEFAULT_TABLE_TEMPLATE_ACU
        default_sheets = {
            "sheet_global": Sheet(
                uid="sheet_global",
                name="全局数据表",
                source_data=SheetMetadata(
                    note="记录当前主角所在地点及时间相关参数。",
                    initNode="插入一条关于当前世界状态的记录。",
                    deleteNode="禁止删除。",
                    updateNode="当主角从当前所在区域离开时,更新所在地点。每轮必须更新时间。",
                ),
                content=[
                    [None, "主角当前所在地点", "当前时间", "上轮场景时间", "经过的时间"]
                ],
                orderNo=0,
            ),
            "sheet_hero": Sheet(
                uid="sheet_hero",
                name="主角信息",
                source_data=SheetMetadata(
                    note="记录主角的核心身份信息。",
                    initNode="游戏初始化时,插入主角的唯一条目。",
                    deleteNode="禁止删除。",
                    updateNode="'过往经历'列会根据剧情发展持续增量更新。",
                ),
                content=[
                    [
                        None,
                        "人物名称",
                        "性别/年龄",
                        "外貌特征",
                        "职业/身份",
                        "过往经历",
                        "性格特点",
                    ]
                ],
                orderNo=1,
            ),
            "sheet_inventory": Sheet(
                uid="sheet_inventory",
                name="背包物品表",
                source_data=SheetMetadata(
                    note="记录主角拥有的所有物品、装备。",
                    initNode="游戏初始化时,根据剧情与设定添加主角的初始携带物品。",
                    deleteNode="物品被完全消耗、丢弃或摧毁时删除。",
                    updateNode="获得已有的物品,使其数量增加时更新。",
                    insertNode="主角获得背包中没有的全新物品时添加。",
                ),
                content=[[None, "物品名称", "数量", "描述/效果", "类别"]],
                orderNo=4,
            ),
            "sheet_quest": Sheet(
                uid="sheet_quest",
                name="任务与事件表",
                source_data=SheetMetadata(
                    note="记录所有当前正在进行的任务。",
                    initNode="游戏初始化时,根据剧情与设定添加一条主线剧情。",
                    deleteNode="任务完成、失败或过期时删除。",
                    updateNode="任务取得关键进展时进行更新。",
                    insertNode="主角接取或触发新的主线或支线任务时添加。",
                ),
                content=[
                    [
                        None,
                        "任务名称",
                        "任务类型",
                        "发布者",
                        "详细描述",
                        "当前进度",
                        "任务时限",
                        "奖励",
                        "惩罚",
                    ]
                ],
                orderNo=5,
            ),
        }

        return UserState(user_id=user_id, sheets=default_sheets, version=1)

    async def apply_edits(
        self, state: UserState, commands: List[TableEditCommand]
    ) -> UserState:
        """
        Apply a list of table edit commands to the state.

        Args:
            state: Current UserState
            commands: List of TableEditCommand to apply

        Returns:
            Updated UserState
        """
        for cmd in commands:
            try:
                sheet = state.get_sheet_by_name(cmd.sheet_name)
                if not sheet:
                    logger.warning(f"Sheet '{cmd.sheet_name}' not found, skipping edit")
                    continue

                if cmd.edit_type == EditType.INSERT:
                    if cmd.row_data:
                        sheet.add_row(cmd.row_data)
                        logger.info(f"Inserted row into '{cmd.sheet_name}'")

                elif cmd.edit_type == EditType.UPDATE:
                    # Simple update logic: find row by condition and replace
                    if cmd.condition and cmd.row_data:
                        col_idx = cmd.condition.get("column", 0)
                        target_value = cmd.condition.get("value")

                        updated = False
                        for i, row in enumerate(sheet.get_rows(), start=1):
                            if len(row) > col_idx and row[col_idx] == target_value:
                                sheet.content[i] = cmd.row_data
                                logger.info(
                                    f"Updated row in '{cmd.sheet_name}' (matched: {target_value})"
                                )
                                updated = True
                                break

                        if not updated:
                            logger.warning(
                                f"No matching row found for update in '{cmd.sheet_name}' "
                                f"(looking for column {col_idx} = {target_value})"
                            )

                elif cmd.edit_type == EditType.DELETE:
                    if cmd.condition:
                        col_idx = cmd.condition.get("column", 0)
                        target_value = cmd.condition.get("value")

                        rows_to_keep = [sheet.content[0]]  # Keep header
                        for row in sheet.get_rows():
                            if not (
                                len(row) > col_idx and row[col_idx] == target_value
                            ):
                                rows_to_keep.append(row)

                        sheet.content = rows_to_keep
                        logger.info(f"Deleted row from '{cmd.sheet_name}'")

            except Exception as e:
                logger.error(f"Failed to apply edit command: {e}")
                continue

        return state

    def get_context_string(self, state: UserState) -> str:
        """
        Get formatted state string for LLM context injection.

        Args:
            state: UserState object

        Returns:
            Markdown-formatted string
        """
        return state.to_context_string()
