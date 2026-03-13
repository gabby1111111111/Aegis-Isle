"""
后台状态更新模块

功能:
- 异步提取状态变更
- 异步应用变更并保存
- 完整的错误处理和日志
"""

import asyncio
from typing import Optional
from ...core.logging import logger
from .manager import StateManager
from .extractor import StateExtractor
from .snapshot import SnapshotManager


async def update_user_state(
    user_id: str, llm_output: str, user_message: Optional[str] = None
) -> bool:
    """
    后台异步更新用户状态(集成快照)

    Args:
        user_id: 用户唯一标识
        llm_output: LLM 的完整输出
        user_message: 用户的原始输入(可选,用于日志)

    Returns:
        是否更新成功

    Workflow:
        1. 提取状态变更指令
        2. 加载当前用户状态
        2.5. 创建快照(更新前)
        3. 应用编辑指令
        4. 保存更新后的状态
        5. 清理旧快照
    """
    try:
        logger.info(f"[StateUpdate] 开始处理用户 {user_id} 的状态更新")

        if user_message:
            logger.debug(f"[StateUpdate] 用户消息: {user_message[:50]}...")

        # Step 1: 提取状态变更指令
        extractor = StateExtractor()
        commands = extractor.extract(llm_output)

        if not commands:
            logger.info(f"[StateUpdate] 用户 {user_id} 无状态变化(<noEdit /> 或无指令)")
            return True  # 无变化也算成功

        logger.info(f"[StateUpdate] 提取到 {len(commands)} 条编辑指令")
        for i, cmd in enumerate(commands, 1):
            logger.debug(f"  指令 {i}: {cmd.edit_type.value} → {cmd.sheet_name}")

        # Step 2: 加载当前状态
        state_manager = StateManager()
        user_state = await state_manager.load_state(user_id)
        logger.debug(f"[StateUpdate] 已加载用户状态,版本: {user_state.version}")

        # 🆕 Step 2.5: 创建快照(更新前)
        snapshot_manager = SnapshotManager()
        change_summary = f"{len(commands)} 个变更: " + ", ".join(
            f"{cmd.edit_type.value} {cmd.sheet_name}" for cmd in commands
        )

        snapshot = await snapshot_manager.create_snapshot(user_state, change_summary)
        if snapshot:
            logger.info(f"[Snapshot] 已创建快照 {snapshot.snapshot_id}")

        # Step 3: 应用编辑指令
        updated_state = await state_manager.apply_edits(user_state, commands)

        # Step 4: 保存更新后的状态
        success = await state_manager.save_state(updated_state)

        if success:
            logger.info(f"[StateUpdate] ✅ 用户 {user_id} 状态更新成功")

            # 输出变更摘要
            for cmd in commands:
                if cmd.edit_type.value == "insert":
                    logger.info(
                        f"  + 插入到 {cmd.sheet_name}: {cmd.row_data[1] if len(cmd.row_data) > 1 else 'N/A'}"
                    )
                elif cmd.edit_type.value == "update":
                    logger.info(f"  ↻ 更新 {cmd.sheet_name}")
                elif cmd.edit_type.value == "delete":
                    logger.info(
                        f"  - 删除自 {cmd.sheet_name}: {cmd.condition.get('value', 'N/A')}"
                    )

            # 🆕 Step 5: 清理旧快照
            deleted = await snapshot_manager.cleanup_old_snapshots(
                user_id, keep_count=10
            )
            if deleted > 0:
                logger.info(f"[Snapshot] 清理了 {deleted} 个旧快照")

            return True
        else:
            logger.error(f"[StateUpdate] ❌ 用户 {user_id} 状态保存失败")
            return False

    except Exception as e:
        logger.error(f"[StateUpdate] ❌ 用户 {user_id} 状态更新异常: {e}")
        import traceback

        traceback.print_exc()
        return False


async def update_user_state_with_retry(
    user_id: str,
    llm_output: str,
    user_message: Optional[str] = None,
    max_retries: int = 2,
) -> bool:
    """
    带重试机制的状态更新

    Args:
        user_id: 用户 ID
        llm_output: LLM 输出
        user_message: 用户消息
        max_retries: 最大重试次数

    Returns:
        是否最终成功
    """
    for attempt in range(max_retries + 1):
        if attempt > 0:
            logger.warning(f"[StateUpdate] 用户 {user_id} 重试第 {attempt} 次")
            await asyncio.sleep(0.5 * attempt)  # 指数退避

        success = await update_user_state(user_id, llm_output, user_message)

        if success:
            return True

    logger.error(f"[StateUpdate] 用户 {user_id} 在 {max_retries} 次重试后仍失败")
    return False


# 导出
__all__ = [
    "update_user_state",
    "update_user_state_with_retry",
]
