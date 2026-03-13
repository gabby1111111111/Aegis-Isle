"""
端到端稳定性测试

测试场景:
- 连续 10 轮对话
- 状态一致性验证
- 快照系统验证
"""

import pytest
import asyncio
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aegis_isle.core.state.manager import StateManager
from aegis_isle.core.state.background_updater import update_user_state
from aegis_isle.core.state.snapshot import SnapshotManager


class TestE2EStability:
    """端到端稳定性测试"""

    @pytest.mark.asyncio
    async def test_10_round_conversation(self):
        """测试连续 10 轮对话的状态一致性"""

        user_id = "e2e_10_rounds"
        # 清理旧数据以防状态累积导致断言失败
        state_file = Path(f"data/state/{user_id}.json")
        if state_file.exists():
            state_file.unlink()

        manager = StateManager()

        # 对话场景(模拟真实使用)
        conversations = [
            # 轮次1:获得装备
            {
                "user": "我在新手村的商店买了一把木剑",
                "llm": """
                <thought>用户购买了木剑,添加到背包</thought>
                <content>
                <tableEdit type="insert" sheet="背包物品表" row='[null, "木剑", "1", "攻击力+5", "武器"]' />
                </content>
                你购买了木剑！
                """,
                "expect": "背包应该有木剑",
            },
            # 轮次2:获得药水
            {
                "user": "村长送了我 3 瓶红色药水",
                "llm": """
                <tableEdit type="insert" sheet="背包物品表" row='[null, "红色药水", "3", "恢复50HP", "消耗品"]' />
                """,
                "expect": "背包应该有药水",
            },
            # 轮次3:使用药水
            {
                "user": "我喝掉了一瓶红色药水",
                "llm": """
                <tableEdit type="update" sheet="背包物品表" condition='{"column": 1, "value": "红色药水"}' row='[null, "红色药水", "2", "恢复50HP", "消耗品"]' />
                """,
                "expect": "药水数量减少到 2",
            },
            # 轮次4:接受任务
            {
                "user": "村长让我去清除哥布林",
                "llm": """
                <tableEdit type="insert" sheet="任务与事件表" row='[null, "清除哥布林", "主线", "村长", "击杀 10 只哥布林", "0/10", "无", "金币×100", "无"]' />
                """,
                "expect": "任务列表有新任务",
            },
            # 轮次5:任务进度更新
            {
                "user": "我击杀了 3 只哥布林",
                "llm": """
                <tableEdit type="update" sheet="任务与事件表" condition='{"column": 1, "value": "清除哥布林"}' row='[null, "清除哥布林", "主线", "村长", "击杀 10 只哥布林", "3/10", "无", "金币×100", "无"]' />
                """,
                "expect": "任务进度更新",
            },
            # 轮次6-10:继续对话
            {
                "user": "我又击杀了 5 只哥布林",
                "llm": """
                <tableEdit type="update" sheet="任务与事件表" condition='{"column": 1, "value": "清除哥布林"}' row='[null, "清除哥布林", "主线", "村长", "击杀 10 只哥布林", "8/10", "无", "金币×100", "无"]' />
                """,
                "expect": "进度 8/10",
            },
            {
                "user": "我完成了任务",
                "llm": """
                <tableEdit type="delete" sheet="任务与事件表" condition='{"column": 1, "value": "清除哥布林"}' />
                """,
                "expect": "任务完成并删除",
            },
            {
                "user": "我获得了奖励 100 金币",
                "llm": "<noEdit />",
                "expect": "无状态变化",
            },
            {
                "user": "我再次使用了药水",
                "llm": """
                <tableEdit type="update" sheet="背包物品表" condition='{"column": 1, "value": "红色药水"}' row='[null, "红色药水", "1", "恢复50HP", "消耗品"]' />
                """,
                "expect": "药水剩余 1",
            },
            {"user": "查看我的背包", "llm": "<noEdit />", "expect": "无变化,仅查询"},
        ]

        print("\n" + "=" * 70)
        print("🧪 开始 10 轮对话稳定性测试")
        print("=" * 70 + "\n")

        # 执行对话
        for i, conv in enumerate(conversations, 1):
            print(f"📍 轮次 {i}: {conv['user'][:30]}...")

            # 后台更新状态
            success = await update_user_state(
                user_id=user_id, llm_output=conv["llm"], user_message=conv["user"]
            )

            assert success, f"轮次 {i} 状态更新失败"

            # 短暂延迟
            await asyncio.sleep(0.1)

            print(f"   ✅ 更新成功 - {conv['expect']}")

        # 最终验证
        print("\n" + "=" * 70)
        print("🔍 验证最终状态")
        print("=" * 70 + "\n")

        final_state = await manager.load_state(user_id)

        # 验证背包
        inventory = final_state.get_sheet_by_name("背包物品表")
        inv_rows = inventory.get_rows()

        print(f"📦 背包物品数: {len(inv_rows)}")
        for row in inv_rows:
            print(f"   - {row[1]} × {row[2]}")

        # 验证应该有 2 个物品(木剑 + 红色药水)
        assert len(inv_rows) == 2, f"背包应该有 2 个物品,实际有 {len(inv_rows)}"

        # 验证药水数量
        potion_row = [r for r in inv_rows if "药水" in str(r[1])]
        assert len(potion_row) == 1, "应该有红色药水"
        assert potion_row[0][2] == "1", f"药水数量应该是 1,实际是 {potion_row[0][2]}"

        # 验证任务列表(应该为空)
        quests = final_state.get_sheet_by_name("任务与事件表")
        quest_rows = quests.get_rows()
        assert len(quest_rows) == 0, (
            f"任务应该已完成,任务列表应为空,实际有 {len(quest_rows)} 个任务"
        )

        print("\n✅ 10 轮对话测试通过,状态逻辑自洽！\n")

    @pytest.mark.asyncio
    async def test_snapshot_during_conversation(self):
        """测试对话过程中的快照系统"""

        user_id = "e2e_snapshot"
        # 清理旧数据
        state_file = Path(f"data/state/{user_id}.json")
        if state_file.exists():
            state_file.unlink()

        # 清理旧快照
        import shutil

        snapshot_dir = Path(f"data/snapshots/{user_id}")
        if snapshot_dir.exists():
            shutil.rmtree(snapshot_dir)

        manager = StateManager()
        snapshot_manager = SnapshotManager()

        print("\n" + "=" * 70)
        print("🧪 测试快照系统在对话中的表现")
        print("=" * 70 + "\n")

        # 初始状态
        await manager.load_state(user_id)

        # 轮次1:添加物品
        llm1 = """
        <tableEdit type="insert" sheet="背包物品表" row='[null, "宝剑", "1", "攻击力+10", "武器"]' />
        """
        await update_user_state(user_id, llm1, "我获得了宝剑")
        print("✅ 轮次1:添加宝剑")

        # 轮次2:添加药水(这次应该自动创建快照)
        llm2 = """
        <tableEdit type="insert" sheet="背包物品表" row='[null, "蓝色药水", "5", "恢复魔法", "消耗品"]' />
        """
        await update_user_state(user_id, llm2, "我获得了蓝色药水")
        print("✅ 轮次2:添加蓝色药水(自动创建快照)")

        # 列出快照
        snapshots = await snapshot_manager.list_snapshots(user_id, limit=10)
        print(f"\n📸 当前快照数: {len(snapshots)}")

        # 应该至少有 2 个快照
        assert len(snapshots) >= 2, f"应该有至少 2 个快照,实际有 {len(snapshots)}"

        # 验证最新快照
        latest_snapshot = snapshots[0]
        print(f"   最新快照: {latest_snapshot.snapshot_id}")
        print(f"   变更摘要: {latest_snapshot.change_summary}")

        # 轮次3:误操作(删除所有物品)
        llm3 = """
        <tableEdit type="delete" sheet="背包物品表" condition='{"column": 1, "value": "宝剑"}' />
        <tableEdit type="delete" sheet="背包物品表" condition='{"column": 1, "value": "蓝色药水"}' />
        """
        await update_user_state(user_id, llm3, "我不小心丢弃了所有物品")
        print("\n⚠️  轮次3:误操作,删除所有物品")

        # 验证背包为空
        state_after_delete = await manager.load_state(user_id)
        inventory_after = state_after_delete.get_sheet_by_name("背包物品表")
        assert len(inventory_after.get_rows()) == 0, "背包应该为空"
        print("   背包已清空")

        # 回滚到前一个快照（即创建药水前的快照，里面应该有宝剑）
        print(f"\n🔄 回滚到快照: {snapshots[0].snapshot_id}")
        restored_state = await snapshot_manager.rollback_to_snapshot(
            user_id, snapshots[0].snapshot_id
        )

        assert restored_state is not None, "回滚失败"

        # 保存回滚后的状态
        await manager.save_state(restored_state)

        # 验证物品已恢复
        inventory_restored = restored_state.get_sheet_by_name("背包物品表")
        restored_rows = inventory_restored.get_rows()

        print(f"   回滚后背包物品数: {len(restored_rows)}")
        assert len(restored_rows) >= 1, "回滚后应该恢复物品"

        print("\n✅ 快照回滚测试通过！\n")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
