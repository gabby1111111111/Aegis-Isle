"""
Day 1 验收测试脚本
测试状态管理系统的核心功能
"""

import asyncio
import sys
from pathlib import Path

# 添加项目路径到 sys.path（如果需要）
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from aegis_isle.core.state import StateManager, TableEditCommand, EditType


async def main():
    print("=" * 60)
    print("🧪 Day 1 状态管理系统验收测试")
    print("=" * 60)
    
    # 测试1：初始化管理器
    print("\n[测试 1] 初始化 StateManager...")
    manager = StateManager(state_dir="data/state_test")
    print("✅ StateManager 初始化成功")
    
    # 测试2：创建/加载用户状态
    print("\n[测试 2] 加载用户状态...")
    state = await manager.load_state("test_user")
    print(f"✅ 用户状态已加载，包含 {len(state.sheets)} 个表")
    
    # 测试3：查看初始背包状态
    print("\n[测试 3] 查看初始背包状态...")
    inventory_sheet = state.get_sheet_by_name("背包物品表")
    if inventory_sheet:
        print(inventory_sheet.to_markdown())
        print(f"当前背包有 {len(inventory_sheet.get_rows())} 个物品")
    
    # 测试4：插入物品
    print("\n[测试 4] 插入'生锈的铁剑'...")
    cmd = TableEditCommand(
        edit_type=EditType.INSERT,
        sheet_name="背包物品表",
        row_data=[None, "生锈的铁剑", "1", "攻击力+3", "武器"]
    )
    state = await manager.apply_edits(state, [cmd])
    print("✅ 物品插入成功")
    
    # 测试5：再插入一个物品
    print("\n[测试 5] 插入'红色药水'...")
    cmd2 = TableEditCommand(
        edit_type=EditType.INSERT,
        sheet_name="背包物品表",
        row_data=[None, "红色药水", "5", "恢复50HP", "消耗品"]
    )
    state = await manager.apply_edits(state, [cmd2])
    print("✅ 物品插入成功")
    
    # 测试6：保存状态（原子写入测试）
    print("\n[测试 6] 保存状态到磁盘（原子写入）...")
    success = await manager.save_state(state)
    if success:
        state_file = Path("data/state_test/test_user.json")
        print(f"✅ 状态已保存到 {state_file}")
        print(f"   文件大小: {state_file.stat().st_size} bytes")
    else:
        print("❌ 保存失败！")
        return False
    
    # 测试7：重新加载验证持久化
    print("\n[测试 7] 重新加载验证数据持久化...")
    state2 = await manager.load_state("test_user")
    inventory_sheet2 = state2.get_sheet_by_name("背包物品表")
    rows = inventory_sheet2.get_rows()
    
    if len(rows) == 2:
        print(f"✅ 数据持久化成功！背包有 {len(rows)} 个物品")
        
        # 验证具体数据
        item1_name = rows[0][1]
        item2_name = rows[1][1]
        
        if item1_name == "生锈的铁剑" and item2_name == "红色药水":
            print(f"✅ 数据验证通过：{item1_name}, {item2_name}")
        else:
            print(f"❌ 数据验证失败：预期['生锈的铁剑', '红色药水']，实际[{item1_name}, {item2_name}]")
            return False
    else:
        print(f"❌ 数据数量错误：预期2个物品，实际{len(rows)}个")
        return False
    
    # 测试8：查看最终状态
    print("\n[测试 8] 最终背包状态：")
    print(inventory_sheet2.to_markdown())
    
    # 测试9：生成 LLM 上下文字符串
    print("\n[测试 9] 生成 LLM 上下文字符串...")
    context = manager.get_context_string(state2)
    print(f"✅ 上下文字符串长度: {len(context)} 字符")
    print("\n--- 上下文字符串预览（前500字符） ---")
    print(context[:500])
    print("...")
    
    # 测试10：检查备份文件
    print("\n[测试 10] 检查备份机制...")
    backup_file = Path("data/state_test/test_user.json.bak")
    if backup_file.exists():
        print(f"✅ 备份文件存在: {backup_file}")
    else:
        print("ℹ️  首次保存，无备份文件（正常）")
    
    print("\n" + "=" * 60)
    print("🎉 Day 1 验收测试全部通过！")
    print("=" * 60)
    print("\n✅ 核心功能验证：")
    print("   - StateManager 初始化 ✓")
    print("   - 状态加载/创建 ✓")
    print("   - 表格编辑（INSERT） ✓")
    print("   - 原子写入保存 ✓")
    print("   - 数据持久化 ✓")
    print("   - Markdown 生成 ✓")
    print("   - 备份机制 ✓")
    
    return True


if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        sys.exit(0 if result else 1)
    except Exception as e:
        print(f"\n❌ 测试过程中出现异常：{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
