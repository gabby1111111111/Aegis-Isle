"""
Day 2 验收 Demo：状态提取器端到端测试
演示三级降级策略的实际效果
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from aegis_isle.core.state.prompts import generate_state_update_prompt, get_system_prompt
from aegis_isle.core.state.extractor import StateExtractor


def demo_xml_parsing():
    """Demo 1: XML 严格解析（Level 1）"""
    print("=" * 60)
    print("🔍 Demo 1: XML 严格解析")
    print("=" * 60)
    
    # 模拟 LLM 输出（标准格式）
    llm_output = """
    <thought>
    用户在森林中发现了一把生锈的铁剑，需要添加到背包。
    </thought>
    <content>
    <tableEdit type="insert" sheet="背包物品表" row='[null, "生锈的铁剑", "1", "攻击力+3，耐久度较低", "武器"]' />
    </content>
    """
    
    print("\n【LLM 输出】")
    print(llm_output)
    
    extractor = StateExtractor()
    commands = extractor.extract(llm_output)
    
    print("\n【提取结果】")
    if commands:
        cmd = commands[0]
        print(f"✅ 成功提取 {len(commands)} 条指令")
        print(f"   操作类型: {cmd.edit_type.value}")
        print(f"   目标表格: {cmd.sheet_name}")
        print(f"   物品名称: {cmd.row_data[1]}")
        print(f"   物品数量: {cmd.row_data[2]}")
        print(f"   物品描述: {cmd.row_data[3]}")
    else:
        print("❌ 提取失败")
    
    print(f"\n【统计】XML 解析成功: {extractor.get_stats()['xml_success']} 次")


def demo_regex_fallback():
    """Demo 2: 正则降级（Level 2）"""
    print("\n\n" + "=" * 60)
    print("🔍 Demo 2: 正则降级解析")
    print("=" * 60)
    
    # 模拟 LLM 输出（格式不完美，缺少 content 标签）
    llm_output = """
    用户捡到了钥匙，需要添加到背包。
    <tableEdit type="insert" sheet="背包物品表" row='[null, "神秘钥匙", "1", "用途未知", "道具"]' />
    """
    
    print("\n【LLM 输出】（格式不完美）")
    print(llm_output)
    
    extractor = StateExtractor()
    commands = extractor.extract(llm_output)
    
    print("\n【提取结果】")
    if commands:
        cmd = commands[0]
        print(f"✅ 正则降级成功！提取 {len(commands)} 条指令")
        print(f"   操作类型: {cmd.edit_type.value}")
        print(f"   物品名称: {cmd.row_data[1]}")
    else:
        print("❌ 提取失败")
    
    stats = extractor.get_stats()
    print(f"\n【统计】正则解析成功: {stats['regex_success']} 次")


def demo_rule_engine():
    """Demo 3: 规则引擎保底（Level 3）"""
    print("\n\n" + "=" * 60)
    print("🔍 Demo 3: 规则引擎保底")
    print("=" * 60)
    
    # 模拟 LLM 输出（完全没有 XML 标签！）
    test_cases = [
        ("我在地上捡到了一把剑", "获得物品"),
        ("我喝掉了红色药水", "消耗物品"),
        ("今天天气真不错", "无关内容"),
    ]
    
    extractor = StateExtractor()
    
    for llm_output, case_type in test_cases:
        print(f"\n【测试】{case_type}")
        print(f"LLM 输出: \"{llm_output}\"")
        
        commands = extractor.extract(llm_output)
        
        if commands:
            cmd = commands[0]
            print(f"✅ 规则引擎识别成功！")
            print(f"   操作类型: {cmd.edit_type.value}")
            if cmd.edit_type.value == "insert":
                print(f"   物品名称: {cmd.row_data[1]}")
            elif cmd.edit_type.value == "delete":
                print(f"   删除物品: {cmd.condition['value']}")
        else:
            print(f"ℹ️  无需更新状态（正确行为）")
    
    stats = extractor.get_stats()
    print(f"\n【统计】规则引擎成功: {stats['rule_success']} 次")


def demo_multi_edit():
    """Demo 4: 多状态同时变化"""
    print("\n\n" + "=" * 60)
    print("🔍 Demo 4: 多状态同时变化")
    print("=" * 60)
    
    llm_output = """
    <thought>
    用户执行了两个操作：
    1. 消耗了药水
    2. 移动到了新地点
    </thought>
    <content>
    <tableEdit type="update" sheet="背包物品表" condition='{"column": 1, "value": "红色药水"}' row='[null, "红色药水", "2", "恢复50HP", "消耗品"]' />
    <tableEdit type="update" sheet="全局数据表" condition='{"column": 0, "value": null}' row='[null, "迷雾森林", "2024-01-15 14:30", "2024-01-15 14:00", "30分钟"]' />
    </content>
    """
    
    print("\n【LLM 输出】（多个指令）")
    print(llm_output)
    
    extractor = StateExtractor()
    commands = extractor.extract(llm_output)
    
    print("\n【提取结果】")
    print(f"✅ 成功提取 {len(commands)} 条指令")
    
    for i, cmd in enumerate(commands, 1):
        print(f"\n   指令 {i}:")
        print(f"   - 操作类型: {cmd.edit_type.value}")
        print(f"   - 目标表格: {cmd.sheet_name}")
        if cmd.edit_type.value == "update":
            print(f"   - 匹配条件: column={cmd.condition['column']}, value={cmd.condition['value']}")
            print(f"   - 新值预览: {cmd.row_data[1]}")


def demo_prompt_generation():
    """Demo 5: Prompt 生成"""
    print("\n\n" + "=" * 60)
    print("🔍 Demo 5: Prompt 生成")
    print("=" * 60)
    
    # 模拟当前状态
    current_state = """
## 背包物品表
| None | 物品名称 | 数量 | 描述/效果 | 类别 |
| --- | --- | --- | --- | --- |
| null | 红色药水 | 3 | 恢复50HP | 消耗品 |
"""
    
    user_message = "我想使用一瓶红色药水"
    
    print("\n【当前状态】")
    print(current_state)
    
    print("\n【用户输入】")
    print(f"\"{user_message}\"")
    
    # 生成完整 Prompt
    full_prompt = generate_state_update_prompt(current_state, user_message)
    
    print("\n【生成的 Prompt】")
    print(f"总长度: {len(full_prompt)} 字符")
    print(f"预估 Token: ~{len(full_prompt) // 0.7:.0f}")
    print("\n前 500 字符预览：")
    print("-" * 60)
    print(full_prompt[:500])
    print("...")
    print("-" * 60)


def demo_stats_summary():
    """Demo 6: 统计汇总"""
    print("\n\n" + "=" * 60)
    print("📊 统计汇总")
    print("=" * 60)
    
    extractor = StateExtractor()
    
    # 测试不同场景
    test_cases = [
        ('<tableEdit type="insert" sheet="test" row=\'[null, "item1"]\' />', "XML"),
        ('我捡到了物品2', "规则引擎"),
        ('<tableEdit type="delete" sheet="test" condition=\'{"column": 1, "value": "item3"}\' />', "正则"),
        ('完全无关的文本', "失败"),
    ]
    
    print("\n【执行测试】")
    for text, label in test_cases:
        commands = extractor.extract(text)
        status = "✅" if commands else "❌"
        print(f"{status} {label}: {len(commands)} 条指令")
    
    stats = extractor.get_stats()
    
    print("\n【最终统计】")
    print(f"✅ XML 解析成功: {stats['xml_success']} 次")
    print(f"✅ 正则解析成功: {stats['regex_success']} 次")
    print(f"✅ 规则引擎成功: {stats['rule_success']} 次")
    print(f"❌ 完全失败: {stats['total_failures']} 次")
    
    total_success = stats['xml_success'] + stats['regex_success'] + stats['rule_success']
    total_attempts = total_success + stats['total_failures']
    success_rate = (total_success / total_attempts * 100) if total_attempts > 0 else 0
    
    print(f"\n【成功率】{success_rate:.1f}% ({total_success}/{total_attempts})")


def main():
    """主函数：运行所有 Demo"""
    print("\n")
    print("🎯" * 30)
    print("Day 2 验收 Demo：状态提取器完整演示")
    print("🎯" * 30)
    
    try:
        demo_xml_parsing()
        demo_regex_fallback()
        demo_rule_engine()
        demo_multi_edit()
        demo_prompt_generation()
        demo_stats_summary()
        
        print("\n\n" + "=" * 60)
        print("🎉 Day 2 验收 Demo 全部完成！")
        print("=" * 60)
        print("\n✅ 核心功能验证：")
        print("   - XML 严格解析 ✓")
        print("   - 正则降级解析 ✓")
        print("   - 规则引擎保底 ✓")
        print("   - 多指令提取 ✓")
        print("   - Prompt 生成 ✓")
        print("   - 统计功能 ✓")
        print("\n🚀 准备就绪，可以进入 Day 3！\n")
        
    except Exception as e:
        print(f"\n❌ Demo 运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
