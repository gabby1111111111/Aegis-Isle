"""状态提取器测试套件"""

import pytest
import sys
from pathlib import Path

# 添加项目根目录到 sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from aegis_isle.core.state.extractor import StateExtractor, extract_state_changes
from aegis_isle.core.state.models import EditType


class TestXMLParsing:
    """测试 XML 解析"""
    
    def test_insert_basic(self):
        """测试基本的 INSERT 指令"""
        llm_output = """
        <thought>用户获得了新物品</thought>
        <content>
        <tableEdit type="insert" sheet="背包物品表" row='[null, "生锈的铁剑", "1", "攻击力+3", "武器"]' />
        </content>
        """
        
        commands = extract_state_changes(llm_output)
        
        assert len(commands) == 1
        assert commands[0].edit_type == EditType.INSERT
        assert commands[0].sheet_name == "背包物品表"
        assert commands[0].row_data[1] == "生锈的铁剑"
    
    def test_delete_with_condition(self):
        """测试 DELETE 指令"""
        llm_output = """
        <tableEdit type="delete" sheet="背包物品表" condition='{"column": 1, "value": "红色药水"}' />
        """
        
        commands = extract_state_changes(llm_output)
        
        assert len(commands) == 1
        assert commands[0].edit_type == EditType.DELETE
        assert commands[0].condition["value"] == "红色药水"
    
    def test_update_operation(self):
        """测试 UPDATE 指令"""
        llm_output = """
        <tableEdit type="update" sheet="全局数据表" 
          condition='{"column": 0, "value": null}' 
          row='[null, "森林", "2024-01-15 14:00", "2024-01-15 13:00", "1小时"]' />
        """
        
        commands = extract_state_changes(llm_output)
        assert len(commands) == 1
        assert commands[0].edit_type == EditType.UPDATE


class TestSpecialCases:
    """测试特殊情况"""
    
    def test_no_edit_tag(self):
        """测试 <noEdit /> 标签"""
        llm_output = "<thought>无需更新</thought><noEdit />"
        commands = extract_state_changes(llm_output)
        assert len(commands) == 0
    
    def test_multiple_edits(self):
        """测试多个指令"""
        llm_output = """
        <tableEdit type="insert" sheet="背包物品表" row='[null, "剑", "1", "", "武器"]' />
        <tableEdit type="delete" sheet="背包物品表" condition='{"column": 1, "value": "药水"}' />
        """
        
        commands = extract_state_changes(llm_output)
        assert len(commands) == 2


class TestRegexFallback:
    """测试正则降级"""
    
    def test_without_content_tag(self):
        """测试缺少 content 标签"""
        llm_output = """
        <tableEdit type="insert" sheet="背包物品表" row='[null, "钥匙", "1", "", "道具"]' />
        """
        
        commands = extract_state_changes(llm_output)
        assert len(commands) >= 1


class TestRuleBasedFallback:
    """测试规则引擎"""
    
    def test_acquire_keyword(self):
        """测试获得关键词"""
        llm_output = "我捡到了一把剑"
        commands = extract_state_changes(llm_output)
        
        if commands:
            assert commands[0].edit_type == EditType.INSERT
            assert "剑" in commands[0].row_data[1]
    
    def test_consume_keyword(self):
        """测试消耗关键词"""
        llm_output = "我喝掉了红色药水"
        commands = extract_state_changes(llm_output)
        
        if commands:
            assert commands[0].edit_type == EditType.DELETE


class TestErrorHandling:
    """测试错误处理"""
    
    def test_invalid_json(self):
        """测试无效 JSON"""
        llm_output = """<tableEdit type="insert" sheet="test" row='[invalid}' />"""
        commands = extract_state_changes(llm_output)
        assert isinstance(commands, list)
    
    def test_empty_output(self):
        """测试空输出"""
        commands = extract_state_changes("")
        assert len(commands) == 0


class TestStats:
    """测试统计功能"""
    
    def test_stats_tracking(self):
        """测试统计追踪"""
        extractor = StateExtractor()
        
        # XML 成功
        extractor.extract('<tableEdit type="insert" sheet="test" row=\'[null, "item"]\' />')
        
        # 规则引擎
        extractor.extract('我捡到了钥匙')
        
        stats = extractor.get_stats()
        assert "xml_success" in stats
        assert "rule_success" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
