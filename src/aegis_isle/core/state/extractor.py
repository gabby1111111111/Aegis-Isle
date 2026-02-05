"""
State Extractor: 三级降级策略的状态提取器

Level 1: XML 严格解析
Level 2: 正则模糊匹配
Level 3: 规则引擎保底
"""

import re
import json
from typing import List, Optional
from xml.etree import ElementTree as ET

from .models import TableEditCommand, EditType
from ...core.logging import logger


class StateExtractor:
    """
    状态提取器:从 LLM 输出中提取结构化指令
    
    Features:
    - 三级降级策略(XML → Regex → Rules)
    - 统计功能(调试用)
    - 详细日志
    """
    
    def __init__(self):
        self.stats = {
            "xml_success": 0,
            "regex_success": 0,
            "rule_success": 0,
            "total_failures": 0
        }
    
    def extract(self, llm_output: str) -> List[TableEditCommand]:
        """
        提取状态编辑指令
        
        Args:
            llm_output: LLM 的原始输出
            
        Returns:
            TableEditCommand 列表
        """
        # 检查 <noEdit /> 标签
        if "<noEdit" in llm_output or "<noEdit/>" in llm_output:
            logger.info("LLM 指示无需更新状态")
            return []
        
        # 三级降级策略
        commands = (
            self._try_xml_parse(llm_output) or
            self._try_regex_parse(llm_output) or
            self._try_rule_based_parse(llm_output)
        )
        
        if commands:
            logger.info(f"成功提取 {len(commands)} 条状态编辑指令")
            return commands
        else:
            logger.warning("未能从 LLM 输出中提取任何指令")
            self.stats["total_failures"] += 1
            return []
    
    def _try_xml_parse(self, text: str) -> Optional[List[TableEditCommand]]:
        """Level 1: 尝试严格 XML 解析"""
        try:
            xml_text = f"<root>{text}</root>"
            root = ET.fromstring(xml_text)
            
            commands = []
            for elem in root.findall(".//tableEdit"):
                cmd = self._parse_xml_element(elem)
                if cmd:
                    commands.append(cmd)
            
            if commands:
                self.stats["xml_success"] += 1
                logger.debug(f"XML 解析成功: {len(commands)} 条指令")
                return commands
                
        except ET.ParseError as e:
            logger.debug(f"XML 解析失败: {e}")
        except Exception as e:
            logger.debug(f"XML 解析异常: {e}")
        
        return None
    
    def _parse_xml_element(self, elem: ET.Element) -> Optional[TableEditCommand]:
        """解析单个 <tableEdit> 元素"""
        try:
            edit_type = EditType(elem.get("type"))
            sheet_name = elem.get("sheet")
            
            row_str = elem.get("row")
            row_data = json.loads(row_str) if row_str else None
            
            condition_str = elem.get("condition")
            condition = json.loads(condition_str) if condition_str else None
            
            return TableEditCommand(
                edit_type=edit_type,
                sheet_name=sheet_name,
                row_data=row_data,
                condition=condition
            )
        except Exception as e:
            logger.warning(f"解析 XML 元素失败: {e}")
            return None
    
    def _try_regex_parse(self, text: str) -> Optional[List[TableEditCommand]]:
        """Level 2: 尝试正则表达式解析"""
        try:
            # Pattern 1: INSERT/UPDATE(带 row)
            pattern_row = r'<tableEdit\s+type="(\w+)"\s+sheet="([^"]+)"\s+row=\'([^\']+)\'\s*/>'
            matches_row = re.findall(pattern_row, text)
            
            # Pattern 2: DELETE(带 condition)
            pattern_delete = r'<tableEdit\s+type="delete"\s+sheet="([^"]+)"\s+condition=\'([^\']+)\'\s*/>'
            matches_delete = re.findall(pattern_delete, text)
            
            # Pattern 3: UPDATE(带 condition 和 row)
            pattern_update = r'<tableEdit\s+type="update"\s+sheet="([^"]+)"\s+condition=\'([^\']+)\'\s+row=\'([^\']+)\'\s*/>'
            matches_update = re.findall(pattern_update, text)
            
            commands = []
            
            # 解析 INSERT/UPDATE(仅 row)
            for match in matches_row:
                edit_type, sheet_name, row_str = match
                try:
                    row_data = json.loads(row_str)
                    commands.append(TableEditCommand(
                        edit_type=EditType(edit_type),
                        sheet_name=sheet_name,
                        row_data=row_data
                    ))
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"JSON 解析失败: {e}")
            
            # 解析 DELETE
            for match in matches_delete:
                sheet_name, condition_str = match
                try:
                    condition = json.loads(condition_str)
                    commands.append(TableEditCommand(
                        edit_type=EditType.DELETE,
                        sheet_name=sheet_name,
                        condition=condition
                    ))
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON 解析失败: {e}")
            
            # 解析 UPDATE(带 condition)
            for match in matches_update:
                sheet_name, condition_str, row_str = match
                try:
                    condition = json.loads(condition_str)
                    row_data = json.loads(row_str)
                    commands.append(TableEditCommand(
                        edit_type=EditType.UPDATE,
                        sheet_name=sheet_name,
                        row_data=row_data,
                        condition=condition
                    ))
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON 解析失败: {e}")
            
            if commands:
                self.stats["regex_success"] += 1
                logger.debug(f"正则解析成功: {len(commands)} 条指令")
                return commands
                
        except Exception as e:
            logger.debug(f"正则解析失败: {e}")
        
        return None
    
    def _try_rule_based_parse(self, text: str) -> Optional[List[TableEditCommand]]:
        """Level 3: 规则引擎(关键词匹配)"""
        commands = []
        
        # 规则1:物品获取
        acquire_keywords = ["获得", "捡到", "得到", "拾取", "发现"]
        for keyword in acquire_keywords:
            if keyword in text:
                match = re.search(rf'{keyword}[了]?(.{{1,10}})', text)
                if match:
                    item_name = match.group(1).strip("了。，、")
                    # 清理常见的后缀词
                    item_name = re.sub(r'(一把|一个|一件|一套)', '', item_name).strip()
                    commands.append(TableEditCommand(
                        edit_type=EditType.INSERT,
                        sheet_name="背包物品表",
                        row_data=[None, item_name, "1", "描述待补充", "其他"]
                    ))
                    logger.info(f"规则引擎: 检测到物品获取 '{item_name}'")
                    break
        
        # 规则2:物品消耗
        consume_keywords = ["使用", "消耗", "喝掉", "吃掉", "用掉"]
        for keyword in consume_keywords:
            if keyword in text:
                match = re.search(rf'{keyword}[了]?(.{{1,10}})', text)
                if match:
                    item_name = match.group(1).strip("了。，、")
                    item_name = re.sub(r'(一瓶|一个|一份)', '', item_name).strip()
                    commands.append(TableEditCommand(
                        edit_type=EditType.DELETE,
                        sheet_name="背包物品表",
                        condition={"column": 1, "value": item_name}
                    ))
                    logger.info(f"规则引擎: 检测到物品消耗 '{item_name}'")
                    break
        
        if commands:
            self.stats["rule_success"] += 1
            logger.debug(f"规则引擎成功: {len(commands)} 条指令")
            return commands
        
        return None
    
    def get_stats(self) -> dict:
        """获取统计数据"""
        return self.stats.copy()


# 便捷函数
def extract_state_changes(llm_output: str) -> List[TableEditCommand]:
    """便捷函数:提取状态变化"""
    extractor = StateExtractor()
    return extractor.extract(llm_output)


# 导出
__all__ = [
    "StateExtractor",
    "extract_state_changes",
]
