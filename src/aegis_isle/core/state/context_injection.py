"""
上下文注入模块

功能:
- 将用户状态注入到 Prompt
- 智能摘要(避免 Token 超限)
"""

from typing import List, Dict, Any
from ...core.logging import logger


def inject_state_context(
    messages: List[Dict[str, Any]],
    state_markdown: str,
    max_tokens: int = 2000
) -> List[Dict[str, Any]]:
    """
    将状态注入到消息列表
    
    Args:
        messages: 原始消息列表
        state_markdown: 状态的 Markdown 字符串
        max_tokens: Token 上限(粗略估算:1 token ≈ 0.7 字符)
        
    Returns:
        注入状态后的消息列表
        
    Strategy:
        - 如果状态较小(< max_tokens * 0.7 字符):完整注入
        - 如果状态较大:注入摘要版本
    """
    # 粗略估算 token 数(中文:1字符≈1.5token,英文:1字符≈0.25token)
    estimated_tokens = len(state_markdown) * 0.7
    
    if estimated_tokens < max_tokens:
        # 策略1:完整注入
        state_content = f"""## 当前用户状态

{state_markdown}

---

请根据对话内容更新状态。如果需要更新,请输出 <tableEdit> 指令。
"""
        logger.debug(f"完整注入状态,长度: {len(state_markdown)} 字符")
    else:
        # 策略2:摘要注入
        summary = summarize_state(state_markdown)
        state_content = f"""## 当前用户状态(摘要)

{summary}

---

请根据对话内容更新状态。
"""
        logger.warning(f"状态过长,使用摘要版本。原长度: {len(state_markdown)}, 摘要后: {len(summary)}")
    
    # 插入为独立的 system message
    state_message = {
        "role": "system",
        "content": state_content
    }
    
    # 查找第一个 system message 的位置
    system_index = None
    for i, msg in enumerate(messages):
        if msg.get("role") == "system":
            system_index = i
            break
    
    # 在第一个 system message 后插入状态
    if system_index is not None:
        return messages[:system_index+1] + [state_message] + messages[system_index+1:]
    else:
        # 如果没有 system message,插入到开头
        return [state_message] + messages


def summarize_state(state_markdown: str) -> str:
    """
    状态摘要:只保留非空表格
    
    Args:
        state_markdown: 完整状态字符串
        
    Returns:
        摘要后的状态字符串
        
    Strategy:
        1. 保留所有表头(## 开头)
        2. 只保留有实际数据的行(不是分隔符,不是空行)
        3. 过滤掉只有 null 的行
    """
    lines = state_markdown.split('\n')
    summary_lines = []
    
    for line in lines:
        # 保留表头
        if line.startswith('##'):
            summary_lines.append(line)
            continue
        
        # 保留表格分隔符(但后续会检查表是否为空)
        if '|' in line:
            # 跳过纯分隔符行
            if line.strip().replace('|', '').replace('-', '').replace(' ', '') == '':
                summary_lines.append(line)
                continue
            
            # 检查是否为有效数据行
            # 有效数据行:包含非 null、非空的实际内容
            cells = [c.strip() for c in line.split('|')]
            has_data = any(
                cell and cell.lower() not in ['null', 'none', '---', '']
                for cell in cells
            )
            
            if has_data:
                summary_lines.append(line)
    
    result = '\n'.join(summary_lines)
    
    # 如果摘要后还是很长,进一步压缩
    if len(result) > 1500:
        # 只保留背包和任务表
        priority_tables = ['背包物品表', '任务与事件表']
        filtered_lines = []
        include_section = False
        
        for line in summary_lines:
            if line.startswith('##'):
                # 检查是否为优先表
                include_section = any(table in line for table in priority_tables)
            
            if include_section or line.startswith('##'):
                filtered_lines.append(line)
        
        result = '\n'.join(filtered_lines)
        logger.info("状态进一步压缩,只保留优先表格")
    
    return result


def get_user_id_from_request(request_data: Dict[str, Any]) -> str:
    """
    从请求数据中提取用户 ID
    
    Args:
        request_data: 请求的 JSON 数据
        
    Returns:
        用户 ID
        
    Strategy:
        1. 从 request_data["user"] 字段获取
        2. 从 request_data["metadata"]["user_id"] 获取
        3. 默认返回 "default"
    """
    # 方案1:标准字段
    user_id = request_data.get("user")
    if user_id:
        return user_id
    
    # 方案2:元数据
    metadata = request_data.get("metadata", {})
    user_id = metadata.get("user_id")
    if user_id:
        return user_id
    
    # 方案3:默认用户
    logger.warning("未找到用户 ID,使用默认值 'default'")
    return "default"


# 导出
__all__ = [
    "inject_state_context",
    "summarize_state",
    "get_user_id_from_request",
]
