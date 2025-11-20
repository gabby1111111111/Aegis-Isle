# LangGraph 多智能体系统升级指南

## 升级概述

本次升级将 Aegis-Isle 项目的多智能体编排系统从传统的 Workflow 架构迁移到基于 LangGraph 的状态图架构，并将路由器从关键字匹配升级为基于 LLM 的语义路由器。

## 主要变更

### 1. 依赖更新

在 `requirements.txt` 中添加了 LangGraph 支持：

```txt
langchain-core==0.1.15
langgraph==0.0.20
```

### 2. 架构升级

#### 原有架构 (Legacy)
- **Workflow + WorkflowStep**: 基于步骤依赖的线性工作流
- **KeywordRouter**: 基于关键字匹配的简单路由

#### 新架构 (LangGraph)
- **LangGraphAgentOrchestrator**: 基于状态图的智能体编排
- **LLMRouter**: 基于 LLM 语义理解的智能路由

### 3. 工作流架构

新的 LangGraph 工作流结构：

```
Router → [Researcher / Retriever / Summarizer / ChartGenerator] → Finalizer → END
```

#### 全局状态类 `AgentState`

```python
class AgentState(TypedDict):
    messages: List[BaseMessage]           # 消息历史
    context: Dict[str, Any]              # 上下文信息
    next_step: Optional[str]             # 下一步动作
    current_query: str                   # 当前查询
    agent_results: Dict[str, Any]        # 智能体结果
    final_answer: Optional[str]          # 最终答案
    execution_metadata: Dict[str, Any]   # 执行元数据
```

## 使用指南

### 基础用法 (保持向后兼容)

```python
from aegis_isle.agents import AgentOrchestrator, AgentRouter

# 创建路由器 (自动使用 LLM 路由)
router = AgentRouter()

# 创建编排器 (自动使用 LangGraph)
orchestrator = AgentOrchestrator(router)

# 执行工作流 (API 保持不变)
result = await orchestrator.execute_workflow(
    workflow_name="rag_query",
    initial_input="什么是人工智能？"
)
```

### 新的 LLM 路由器

```python
from aegis_isle.agents import LLMRouter, create_llm_router

# 方式1: 直接创建
llm_router = LLMRouter(provider="openai", model="gpt-4")

# 方式2: 工厂函数
llm_router = create_llm_router(provider="anthropic", model="claude-3")

# 注册智能体
llm_router.register_agent(research_agent)
llm_router.register_agent(retriever_agent)

# 智能路由 - LLM 将分析用户意图
target_agents = await llm_router.route_message("帮我分析这篇论文的主要观点")
```

### 新的 LangGraph 编排器

```python
from aegis_isle.agents import LangGraphAgentOrchestrator

# 直接使用 LangGraph 编排器
orchestrator = LangGraphAgentOrchestrator(router)

# 执行查询
result = await orchestrator.execute_workflow(
    query="分析人工智能在医疗领域的应用",
    initial_context={"domain": "healthcare", "focus": "AI applications"}
)

# 结果包含详细信息
print(f"最终答案: {result['final_answer']}")
print(f"执行时间: {result['execution_time']:.2f}秒")
print(f"智能体结果: {result['agent_results']}")
```

## LLM 路由器详解

### 语义理解路由

LLM 路由器通过以下方式工作：

1. **意图分析**: 使用 LLM 分析用户输入的语义意图
2. **JSON 响应**: LLM 返回结构化的路由决策
3. **智能体映射**: 将意图映射到最合适的智能体
4. **错误处理**: 自动降级到关键字路由作为备用

### 路由决策示例

用户输入: "帮我制作一个销售数据的图表"

LLM 分析返回:
```json
{
    "target_agent": "chart_generator",
    "reason": "用户请求创建图表，需要数据可视化功能"
}
```

### 可用的智能体类型

| 智能体角色 | 描述 | 适用场景 |
|------------|------|----------|
| researcher | 研究、搜索、调查 | "帮我研究量子计算的发展历史" |
| retriever | 检索文档、获取知识 | "从知识库中找到相关的医疗指南" |
| summarizer | 总结、摘要、概述 | "总结这份报告的要点" |
| chart_generator | 创建图表、可视化 | "制作一个销售趋势图" |
| tool_caller | 执行工具、API 调用 | "运行数据分析脚本" |
| coordinator | 协调任务、通用助理 | "帮我制定项目计划" |

## 迁移指导

### 从旧系统迁移

#### 1. 更新导入

```python
# 旧写法
from aegis_isle.agents import AgentRouter

# 新写法 (推荐)
from aegis_isle.agents import LLMRouter, create_llm_router
```

#### 2. 路由器升级

```python
# 旧的关键字路由
router = AgentRouter()  # 默认现在使用 LLM 路由

# 显式使用 LLM 路由
router = LLMRouter()

# 或者手动升级现有路由器
if not router.upgrade_to_llm_routing():
    print("LLM 路由初始化失败，使用关键字路由")
```

#### 3. 工作流定义

```python
# 旧写法 - 手动定义工作流步骤
workflow = Workflow("custom_flow")
step1 = WorkflowStep("retrieve", ["retriever"], "检索文档: {input}")
workflow.add_step(step1)

# 新写法 - 直接执行查询 (推荐)
result = await orchestrator.execute_workflow(
    query="检索文档相关信息"
)
```

## 性能优化

### LLM 路由器配置

```python
# 使用较快的模型进行路由
fast_router = LLMRouter(provider="openai", model="gpt-3.5-turbo")

# 使用本地模型 (如果配置了 Hugging Face)
local_router = LLMRouter(provider="huggingface", model="microsoft/DialoGPT-medium")
```

### 缓存和优化

LangGraph 编排器自动提供：
- 状态管理和持久化
- 并行执行支持
- 错误恢复机制
- 执行时间跟踪

## 错误处理

### 自动降级

```python
# LLM 路由失败时自动降级到关键字路由
router = AgentRouter(use_llm_routing=True)  # 自动处理降级

# 手动控制降级
try:
    router.upgrade_to_llm_routing()
except Exception:
    router.downgrade_to_keyword_routing()
```

### 调试信息

```python
# 开启详细日志
import logging
logging.getLogger('aegis_isle.agents').setLevel(logging.DEBUG)

# 检查路由决策
result = await router.route_message("分析数据")
# 日志将显示: "LLM router decision: researcher - Reason: 需要数据分析研究"
```

## API 兼容性

### 保持兼容的 API

所有现有的公共 API 都保持向后兼容：

- `AgentOrchestrator.execute_workflow()`
- `AgentRouter.route_message()`
- `AgentRouter.register_agent()`
- 所有配置类和枚举

### 新增的 API

```python
# 新的 LangGraph 特定方法
result = await orchestrator.execute_workflow(
    query="用户查询",
    initial_context={"key": "value"}
)

# 新的路由器控制方法
router.upgrade_to_llm_routing(provider="anthropic")
router.downgrade_to_keyword_routing()
```

## 总结

本次升级实现了以下目标：

1. ✅ **保留现有 API 兼容性** - 无需修改现有调用代码
2. ✅ **引入 LangGraph 架构** - 更强大的状态管理和工作流控制
3. ✅ **升级为语义路由** - 从关键字匹配升级到 LLM 意图理解
4. ✅ **增量重构** - 保持项目结构不变，无破坏性更改
5. ✅ **自动降级机制** - LLM 失败时自动回退到关键字路由

系统现在具备了更智能的路由能力和更灵活的工作流编排，为未来的扩展和优化奠定了坚实基础。