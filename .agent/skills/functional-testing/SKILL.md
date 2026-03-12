---
description: FastAPI + Streamlit 功能测试与 RAG 质量评估（2026 年最佳实践）
---

# 功能测试 Skill

## 概述
本 Skill 覆盖三层功能测试，确保"用户体验"维度的质量：

1. **API 端点测试** — `httpx.AsyncClient` + `pytest-asyncio`
2. **Streamlit UI 测试** — `st.testing.v1.AppTest`
3. **RAG 检索质量评估** — `DeepEval` (可选)

## 依赖安装

```bash
# 必装
pip install httpx pytest-asyncio

# 可选 (RAG 评估, 需要 LLM API Key)
pip install deepeval
```

## 1. FastAPI API 功能测试

### 文件位置
`tests/test_functional_api.py`

### 运行方式
```bash
python -m pytest tests/test_functional_api.py -v --tb=short
```

### 编写规范
```python
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from src.aegis_isle.api.main import create_app

@pytest_asyncio.fixture
async def client():
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c

class TestMyEndpoint:
    @pytest.mark.asyncio
    async def test_scenario_name(self, client):
        """用中文描述这个测试模拟的用户场景"""
        resp = await client.post("/v1/endpoint", json={...})
        assert resp.status_code == 200
        data = resp.json()
        assert "expected_field" in data
```

### 关键原则
- 每个测试模拟一个真实用户操作
- 测试 HTTP 状态码和响应结构，不测内部实现
- 缺少字段时应返回 422（Pydantic 验证）
- 不需要真实 LLM API Key（端点应优雅降级）

## 2. Streamlit AppTest 功能测试

### 文件位置
`tests/test_streamlit_ui.py`

### 编写规范
```python
from streamlit.testing.v1 import AppTest

def test_charlife_empty_queue_shows_success():
    at = AppTest.from_file("frontend/charlife_review_app.py").run()
    at.session_state["pending_events"] = []
    at.run()
    assert len(at.success) > 0  # 应显示绿色成功消息

def test_interview_question_generation():
    at = AppTest.from_file("frontend/interview_app.py").run()
    # 验证初始页面加载没有异常
    assert not at.exception
```

### 关键原则
- 使用 `AppTest.from_file()` 加载应用
- 通过 `at.session_state` 注入测试数据
- 用 `.click().run()` 模拟按钮点击
- 用 `at.markdown`, `at.success`, `at.error` 检查输出

## 3. RAG 检索质量评估 (DeepEval)

### 文件位置
`tests/test_rag_quality.py`

### 编写规范
```python
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, ContextualRecallMetric

def test_memory_retrieval_quality():
    test_case = LLMTestCase(
        input="你还记得那次在法餐厅的事吗？",
        actual_output="那次在法餐厅，你点了鹅肝...",
        retrieval_context=["邹峥在法餐厅吃鹅肝，感觉紧张..."],
        expected_output="关于法餐厅鹅肝的回忆"
    )
    
    faithfulness = FaithfulnessMetric(threshold=0.7)
    recall = ContextualRecallMetric(threshold=0.7)
    
    assert_test(test_case, [faithfulness, recall])
```

### 评估指标说明
| 指标 | 含义 | 阈值建议 |
|:----:|:----:|:------:|
| Faithfulness | LLM 回答是否忠于检索到的上下文 | ≥ 0.7 |
| Contextual Recall | 检索器是否召回了相关文档 | ≥ 0.7 |
| Answer Relevancy | 回答是否与问题相关 | ≥ 0.7 |

## 检查清单

在提交前确认:

- [ ] `python -m pytest tests/test_functional_api.py -v` 全绿
- [ ] 新增的 API 端点有对应的功能测试
- [ ] 测试名称用中文 docstring 描述用户场景
- [ ] 不依赖外部服务（LLM、Redis 等）即可运行
