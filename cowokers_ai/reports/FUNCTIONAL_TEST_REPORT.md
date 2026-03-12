# 功能测试报告 — API 端点 (用户体验视角)

- 日期: 2026-03-11
- 分支: auto-fix/cleanup-root @ 9731b22
- 测试工具: `httpx.AsyncClient` + `pytest-asyncio`  
- 结果: **12/12 PASSED** ✅ (耗时 ~78s)

---

## 测试场景覆盖

| # | 测试场景 | 模拟的用户操作 | 结果 |
|:-:|:------:|:---------:|:---:|
| 1 | 访问根路径 | 浏览器打开首页 | ✅ 200 |
| 2 | 查看系统信息 | GET /info | ✅ 返回版本+功能开关 |
| 3 | ST发送无messages的请求 | POST /v1/chat/completions | ✅ 200 (优雅处理) |
| 4 | ST发送空messages的请求 | POST /v1/chat/completions | ✅ 不崩溃 |
| 5 | 记忆检索(有效请求) | POST /v1/memory/search | ✅ 200 + 结构化响应 |
| 6 | 记忆检索(缺字段) | 少写query字段 | ✅ 422 验证拦截 |
| 7 | 获取宇宙列表 | GET /v1/memory/universes | ✅ 200 + 空数组 |
| 8 | 发送日记事件 | POST /v1/diary/event | ✅ 200 |
| 9 | 日记事件(缺字段) | 少写source字段 | ✅ 422 验证拦截 |
| 10 | 查看用户状态 | GET /v1/state/新用户 | ✅ 200 + 默认状态 |
| 11 | 查看快照列表 | GET /v1/state/用户/snapshots | ✅ 200 |
| 12 | ST完整流程模拟 | 健康检查→记忆查询→发消息 | ✅ 全链路通 |

---

## 🔍 功能测试发现的真实问题

### 发现 1: `chat/completions` 缺少输入验证 ⚠️

| 字段 | 详情 |
|:----:|:----:|
| 端点 | `POST /v1/chat/completions` |
| 问题 | 不传 `messages` 字段时，端点仍返回 200（使用 `Request.json()` 手动解析） |
| 影响 | SillyTavern 如果因 bug 发了空请求，不会收到任何错误提示 |
| 建议 | 添加显式验证: `if not messages: return JSONResponse(status_code=400, ...)` |

### 发现 2: universes 路由设计不符合 RESTful 直觉 ⚠️

| 字段 | 详情 |
|:----:|:----:|
| 端点 | `GET /v1/memory/universes?character_name=xxx` |
| 问题 | 用 query param 而非路径参数，不够 RESTful |
| 建议 | 考虑改为 `GET /v1/memory/{character_name}/universes` |

### 发现 3: Pydantic V2 迁移告警 (101 warnings) ⚠️

| 字段 | 详情 |
|:----:|:----:|
| 来源 | `config.py` 中的 `Field(..., env="XXX")` |
| 问题 | Pydantic V2 已弃用 `env` 参数，应使用 `model_config = ConfigDict(...)` |
| 影响 | 日志中大量 warning，升级到 V3 后会崩溃 |

---

## 与之前静态 QA 的对比

| 维度 | 静态 QA (之前做的) | 功能测试 (这次新增的) |
|:----:|:--------:|:---------:|
| 工具 | flake8 + eyeball review | httpx + pytest-asyncio |
| 能发现什么 | 代码风格、未使用变量 | **端点不返回错误**、**路由路径错误** |
| 需要服务器吗 | ❌ | ✅ (ASGITransport 模拟) |
| 耗时 | ~0.5s | ~78s |
| 真的有用吗 | 有用但浅层 | **直接模拟用户操作** |

---

## 下一步建议

1. 📱 **Streamlit AppTest**: 覆盖面试系统和宇宙管理器的 UI 功能
2. 🧠 **DeepEval**: RAG 检索质量评估 (需确认是否安装)
3. 🔧 **修复**: chat/completions 输入验证 + Pydantic V2 迁移
