---
name: nicegui-refactoring
description: 使用 NiceGUI 框架对现有 Streamlit 页面进行现代化、高颜值的前端重构指南 (2026年标准)
---

# 🎨 现代化前端重构指南 - NiceGUI (2026)

本 Skill 旨在指导 Agent 如何将现有陈旧的、基于 Streamlit 的数据面板风格页面，使用 **NiceGUI** 框架重构为具有高颜值、"毛玻璃"质感、圆角设计和现代交互体验的 2026 工业级 Web 应用。

## 🎯 核心目标
1. **替换 Streamlit**：用 NiceGUI 替换 Streamlit，保留纯 Python 开发体验，但解锁完整的现代网页设计能力。
2. **应用现代 UI 设计语言**：
   - 阴影层级 (`shadow-md`, `shadow-xl`) 区分空间关系
   - 圆角 (`rounded-xl`, `rounded-borders`)
   - 半透明毛玻璃质感 (`bg-white/70`, `backdrop-blur-md`)
   - 交互动效 (`transition`, `hover:scale-105`)
3. **响应式布局**：基于 Tailwind CSS 语义类，实现优秀的行列布局。

## 🧱 基础架构替换指南

### 1. 启动方式
**Streamlit (旧):**
```python
import streamlit as st
st.set_page_config(layout="wide")
# 运行: streamlit run app.py
```

**NiceGUI (新):**
```python
from nicegui import ui

@ui.page('/')
def index():
    ui.label('欢迎来到应允之地').classes('text-2xl font-bold')

# 运行: python app.py
if __name__ in {"__main__", "__mp_main__"}:
    ui.run(title="Aegis-Isle", port=8501, dark=None, tailwind=True) # dark=None 跟随系统
```

### 2. 核心布局范例 (The "Glassmorphism" Card)
在 NiceGUI 中，不要铺满整个纯白/纯黑屏幕，使用背景色 + 毛玻璃卡片是让页面变高级的秘诀。

```python
from nicegui import ui

# 设定整个页面的高级背景底色（如淡紫白 / 深空蓝）
ui.query('body').classes('bg-gradient-to-br from-indigo-50 to-blue-100')

with ui.card().classes(
    'w-full max-w-4xl mx-auto mt-10 p-8 '
    'bg-white/80 backdrop-blur-lg shadow-2xl rounded-2xl border border-white/50'
):
    ui.label('✨ Universe Manager').classes('text-3xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-indigo-600 to-purple-600')
    
    with ui.row().classes('w-full mt-6 justify-between items-center'):
        ui.input('搜索角色记忆...').props('rounded outlined dense').classes('w-2/3')
        ui.button('进行搜索', icon='search').props('rounded unelevated color="primary"')
```

## 🛠️ 组件平替对照表

| Streamlit 组件 | NiceGUI 现代平替 | 样式强化建议 (Tailwind Classes) |
|---|---|---|
| `st.title("文本")` | `ui.label("文本").classes('text-h4')` | `.classes('font-bold text-gray-800 tracking-tight')` |
| `st.write("正文")` | `ui.label("正文")` | `.classes('text-gray-600 leading-relaxed')` |
| `st.columns(2)` | `with ui.row().classes('w-full gap-4'):`<br>`  with ui.column().classes('w-1/2'):` | 使用 `.classes('max-w-md w-full gap-6')` 灵活约束 |
| `st.button("你好")` | `ui.button("你好")` | `.props('rounded unelevated').classes('shadow-md hover:shadow-lg transition-all')` |
| `st.expander("展开")` | `with ui.expansion("展开")` | `.classes('bg-gray-50 rounded-xl overflow-hidden')` |
| `st.sidebar` | `with ui.left_drawer():` | 可设置 `.classes('bg-white/90 backdrop-blur-md')` |
| `st.chat_message` | 需自定义 Row + Avatar 组合 | 圆润的气泡：`.classes('bg-blue-500 text-white rounded-2xl rounded-tl-none p-3')` |

## 🌟 实战演练：重写审核面板
如果是重写 `charlife_review_app.py`（审核面板），页面结构设计思路如下：

1. **暗黑模式护眼**：如果是夜间审核，默认设定为暗色主题。
2. **卡片流排布**：每一条待审核的内心独白，都是一个独立的 Card。
3. **按钮悬停反馈**：通过和不通过按钮给予颜色反馈（通过=绿色渐变，驳回=红灰幽灵按钮）。

```python
with ui.card().classes('w-full border-l-4 border-indigo-500 mb-4'):
    with ui.row().classes('w-full items-center justify-between'):
        ui.label('邹峥 | 极度压抑的烦躁').classes('font-bold text-lg')
        ui.label('10 分钟前').classes('text-sm text-gray-400')
    ui.separator()
    ui.markdown('> 怎么，这段时间又跑到哪里去疯了？').classes('text-base italic')
    
    with ui.row().classes('w-full justify-end mt-2 gap-2'):
        ui.button('驳回').props('outline color="negative" rounded')
        ui.button('准奏写进记忆').props('unelevated color="positive" rounded')
```

## ⚠️ 注意事项
1. **异步支持**：NiceGUI 完全支持 `async/await`，所有的按钮点击事件都可以直接绑定 `async def on_click():`，非常适合对接着 Aegis-Isle 的异步 API。
2. **状态共享**：如果在多用户环境下，避免使用全局变量，状态应绑定在 `app.storage.user` 或页面会话类中。(在本地单人工具中则可以适当放宽)。
3. **Tailwind 调试**：Tailwind classes 直接写在 `.classes('')` 里面就可以即时生效。
