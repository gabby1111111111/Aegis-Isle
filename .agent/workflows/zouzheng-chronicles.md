---
description: 邹峥编年史 - AI 人生模拟视觉小说生成器（每次运行生成一批内容并追加到网站）
---

# 邹峥编年史工作流

// turbo-all

本工作流用于让 AI Agent 自动生成邹峥从出生到现在的人生编年史，每次运行生成一批天数的内容，保存为独立文件，并在 NiceGUI 网站上展示。

## 前置条件

确保 `E:\zouzheng_chronicles\` 目录存在，且已安装 nicegui：

```
pip install nicegui
```

## 核心规则

1. 角色卡位于 `E:\chromes_save\邹峥1.json`
2. 世界书位于 `E:\chromes_save\^_^邹峥.json`
3. 所有生成内容保存到 `E:\zouzheng_chronicles\archive\` 目录
4. 每个时间段保存为独立 JSON 文件，命名格式：`{year}_{month}_{day}.json`
5. 绝对不覆盖已有文件，如果文件已存在，则找到最新的那一天，**必须严格 +1 天，一天一天连续往下写，绝不允许出现时间跳跃！无论这一天多么平淡，都必须生成记录。**
6. NiceGUI 网站入口：`E:\zouzheng_chronicles\app.py`，端口 8080
7. **额度保护机制：** 每次执行前，询问用户这次想挂机多久（比如：“帮我跑午休的2个小时”或“跑40天”）。Agent 在执行过程中计算时间，到达预定时间/天数后，主动停止任务并生成进度日记，防止悄无声息地跑空额度。

## 工作步骤

### 1. 读取角色卡和世界书

用 `view_file` 读取：
- `E:\chromes_save\邹峥1.json` — 获取角色性格、背景、习惯
- `E:\chromes_save\^_^邹峥.json` — 获取世界观（申都、东陆大学、静安府邸、配角信息）

从中提取以下关键设定：
- 邹峥基本信息（31岁，刑法学教授，185cm）
- 世界观（申都/沧澜江/旧申府/沧东新区/东陆大学）
- 住所细节（静安府邸顶层复式每个房间）
- 每日作息模式（但允许自由变化）
- 性格核心（温和真诚、冷心冷肺、随心而动）
- 配角（江越、季朗、许晏清，占比 ≤5%）
- 写作禁忌（禁用词列表、去油腻化、去人机）

### 2. 检查进度

扫描 `E:\zouzheng_chronicles\archive\` 目录，找到最后生成的日期文件。
- 如果目录为空，从出生年（1994年出生，即 1994_01_01.json 前后）开始
- 如果有已生成文件，找到最后日期，**下一篇必须是最后日期的明天（+1天），严禁任何时间跳跃。如果最后一天是 1994_10_15，下一篇必须是 1994_10_16。**

### 3. 搭建/更新 NiceGUI 网站

如果 `E:\zouzheng_chronicles\app.py` 不存在，创建网站骨架：

```python
from nicegui import ui
from pathlib import Path
import json
import os

ARCHIVE_DIR = Path("E:/zouzheng_chronicles/archive")
IMAGE_DIR = Path("E:/zouzheng_chronicles/images")

def load_entries():
    """从 archive 目录加载所有日期条目"""
    entries = []
    if ARCHIVE_DIR.exists():
        for f in sorted(ARCHIVE_DIR.glob("*.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                entries.append(data)
            except:
                pass
    return entries

def group_by_era(entries):
    """按人生阶段分组"""
    eras = {
        "出生与幼年 (1994-2000)": [],
        "童年 (2001-2006)": [],
        "少年 (2007-2012)": [],
        "青年 (2013-2018)": [],
        "学者之路 (2019-2024)": [],
        "当下 (2025-)": [],
    }
    for e in entries:
        year = int(e.get("year", 1994))
        if year <= 2000:
            eras["出生与幼年 (1994-2000)"].append(e)
        elif year <= 2006:
            eras["童年 (2001-2006)"].append(e)
        elif year <= 2012:
            eras["少年 (2007-2012)"].append(e)
        elif year <= 2018:
            eras["青年 (2013-2018)"].append(e)
        elif year <= 2024:
            eras["学者之路 (2019-2024)"].append(e)
        else:
            eras["当下 (2025-)"].append(e)
    return eras

@ui.page("/")
def index():
    # 全局样式
    ui.add_head_html('''
        <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Noto+Sans+SC:wght@300;400;700&display=swap" rel="stylesheet">
        <style>
            body { font-family: 'Noto Sans SC', sans-serif; }
            .gold-title { font-family: 'Playfair Display', serif; color: #C9A96E; }
            .entry-card { 
                background: rgba(20,20,20,0.85); 
                backdrop-filter: blur(12px); 
                border: 1px solid rgba(201,169,110,0.2);
                border-radius: 16px; 
                padding: 24px; 
                margin-bottom: 16px;
                transition: all 0.3s ease;
            }
            .entry-card:hover { 
                border-color: rgba(201,169,110,0.5);
                box-shadow: 0 4px 20px rgba(201,169,110,0.1);
            }
            .entry-text { color: #E0E0E0; line-height: 1.8; }
            .entry-meta { color: #C9A96E; font-size: 0.85em; }
            .entry-important { border-left: 3px solid #C9A96E; }
            .nav-era { color: #C9A96E; font-weight: 700; }
            .nav-date { color: #999; font-size: 0.85em; }
            .sidebar-bg { background: rgba(10,10,10,0.95); }
        </style>
    ''')
    ui.query('body').style('background-color: #0a0a0a; color: #E0E0E0;')

    entries = load_entries()
    eras = group_by_era(entries)

    # 左侧导航
    with ui.left_drawer().classes('sidebar-bg').style('width: 280px; padding: 20px;'):
        ui.label('邹 峥 编 年 史').classes('gold-title text-xl mb-2')
        ui.label('温和的精确主义').style('color: #888; font-size: 0.8em; margin-bottom: 20px;')
        ui.separator().style('background: rgba(201,169,110,0.3);')

        total = len(entries)
        ui.label(f'已记录 {total} 天').style('color: #666; font-size: 0.75em; margin: 10px 0;')

        for era_name, era_entries in eras.items():
            if era_entries:
                with ui.expansion(f'{era_name} ({len(era_entries)}篇)').classes('nav-era').style('color: #C9A96E;'):
                    for entry in era_entries[-20:]:  # 每个阶段最多显示最近20条
                        date_str = f"{entry.get('year')}-{entry.get('month','?'):>02}-{entry.get('day','?'):>02}"
                        title = entry.get('title', '日常')
                        ui.label(f'{date_str} {title}').classes('nav-date cursor-pointer')

    # 主内容区
    with ui.column().classes('w-full max-w-4xl mx-auto p-6'):
        ui.label('邹峥编年史').classes('gold-title text-4xl mb-1')
        ui.label('—— 从出生到此刻，一个精确主义者的每一天').style('color: #888; margin-bottom: 30px;')

        if not entries:
            with ui.card().classes('entry-card'):
                ui.label('尚无条目。运行工作流开始生成邹峥的人生...').classes('entry-text')
        else:
            # 显示最新的条目在最前面
            for entry in reversed(entries[-50:]):  # 显示最近50条
                important = entry.get('important', False)
                card_class = 'entry-card entry-important' if important else 'entry-card'
                with ui.card().classes(card_class):
                    date_str = f"{entry.get('year')}年{entry.get('month')}月{entry.get('day')}日"
                    age = int(entry.get('year', 1994)) - 1994
                    age_str = f"（{age}岁）" if age > 0 else "（出生）"

                    with ui.row().classes('w-full justify-between items-center mb-2'):
                        ui.label(f"📅 {date_str} {age_str}").classes('entry-meta')
                        if entry.get('location'):
                            ui.label(f"📍 {entry['location']}").classes('entry-meta')

                    if entry.get('title'):
                        ui.label(entry['title']).classes('gold-title text-lg mb-2')

                    ui.markdown(entry.get('content', '')).classes('entry-text')

                    if entry.get('image'):
                        img_path = entry['image']
                        if os.path.exists(img_path):
                            ui.image(img_path).classes('w-full rounded-xl mt-3').style('max-height: 300px; object-fit: cover;')

                    if entry.get('real_world_event'):
                        ui.separator().style('background: rgba(201,169,110,0.2); margin: 12px 0;')
                        ui.label(f"🌍 {entry['real_world_event']}").style('color: #888; font-size: 0.8em; font-style: italic;')

# 提供 images 目录的静态文件服务
app.add_static_files('/images', str(IMAGE_DIR))

if __name__ in {"__main__", "__mp_main__"}:
    ui.run(title="邹峥编年史", port=8080, dark=True, tailwind=True)
```

### 4. 生成内容

对于每一天需要生成的内容，执行以下流程：

#### 4a. 判断内容厚薄
**[最高指令] 日期必须是上一篇记录的 +1 天，绝对不能跳日！**
- **普通日**：写 200-300 字简述，按邹峥性格自由安排他这天做什么。哪怕是做同样的事，也要换个观察视角或找点细节。
- **重要日**（人生转折点）：写 800-1500 字详写（如入学、得奖、认识江越等）。

#### 4b. 搜索真实新闻
用 `search_web` 搜索该年份的真实世界大事（科技/法律/文化新闻），融入叙事背景。

#### 4c. 写叙事
- 第三人称全知视角
- 文学风格："温和的精确主义"，精确、克制、有温度
- 遵守写作禁忌（禁用词列表、不模糊不油腻）
- 说话像普通人，不套专业术语到日常对白
- 他的每天要自由——按性格"随心而动"，不写成机器人日程

#### 4d. 画插图（重要日）
用 `generate_image` 为重要场景画插图：
- 画风景/建筑/食物/书桌/乐器/棋盘/唱片/咖啡等静物场景
- 不画人脸
- 保存到 `E:\zouzheng_chronicles\images\{year}_{month}_{day}.png`

#### 4e. 保存 JSON 文件
每天的内容保存为 `E:\zouzheng_chronicles\archive\{year}_{month:02d}_{day:02d}.json`：

```json
{
    "year": 1994,
    "month": 10,
    "day": 15,
    "title": "书房的晨光",
    "location": "申都·旧申府·家中",
    "important": true,
    "content": "叙事正文...",
    "image": "E:/zouzheng_chronicles/images/1994_10_15.png",
    "real_world_event": "1994年，南非首次全民大选，曼德拉当选总统",
    "tags": ["童年", "书房", "父亲"]
}
```

### 5. 每批次结束后

- 检查已生成的总天数
- 记录进度到 `E:\zouzheng_chronicles\progress.json`：
```json
{
    "last_date": "2005_03_15",
    "total_entries": 142,
    "last_updated": "2026-03-12T16:30:00"
}
```
- 网站 app.py 会自动读取 archive 目录的所有 JSON 显示，无需手动更新

### 6. 启动网站查看

```
python E:\zouzheng_chronicles\app.py
```
然后访问 http://localhost:8080 查看编年史。

## 邹峥核心设定速查

### 基本信息
- 姓名：邹峥，男，31岁，185cm
- 东陆大学刑法学正教授
- 外貌：乌黑短发，深棕色眼，细框眼镜，嘴角浅笑
- 性格：温和真诚+冷心冷肺+情绪极稳+随心而动
- 思考时擦眼镜，对话时专注凝视

### 世界观
- 申都（架空超级都市）：沧澜江分隔旧申府(历史人文)和沧东新区(科技金融)
- 东陆大学：旧申府黄金地段，红砖+常春藤钟楼，法学院大楼（上世纪三十年代花岗岩建筑）
- 静安府邸：顶层复式，深胡桃木+落地窗+丹麦音响+整面墙书架+智能灯光

### 每日作息（但允许自由变化）
- 6:00 起床→拉伸→沧澜江边晨跑5km
- 7:00 淋浴→做早餐（黑咖啡+全麦面包+水波蛋）
- 上午：授课/学术研究
- 下午：书房读文献/处理邮件
- 晚间：听黑胶唱片/阅读/健身1h
- 23:00前就寝（看书+冥想10min）

### 配角（占比≤5%，每10-15天出场一次）
- 江越：刑辩律师，大学学弟兼至交，雅痞不羁，右脸酒窝
- 季朗：检察官学弟，正直锐利，指关节敲桌面
- 许晏清：研究生，瘦高文弱，桃花眼琥珀瞳色，右眼角泪痣
- 三人与邹峥绝无浪漫关系

### 写作规范
- 风格：精确、克制、有温度
- 禁用词：一丝/一抹/仿佛/似乎/野兽/闪过/闪烁/低吼/该死的/灭顶/发白/泛白/尖叫
- 不用模糊量词，用具体物理细节
- 不用涟漪/石子打破湖面等老套意象
- 对白自然口语化
- 不写恋爱/亲密/BDSM/性相关内容
