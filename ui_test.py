import streamlit as st

st.set_page_config(layout="wide", page_title="Aegis-Isle UI Final")

# ==========================================
# 1. 定义 CSS (注意：这里不要加 f 前缀)
# ==========================================
CSS_STYLE = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700&family=JetBrains+Mono:wght@400&family=Noto+Sans+SC:wght@300;400;700&display=swap');

    /* 全局设置 */
    .stApp {
        background-color: #05040a;
        color: #e0e0e0;
        font-family: 'Noto Sans SC', sans-serif;
    }
    
    /* 隐藏 Streamlit 默认元素 */
    .block-container { padding: 0 !important; max-width: 100% !important; }
    header, footer { display: none !important; }
    [data-testid="stVerticalBlock"] { gap: 0; }

    /* 左侧舞台 */
    .stage-container {
        position: relative;
        height: 100vh;
        overflow: hidden;
        display: flex;
        flex-direction: column;
        justify-content: flex-end;
        padding: 40px;
        background: #000;
    }

    .char-bg {
        position: absolute;
        top: 0; left: 0; width: 100%; height: 100%;
        background-image: url('https://i.pinimg.com/736x/2b/3a/0d/2b3a0d58700994d52140411760619623.jpg');
        background-size: cover;
        background-position: center top; 
        mask-image: linear-gradient(to bottom, rgba(0,0,0,1) 50%, rgba(0,0,0,0.3) 100%);
        -webkit-mask-image: linear-gradient(to bottom, rgba(0,0,0,1) 50%, rgba(0,0,0,0.3) 100%);
        z-index: 0;
    }

    /* 题目卡片 */
    .question-card {
        position: relative;
        z-index: 10;
        max-width: 900px;
        margin: 0 auto 100px auto;
        background: rgba(16, 12, 20, 0.85);
        border: 1px solid rgba(255, 215, 0, 0.3);
        border-radius: 12px;
        backdrop-filter: blur(10px);
        box-shadow: 0 0 40px rgba(0,0,0,0.8);
        overflow: hidden;
    }

    .q-header {
        background: linear-gradient(90deg, rgba(50, 30, 10, 0.9), transparent);
        border-left: 4px solid #ffd700;
        padding: 15px 25px;
        font-family: 'Cinzel', serif;
        color: #ffd700;
        font-size: 1.2rem;
        border-bottom: 1px solid rgba(255, 215, 0, 0.1);
    }

    .q-body {
        padding: 25px;
        font-family: 'Noto Sans SC', serif;
        font-size: 1.1rem;
        line-height: 1.6;
        color: #eee;
    }

    .q-tech {
        background: rgba(0, 20, 30, 0.5);
        border-left: 3px solid #00f3ff;
        margin: 15px 0;
        padding: 15px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.95rem;
        color: #b3e5fc;
    }

    .q-footer {
        padding: 10px 25px;
        background: rgba(0,0,0,0.4);
        text-align: right;
        font-style: italic;
        color: #888;
        font-size: 0.85rem;
    }

    /* 右侧 HUD */
    .hud-panel {
        background: #0b0a12;
        height: 100vh;
        border-left: 1px solid #333;
        padding: 20px;
        display: flex;
        flex-direction: column;
        gap: 20px;
    }

    .hud-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid #3d2c5e;
        border-radius: 8px;
        padding: 15px;
    }

    .hud-title {
        color: #a55eea;
        font-family: 'Rajdhani', sans-serif;
        font-weight: bold;
        border-bottom: 1px solid #3d2c5e;
        padding-bottom: 5px;
        margin-bottom: 10px;
    }
    
    .stat-row {
        display: flex;
        justify-content: space-between;
        font-size: 0.9rem;
        margin-bottom: 5px;
        color: #ccc;
    }
</style>
"""

# ==========================================
# 2. 定义 HTML 内容 (HTML 模板)
# ==========================================
HTML_LEFT = """
<div class="stage-container">
    <div class="char-bg"></div>
    
    <div class="question-card">
        <div class="q-header">👑 黄金王座 · 神圣泰拉</div>
        <div class="q-body">
            “星炬的导航算法出现了亚空间逻辑死锁。凡人，为了防止黑色舰队迷航，告诉我，如何重构这段查询逻辑？”
            
            <div class="q-tech">
                /// MISSION OBJECTIVE ///<br>
                请解释 SQL 查询中 Index (索引) 的工作原理及优缺点。
            </div>
            
            <div style="display:grid; grid-template-columns: 1fr 1fr; gap:10px; margin-top:10px;">
                <div style="color:#69f0ae; font-size:0.9rem;">
                    <strong>⚙️ 机神低语:</strong> B+树, 二分查找, 空间换时间
                </div>
                <div style="color:#ff80ab; font-size:0.9rem;">
                    <strong>🍼 机仆速记:</strong> 就像字典的目录！查字快，但改起来慢！
                </div>
            </div>
        </div>
        <div class="q-footer">—— 帝皇金色的眼眸正注视着你，等待你的回应。</div>
    </div>
</div>
"""

HTML_RIGHT = """
<div class="hud-panel">
    <div class="hud-card">
        <div class="hud-title">👤 PROFILE</div>
        <div class="stat-row"><span>NAME</span> <span>Gabriella</span></div>
        <div class="stat-row"><span>RANK</span> <span>Neophyte</span></div>
        <div style="background:#333; height:4px; margin-top:5px;"><div style="background:#a55eea; width:40%; height:100%;"></div></div>
    </div>

    <div class="hud-card">
        <div class="hud-title">⚡ STATUS</div>
        <div class="stat-row"><span style="color:#4ade80">Sanity</span> <span>92%</span></div>
        <div class="stat-row"><span style="color:#ef5350">Corruption</span> <span>18%</span></div>
    </div>
    
    <div style="margin-top:auto; font-family:'JetBrains Mono'; font-size:0.7rem; color:#444;">
        > SYSTEM_CHECK: ONLINE<br>
        > LATENCY: 2ms
    </div>
</div>
"""

# ==========================================
# 3. 执行渲染 (注意顺序)
# ==========================================

# 注入 CSS (必须加 unsafe_allow_html=True)
st.markdown(CSS_STYLE, unsafe_allow_html=True)

# 布局
c1, c2 = st.columns([3, 1])

with c1:
    # 注入左侧 HTML
    st.markdown(HTML_LEFT, unsafe_allow_html=True)

with c2:
    # 注入右侧 HTML
    st.markdown(HTML_RIGHT, unsafe_allow_html=True)