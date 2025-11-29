"""
Interview Prep System - The Infinite Interview
"Project Love & Code"

A Cinematic, Otome-Game Style, Infinite Role-Play Interview System.
"""

import sys
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any
import base64
import os

import streamlit as st

# Add src to path for imports - get absolute path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
src_path = project_root / "src"

# Add to Python path if not already there
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Also add project root
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from aegis_isle.interview import (
    KnowledgeEngine,
    PersonaManager,
    Question,
    Generator
)
from aegis_isle.interview.story_manager import StoryManager


# ============================================================================
# Language Configuration
# ============================================================================

TRANSLATIONS = {
    "en": {
        "title": "✨ The Infinite Interview ✨",
        "subtitle": "A Cinematic Role-Play Experience",
        "sidebar_config": "⚙️ Configuration",
        "lang_select": "🌐 Language",
        "jd_section": "📄 Job Description",
        "jd_label": "Enter Job Description",
        "jd_help": "Paste the job description to generate relevant questions",
        "kb_section": "📚 Knowledge Base",
        "kb_upload": "Upload Study Material",
        "kb_help": "Upload text file to generate interview questions",
        "kb_process": "📥 Process Knowledge Base",
        "kb_success": "Generated {} questions from knowledge base!",
        "card_section": "🎭 Character Card",
        "card_upload": "Upload Character Card",
        "card_help": "Upload SillyTavern character card (JSON or PNG)",
        "card_load": "📥 Summon Character",
        "card_success": "Summoned: {} ({})",
        "card_error": "Error summoning character: {}",
        "start_button": "🎬 Enter the World",
        "submit_answer": "📤 Submit Response",
        "next_question": "⏭️ Next Challenge",
        "feedback_title": "Judgment",
        "correct": "✅ Correct",
        "incorrect": "❌ Incorrect",
        "partial": "⚠️ Partial",
        "score": "Score",
        "loading": "The world is shifting...",
        "no_questions": "The void is empty. (Add questions via Knowledge Base)",
        "intro_placeholder": "Type your role-play response...",
        "answer_placeholder": "Speak your answer...",
        "retry": "Retry",
        "hints_title": "💡 Hints & Analogies",
        "keywords": "Keywords:",
        "eli5": "ELI5:",
        "tech_q": "Technical Question:",
        "std_ans": "Standard Answer:",
        "config_info": "Configure your session in the sidebar, then click Start.",
        "start_session": "Start Session",
        "current_char": "Current: {}",
    },
    "zh": {
        "title": "✨ 无限面试系统 ✨",
        "subtitle": "沉浸式角色扮演面试体验",
        "sidebar_config": "⚙️ 配置",
        "lang_select": "🌐 语言 / Language",
        "jd_section": "📄 职位描述 (JD)",
        "jd_label": "输入职位描述",
        "jd_help": "粘贴职位描述以生成相关问题",
        "kb_section": "📚 知识库",
        "kb_upload": "上传学习资料",
        "kb_help": "上传文本文件以生成面试题",
        "kb_process": "📥 处理知识库",
        "kb_success": "从知识库生成了 {} 道题目！",
        "card_section": "🎭 角色卡片",
        "card_upload": "上传角色卡",
        "card_help": "上传 SillyTavern 格式的角色卡 (JSON 或 PNG)",
        "card_load": "📥 召唤角色",
        "card_success": "已召唤: {} ({})",
        "card_error": "召唤失败: {}",
        "start_button": "🎬 进入世界",
        "submit_answer": "📤 提交回答",
        "next_question": "⏭️ 下一题",
        "feedback_title": "审判",
        "correct": "✅ 正确",
        "incorrect": "❌ 错误",
        "partial": "⚠️ 不完全正确",
        "score": "得分",
        "loading": "世界正在重构...",
        "no_questions": "虚空之中空无一物。（请通过知识库添加题目）",
        "intro_placeholder": "输入你的回应...",
        "answer_placeholder": "说出你的答案...",
        "retry": "重试",
        "hints_title": "💡 提示与类比",
        "keywords": "关键词:",
        "eli5": "通俗解释:",
        "tech_q": "技术问题:",
        "std_ans": "标准答案:",
        "config_info": "请在侧边栏配置，然后点击开始。",
        "start_session": "开始会话",
        "current_char": "当前角色: {}",
    }
}

def t(key: str) -> str:
    """Get translated text."""
    lang = st.session_state.get("language", "zh")
    return TRANSLATIONS.get(lang, TRANSLATIONS["zh"]).get(key, key)


# ============================================================================
# Styling
# ============================================================================

def load_custom_css():
    """Load visual novel style with white background and large black text."""
    
    # Load background image
    bg_image_data = ""
    try:
        bg_path = Path("data/emperor_background.jpg")
        if bg_path.exists():
            with open(bg_path, "rb") as f:
                import base64
                encoded = base64.b64encode(f.read()).decode()
                bg_image_data = f"background-image: url('data:image/jpeg;base64,{encoded}');"
    except Exception as e:
        print(f"Background image not loaded: {e}")
    
    css = f"""
    <style>
    /* 视觉小说风格 - 白底黑字 + 背景图 */
    .stApp {{
        background: #f5f5f5;
        {bg_image_data}
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: #1a1a1a;
        font-size: 18px;
        line-height: 1.8;
    }}
    
    /* 白色半透明遮罩 - 移除，让背景完整显示 */
    
    /* 主内容区域 - 固定在底部的对话框 */
    .main .block-container {{
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: rgba(255, 255, 255, 0.98);
        border: 3px solid #000000;
        border-bottom: none;
        border-radius: 15px 15px 0 0;
        padding: 2.5rem;
        box-shadow: 0 -10px 30px rgba(0, 0, 0, 0.3);
        max-width: 100%;
        margin: 0;
        z-index: 1000;
        max-height: 25vh;
        overflow-y: auto;
    }}
    
    /* 侧边栏 */
    section[data-testid="stSidebar"] {{
        background: #ffffff;
        border-right: 3px solid #333333;
        box-shadow: 4px 0 10px rgba(0, 0, 0, 0.1);
    }}
    
    /* 侧边栏文字 - 超大 */
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div {{
        color: #1a1a1a !important;
        font-size: 18px !important;
        font-weight: 500 !important;
    }}
    
    /* 侧边栏标题 - 更大 */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {{
        color: #000000 !important;
        font-size: 24px !important;
        font-weight: bold !important;
        margin-bottom: 1rem !important;
    }}
    
    /* 侧边栏按钮 */
    section[data-testid="stSidebar"] .stButton > button {{
        font-size: 18px !important;
        width: 100%;
        margin-bottom: 12px;
    }}
    
    /* 标题 - 超超大 */
    h1 {{
        color: #000000;
        font-weight: bold;
        font-size: 42px;
        letter-spacing: 0.5px;
        margin-bottom: 1.5rem;
    }}
    
    h2 {{
        color: #1a1a1a;
        font-size: 32px;
        font-weight: bold;
        margin-bottom: 1rem;
    }}
    
    h3 {{
        color: #333333;
        font-size: 26px;
        font-weight: 600;
    }}
    
    /* 普通文字 - 超大号 */
    p, div, span, label {{
        font-size: 22px;
        line-height: 2.0;
    }}
    
    /* 按钮 - 简洁黑白 */
    .stButton > button {{
        background: #ffffff;
        color: #000000;
        border: 3px solid #000000;
        border-radius: 8px;
        padding: 16px 40px;
        font-weight: bold;
        font-size: 22px;
        box-shadow: 4px 4px 0px #000000;
        transition: all 0.2s;
    }}
    
    .stButton > button:hover {{
        background: #000000;
        color: #ffffff;
        transform: translate(2px, 2px);
        box-shadow: 2px 2px 0px #000000;
    }}
    
    /* 输入框 - 清晰边框 */
    .stTextInput input, .stTextArea textarea {{
        background: #ffffff;
        border: 2px solid #333333;
        border-radius: 6px;
        color: #000000;
        font-size: 22px;
        padding: 16px;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.05);
    }}
    
    .stTextInput input:focus, .stTextArea textarea:focus {{
        border-color: #000000;
        border-width: 3px;
        box-shadow: 0 0 0 2px rgba(0, 0, 0, 0.1);
    }}
    
    /* 信息框 */
    .stAlert, .stInfo {{
        background: #ffffff;
        border: 2px solid #333333;
        border-radius: 6px;
        border-left: 6px solid #000000;
        padding: 1.2rem;
        color: #1a1a1a;
        font-size: 17px;
    }}
    
    /* 选择框 */
    .stSelectbox select {{
        background: #ffffff;
        border: 2px solid #333333;
        color: #000000;
        border-radius: 6px;
        font-size: 18px;
        padding: 10px;
    }}
    
    /* 文件上传 */
    [data-testid="stFileUploader"] {{
        background: #ffffff;
        border: 3px dashed #333333;
        border-radius: 8px;
        padding: 2rem;
    }}
    
    /* 分隔线 */
    hr {{
        border: none;
        height: 3px;
        background: #000000;
        margin: 2.5rem 0;
    }}
    
    /* Streamlit标记文字超大 */
    .stMarkdown {{
        font-size: 22px;
        line-height: 2.0;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)



# ============================================================================
# Session State
# ============================================================================

def initialize_session_state():
    """Initialize Streamlit session state."""
    if "stage" not in st.session_state:
        st.session_state.stage = "config"  # config, intro, interview

    if "language" not in st.session_state:
        st.session_state.language = "zh"  # Default to Chinese

    if "knowledge_engine" not in st.session_state:
        st.session_state.knowledge_engine = KnowledgeEngine()

    if "persona_manager" not in st.session_state:
        st.session_state.persona_manager = PersonaManager()

    if "generator" not in st.session_state:
        st.session_state.generator = Generator()
    
    if "story_manager" not in st.session_state:
        st.session_state.story_manager = StoryManager()

    if "current_persona" not in st.session_state:
        # Default to Gojo if no card uploaded
        st.session_state.current_persona = st.session_state.persona_manager.get_default_persona()

    if "current_question" not in st.session_state:
        st.session_state.current_question = None

    if "polyphonic_question" not in st.session_state:
        st.session_state.polyphonic_question = None

    if "feedback_data" not in st.session_state:
        st.session_state.feedback_data = None

    if "jd_context" not in st.session_state:
        st.session_state.jd_context = ""
    
    # Track answered questions in current session to prevent immediate repetition
    if "answered_question_ids" not in st.session_state:
        st.session_state.answered_question_ids = []
    
    # Track if we should show a story node
    if "pending_story_node" not in st.session_state:
        st.session_state.pending_story_node = None


# ============================================================================
# Logic Functions
# ============================================================================

async def generate_new_question():
    """Fetch next question and generate polyphonic version."""
    # Check if we should trigger a story node first
    if st.session_state.pending_story_node:
        return  # Story node will be rendered instead
    
    # Get recently answered IDs to avoid immediate repetition
    recent_ids = st.session_state.answered_question_ids[-3:] if len(st.session_state.answered_question_ids) > 0 else []
    
    # Get next question with exclusions
    question = st.session_state.knowledge_engine.get_next_question(exclude_ids=recent_ids)
    
    if not question:
        st.warning(t("no_questions"))
        return

    st.session_state.current_question = question
    
    with st.spinner(t("loading")):
        poly_q = await st.session_state.generator.generate_question_interaction(
            st.session_state.current_persona,
            question,
            st.session_state.jd_context,
            language=st.session_state.language
        )
        st.session_state.polyphonic_question = poly_q
        st.session_state.feedback_data = None  # Reset feedback


async def submit_answer(user_answer: str):
    """Process user answer and generate feedback."""
    if not user_answer.strip():
        return

    with st.spinner(t("loading")):
        feedback = await st.session_state.generator.generate_feedback(
            st.session_state.current_persona,
            st.session_state.current_question,
            user_answer,
            {},
            language=st.session_state.language
        )
        st.session_state.feedback_data = feedback
        
        # Update progress
        is_correct = feedback.get("verdict", {}).get("status") == "correct"
        st.session_state.knowledge_engine.update_progress(
            st.session_state.current_question.id,
            is_correct
        )
        
        # Track this question as answered
        st.session_state.answered_question_ids.append(st.session_state.current_question.id)
        
        # Record answer in story manager
        st.session_state.story_manager.record_answer(is_correct)
        
        # Check if we should trigger a story node
        all_questions = st.session_state.knowledge_engine.questions.values()
        box_levels = [q.review_box for q in all_questions]
        story_trigger = st.session_state.story_manager.check_box_milestone(box_levels)
        
        if story_trigger:
            st.session_state.pending_story_node = story_trigger


async def ingest_kb(file):
    """Ingest knowledge base from uploaded file."""
    with st.spinner(t("processing")):
        text_content = file.read().decode('utf-8')
        questions = await st.session_state.knowledge_engine.ingest_data(text_content, st.session_state.jd_context)
        st.success(f"{t('success_kb')} {len(questions)} {t('questions_generated')}")


def load_emperor_test():
    """Load Emperor test scenario without file upload."""
    import json
    from pathlib import Path
    
    try:
        # Load emperor card
        card_path = Path("data/emperor_card.json")
        if not card_path.exists():
            st.error("测试数据不存在，请先运行: python create_emperor_test.py")
            return False
        
        with open(card_path, 'r', encoding='utf-8') as f:
            card_data = json.load(f)
        
        # Create Persona from card
        from aegis_isle.interview.persona_manager import Persona
        
        emperor = Persona(
            name=card_data.get("name", "人类帝皇"),
            role="人类之主，黄金王座的统治者",
            description=card_data.get("description", ""),
            personality=card_data.get("personality", ""),
            first_message=card_data.get("first_mes", ""),
            example_messages=card_data.get("mes_example", ""),
            scenario=card_data.get("scenario", ""),
            character_book=card_data.get("character_book", {}),
            avatar_path=None
        )
        
        st.session_state.current_persona = emperor
        
        # Load question database
        db_path = Path("data/emperor_test_db.json")
        if not db_path.exists():
            st.error("题库不存在，请先运行: python create_emperor_test.py")
            return False
        
        with open(db_path, 'r', encoding='utf-8') as f:
            db_data = json.load(f)
        
        # Load questions into knowledge engine
        from aegis_isle.interview.knowledge_engine import Question
        
        st.session_state.knowledge_engine.questions = {}
        for qid, qdata in db_data.get("questions", {}).items():
            question = Question(**qdata)
            st.session_state.knowledge_engine.questions[qid] = question
        
        st.session_state.knowledge_engine.save_database()
        
        return True
    
    except Exception as e:
        st.error(f"加载失败: {e}")
        return False


# ============================================================================
# UI Components
# ============================================================================

def render_sidebar():
    """Render configuration sidebar."""
    with st.sidebar:
        st.header(t("sidebar_config"))
        
        # Language Selector
        selected_lang = st.selectbox(
            t("language_selector"),
            options=["zh", "en"],
            format_func=lambda x: "中文" if x == "zh" else "English",
            index=0 if st.session_state.language == "zh" else 1
        )
        if selected_lang != st.session_state.language:
            st.session_state.language = selected_lang
            st.rerun()
        
        st.divider()
        
        # === 快速测试 ===
        st.subheader("⚡ 快速测试" if st.session_state.language == "zh" else "⚡ Quick Test")
        
        if st.button("👑 加载帝皇测试剧本" if st.session_state.language == "zh" else "👑 Load Emperor Test"):
            with st.spinner("正在召唤人类帝皇..." if st.session_state.language == "zh" else "Summoning the Emperor..."):
                success = load_emperor_test()
                if success:
                    st.success("✅ 帝皇测试剧本已加载！" if st.session_state.language == "zh" else "✅ Emperor test loaded!")
                    st.info("📋 已加载 5 道题目\n👑 角色：人类帝皇" if st.session_state.language == "zh" else "📋 5 questions loaded\n👑 Character: Emperor of Mankind")
                else:
                    st.error("❌ 加载失败，请确保测试数据存在" if st.session_state.language == "zh" else "❌ Failed to load test data")
        
        st.divider()
        
        # Job Description
        st.subheader(t("jd_section"))
        st.session_state.jd_context = st.text_area(
            t("jd_label"), 
            value=st.session_state.jd_context,
            height=100,
            help=t("jd_help")
        )

        # Knowledge Base
        st.subheader(t("kb_section"))
        kb_file = st.file_uploader(t("kb_upload"), type=["txt", "md"])
        if kb_file and st.button(t("kb_process")):
            asyncio.run(ingest_kb(kb_file))

        # Character Card
        st.subheader(t("card_section"))
        card_file = st.file_uploader(t("card_upload"), type=["json", "png"])
        if card_file and st.button(t("card_load")):
            try:
                # Save temp
                temp_path = Path(f"temp_{card_file.name}")
                with open(temp_path, "wb") as f:
                    f.write(card_file.read())
                
                # Load
                persona = st.session_state.persona_manager.load_card(temp_path)
                st.session_state.current_persona = persona
                st.success(t("card_success").format(persona.name, persona.role))
                temp_path.unlink()
            except Exception as e:
                st.error(t("card_error").format(e))
        
        st.markdown("---")
        if st.session_state.current_persona:
            st.image(st.session_state.current_persona.avatar_path or "https://placehold.co/200x200?text=Avatar", width=150)
            st.caption(t("current_char").format(st.session_state.current_persona.name))


def render_intro():
    """Render the Cinematic Intro."""
    persona = st.session_state.current_persona
    
    st.markdown(f"<h1 style='text-align: center; font-family: Cinzel'>{persona.name}</h1>", unsafe_allow_html=True)
    
    # Cinematic Text Box
    st.markdown(f"""
    <div class="cinematic-box">
        {persona.first_message}
    </div>
    """, unsafe_allow_html=True)
    
    # User Response
    user_response = st.text_input(t("intro_placeholder"), key="intro_input")
    
    if st.button(t("start_button")):
        st.session_state.stage = "interview"
        # Trigger first question generation
        asyncio.run(generate_new_question())
        st.rerun()


def render_story_node():
    """Render a story node (cinematic moment)."""
    story_trigger = st.session_state.pending_story_node
    
    if not story_trigger:
        return False
    
    # Get trigger description
    trigger_info = st.session_state.story_manager.triggers.get(story_trigger)
    
    st.markdown(f"""
    <div class="cinematic-box" style="border: 3px solid #ffd700; background: linear-gradient(180deg, #1a0a0a 0%, #000000 100%); box-shadow: 0 0 30px rgba(255, 215, 0, 0.5);">
        <h2 style="text-align: center; color: #ffd700; font-size: 28px;">🌟 {trigger_info.description if trigger_info else '剧情节点'} 🌟</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Generate story content
    success_rate = st.session_state.story_manager.get_success_rate()
    
    # Determine node type based on trigger
    if "box_1" in story_trigger:
        node_type = "node_a"
        title = "🧬 初次觉醒 - Gene Awakening"
    elif "box_3" in story_trigger:
        node_type = "node_b"  
        title = "⚔️ 晋升试炼 - Ascension Trial"
    else:
        node_type = "mastery"
        title = "👑 荣誉时刻 - Moment of Glory"
    
    # Story content placeholder
    with st.spinner("剧情生成中..."):
        story_data = asyncio.run(st.session_state.generator.generate_story_node(
            st.session_state.current_persona,
            node_type,
            success_rate,
            language=st.session_state.language
        ))
        
        story_content = story_data.get("story_content", "剧情生成中...")
        
    st.markdown(f"""
    <div class="cinematic-box" style="padding: 40px;">
        <h3 style="color: #ff6b9d; text-align: center; margin-bottom: 20px;">{title}</h3>
        <p style="font-size: 19px; line-height: 2.0; text-align: justify;">{story_content}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Button to continue
    if st.button("✨ 继续修行 ✨", key="continue_from_story"):
        st.session_state.pending_story_node = None
        asyncio.run(generate_new_question())
        st.rerun()
    
    return True


def render_interview():
    """Render the Interview Loop with Four-Layer Sandwich UI and Otome Game Elements."""
    # Check if we should show a story node first
    if st.session_state.pending_story_node:
        if render_story_node():
            return  # Story node is being displayed
    
    poly_q = st.session_state.polyphonic_question
    
    if not poly_q:
        st.error("No question generated.")
        if st.button(t("retry")):
            asyncio.run(generate_new_question())
            st.rerun()
        return

    # === 乙女游戏元素：好感度条 ===
    success_rate = st.session_state.story_manager.get_success_rate()
    affection_percentage = int(success_rate * 100)
    
    affection_label = "好感度" if st.session_state.language == "zh" else "Affection"
    st.markdown(f"""
    <div class="affection-meter">
        <div class="affection-label">♡ {affection_label}: {affection_percentage}% ♡</div>
        <div class="affection-bar">
            <div class="affection-fill" style="width: {affection_percentage}%"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # === 乙女游戏元素：角色名牌 ===
    character_name = st.session_state.current_persona.name
    st.markdown(f"""
    <div class="character-nameplate">{character_name}</div>
    """, unsafe_allow_html=True)
    
    # === 对话框装饰容器 ===
    st.markdown('<div class="dialogue-container">', unsafe_allow_html=True)

    # 1. Crown Layer (The Lore)
    lore = poly_q.get("lore_flavor", "")
    st.markdown(f"""
    <div class="crown-box">
        <span class="crown-icon">👑</span>
        {lore}
    </div>
    """, unsafe_allow_html=True)

    # 2. Core Layer (The Question)
    st.markdown(f"""
    <div class="core-box">
        <strong>⚡ {t("tech_q")}</strong><br>
        {poly_q.get('original_question', '')}
    </div>
    """, unsafe_allow_html=True)

    # 3. Hint Layer (Split View)
    tech_hint = poly_q.get("tech_hint", "N/A")
    eli5_hint = poly_q.get("eli5_hint", "N/A")
    st.markdown(f"""
    <div class="hint-container">
        <div class="tech-hint">
            {tech_hint}
        </div>
        <div class="eli5-hint">
            {eli5_hint}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 4. Gaze Layer (Encouragement)
    encouragement = poly_q.get("encouragement", "")
    if encouragement:
        st.markdown(f"""
        <div class="gaze-box">
            {encouragement}
        </div>
        """, unsafe_allow_html=True)
    
    # 关闭对话框装饰容器
    st.markdown('</div>', unsafe_allow_html=True)

    # Answer Input
    if not st.session_state.feedback_data:
        user_answer = st.text_area(t("answer_placeholder"), height=150, key="answer_input")
        if st.button(t("submit_answer")):
            asyncio.run(submit_answer(user_answer))
            st.rerun()
    else:
        # Feedback Display (Tri-Fold Judgment)
        fb = st.session_state.feedback_data
        verdict_data = fb.get("verdict", {})
        status = verdict_data.get("status", "partial")
        color = "#4ade80" if status == "correct" else "#ef4444" if status == "incorrect" else "#ffa500"
        
        st.markdown(f"""
        <div class="feedback-box" style="border-left: 5px solid {color}">
            <h3 style="color: {color}">【{status.upper()}】</h3>
            <p style="font-size: 18px; font-style: italic; color: #ffecb3;">"{verdict_data.get('comment', '')}"</p>
            <hr style="border-color: #444;">
            <p><strong>📖 {t("std_ans")}</strong> {fb.get('standard_answer', '')}</p>
            <p><strong>🍼 {t("eli5")}</strong> {fb.get('servitor_explanation', '')}</p>
        </div>
        """, unsafe_allow_html=True)

        if st.button(t("next_question")):
            asyncio.run(generate_new_question())
            st.rerun()


# ============================================================================
# Main App
# ============================================================================

def main():
    st.set_page_config(
        page_title="The Infinite Interview",
        page_icon="🔮",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Load custom CSS (simplified version)
    load_custom_css()
    
    # Render Sidebar
    render_sidebar()
    
    # Main Content Area
    if st.session_state.stage == "config":
        st.title(t("title"))
        st.subheader(t("subtitle"))
        st.info(t("config_info"))
        
        if st.button(t("start_session")):
            st.session_state.stage = "intro"
            st.rerun()
            
    elif st.session_state.stage == "intro":
        render_intro()
        
    elif st.session_state.stage == "interview":
        render_interview()

if __name__ == "__main__":
    main()  # Fixed: removed asyncio.run() since main() is not async
