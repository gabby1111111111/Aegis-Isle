"""
Interview Prep System - Visual Novel Style Frontend

A beautiful Otome Game-inspired UI for the gamified interview preparation system.
Features character tachie, dynamic persona switching, and elegant styling.
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
    app as interview_graph,
    InterviewState
)


# ============================================================================
# Language Configuration
# ============================================================================

TRANSLATIONS = {
    "zh": {
        "title": "✨ 面试准备学院 ✨",
        "subtitle": "通过AI驱动的角色系统掌握技能",
        "sidebar_config": "⚙️ 配置",
        "jd_section": "📄 职位描述",
        "jd_label": "输入职位描述",
        "jd_help": "粘贴职位描述以生成相关问题",
        "kb_section": "📚 知识库",
        "kb_upload": "上传学习材料",
        "kb_help": "上传文本文件以生成面试问题",
        "kb_process": "📥 处理知识库",
        "kb_success": "从知识库生成了 {} 个问题！",
        "card_section": "🎭 角色卡片",
        "card_upload": "上传角色卡片",
        "card_help": "上传SillyTavern角色卡片（JSON或PNG）",
        "card_load": "📥 加载角色",
        "card_success": "已加载角色: {} ({})",
        "card_error": "加载角色出错: {}",
        "persona_section": "🎯 角色分配",
        "persona_interviewer": "面试官（评估者）",
        "persona_interviewer_help": "谁来评估你的答案？",
        "persona_tutor": "导师（错误答案）",
        "persona_tutor_help": "答错时谁来教你？",
        "persona_mentor": "顾问（正确答案）",
        "persona_mentor_help": "答对时谁来鼓励你？",
        "progress_section": "📊 你的进度",
        "total_questions": "总题数",
        "due_review": "待复习",
        "success_rate": "正确率",
        "mastered": "已掌握",
        "start_button": "🎬 开始面试",
        "current_speaker": "👤 当前发言者",
        "interview_session": "💬 面试会话",
        "answer_key": "💡 答案要点（参考）",
        "correct": "✅ 正确！",
        "incorrect": "❌ 错误",
        "score": "得分",
        "feedback_label": "的反馈：",
        "next_question": "⏭️ 下一题",
        "your_answer": "✍️ 你的答案",
        "answer_placeholder": "在此输入你的答案...",
        "answer_help": "请回答上面的问题",
        "submit_answer": "📤 提交答案",
        "session_stats": "📈 会话统计",
        "no_questions": "没有可用的问题！请通过知识库添加问题。",
        "no_question_loaded": "未加载问题！",
        "empty_answer": "请在提交前提供答案！",
        "evaluating": "正在评估你的答案...",
        "eval_complete": "评估完成！",
        "eval_error": "评估过程中出错: {}",
        "processing_kb": "正在处理知识库...",
        "kb_error": "处理知识库出错: {}",
        "character_tachie": "🎨 角色立绘\n\n（根据角色放置角色图片）",
        "question_label": "问题 {}:",
        "difficulty": "难度",
        "category": "类别",
        "language": "🌐 语言",
    },
    "en": {
        "title": "✨ Interview Prep Academy ✨",
        "subtitle": "Master Your Skills with AI-Powered Personas",
        "sidebar_config": "⚙️ Configuration",
        "jd_section": "📄 Job Description",
        "jd_label": "Enter Job Description",
        "jd_help": "Paste the job description to generate relevant questions",
        "kb_section": "📚 Knowledge Base",
        "kb_upload": "Upload Study Material",
        "kb_help": "Upload text file to generate interview questions",
        "kb_process": "📥 Process Knowledge Base",
        "kb_success": "Generated {} questions from knowledge base!",
        "card_section": "🎭 Character Cards",
        "card_upload": "Upload Character Card",
        "card_help": "Upload SillyTavern character card (JSON or PNG)",
        "card_load": "📥 Load Character",
        "card_success": "Loaded character: {} ({})",
        "card_error": "Error loading character: {}",
        "persona_section": "🎯 Persona Assignments",
        "persona_interviewer": "Interviewer (Evaluator)",
        "persona_interviewer_help": "Who evaluates your answers?",
        "persona_tutor": "Tutor (Wrong Answer)",
        "persona_tutor_help": "Who teaches you when wrong?",
        "persona_mentor": "Mentor (Correct Answer)",
        "persona_mentor_help": "Who encourages you when correct?",
        "progress_section": "📊 Your Progress",
        "total_questions": "Total Questions",
        "due_review": "Due for Review",
        "success_rate": "Success Rate",
        "mastered": "Mastered",
        "start_button": "🎬 Start Interview Session",
        "current_speaker": "👤 Current Speaker",
        "interview_session": "💬 Interview Session",
        "answer_key": "💡 Answer Key (for reference)",
        "correct": "✅ Correct!",
        "incorrect": "❌ Incorrect",
        "score": "Score",
        "feedback_label": "'s Feedback:",
        "next_question": "⏭️ Next Question",
        "your_answer": "✍️ Your Answer",
        "answer_placeholder": "Type your answer here...",
        "answer_help": "Provide your answer to the question above",
        "submit_answer": "📤 Submit Answer",
        "session_stats": "📈 Session Statistics",
        "no_questions": "No questions available! Please add questions via the knowledge base.",
        "no_question_loaded": "No question loaded!",
        "empty_answer": "Please provide an answer before submitting!",
        "evaluating": "Evaluating your answer...",
        "eval_complete": "Evaluation complete!",
        "eval_error": "Error during evaluation: {}",
        "processing_kb": "Processing knowledge base...",
        "kb_error": "Error processing knowledge base: {}",
        "character_tachie": "🎨 Character Tachie\n\n(Place character images here based on persona)",
        "question_label": "Question {}:",
        "difficulty": "Difficulty",
        "category": "Category",
        "language": "🌐 Language",
    }
}

def t(key: str) -> str:
    """Get translated text based on current language."""
    lang = st.session_state.get("language", "zh")
    return TRANSLATIONS.get(lang, TRANSLATIONS["zh"]).get(key, key)


# ============================================================================
# Configuration & Styling
# ============================================================================

def load_custom_css():
    """Load custom CSS for Visual Novel styling."""
    css = """
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600;700&family=Crimson+Text:ital,wght@0,400;0,600;1,400&display=swap');

    /* Main Theme */
    :root {
        --primary-pink: #ff6b9d;
        --secondary-purple: #8b5cf6;
        --dark-bg: #1a1625;
        --card-bg: #2d1b3d;
        --text-light: #f0e6ff;
        --accent-gold: #ffd700;
        --success-green: #4ade80;
        --error-red: #ef4444;
    }

    /* Main Container */
    .main {
        background: linear-gradient(135deg, #1a1625 0%, #2d1b3d 100%);
        color: var(--text-light);
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2d1b3d 0%, #1a1625 100%);
        border-right: 2px solid var(--primary-pink);
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        font-family: 'Cinzel', serif;
        color: var(--primary-pink);
        text-shadow: 0 0 10px rgba(255, 107, 157, 0.5);
    }

    /* Character Card */
    .character-card {
        background: linear-gradient(135deg, var(--card-bg) 0%, #3d2b4d 100%);
        border: 3px solid var(--primary-pink);
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 8px 32px rgba(255, 107, 157, 0.3);
        margin: 15px 0;
        animation: fadeIn 0.5s ease-in;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Character Name Tag */
    .character-name {
        font-family: 'Cinzel', serif;
        font-size: 24px;
        font-weight: 700;
        color: var(--accent-gold);
        text-shadow: 0 0 15px rgba(255, 215, 0, 0.5);
        margin-bottom: 10px;
        padding: 10px 20px;
        background: rgba(255, 215, 0, 0.1);
        border-left: 5px solid var(--accent-gold);
        border-radius: 10px;
    }

    /* Dialogue Box */
    .dialogue-box {
        font-family: 'Crimson Text', serif;
        font-size: 18px;
        line-height: 1.8;
        background: rgba(45, 27, 61, 0.95);
        border: 2px solid var(--secondary-purple);
        border-radius: 15px;
        padding: 20px 25px;
        margin: 10px 0;
        box-shadow: 0 4px 20px rgba(139, 92, 246, 0.3);
        position: relative;
    }

    .dialogue-box::before {
        content: '';
        position: absolute;
        top: -10px;
        left: 30px;
        width: 0;
        height: 0;
        border-left: 10px solid transparent;
        border-right: 10px solid transparent;
        border-bottom: 10px solid var(--secondary-purple);
    }

    /* Question Box - Sukuna Style */
    .question-box {
        background: linear-gradient(135deg, #4a0e0e 0%, #8b0000 100%);
        border: 3px solid #dc143c;
        border-radius: 15px;
        padding: 25px;
        margin: 20px 0;
        box-shadow: 0 8px 32px rgba(220, 20, 60, 0.4);
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0%, 100% { box-shadow: 0 8px 32px rgba(220, 20, 60, 0.4); }
        50% { box-shadow: 0 8px 32px rgba(220, 20, 60, 0.7); }
    }

    .question-text {
        font-family: 'Crimson Text', serif;
        font-size: 20px;
        font-weight: 600;
        color: #ffe4e4;
        line-height: 1.6;
    }

    /* Evaluation Box */
    .evaluation-box {
        background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%);
        border: 2px solid #ef4444;
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 4px 20px rgba(239, 68, 68, 0.3);
    }

    .evaluation-correct {
        background: linear-gradient(135deg, #14532d 0%, #166534 100%);
        border: 2px solid var(--success-green);
        box-shadow: 0 4px 20px rgba(74, 222, 128, 0.3);
    }

    /* Tutor Box - Gojo Style */
    .tutor-box {
        background: linear-gradient(135deg, #0c4a6e 0%, #0369a1 100%);
        border: 3px solid #38bdf8;
        border-radius: 15px;
        padding: 25px;
        margin: 20px 0;
        box-shadow: 0 8px 32px rgba(56, 189, 248, 0.4);
    }

    /* Mentor Box - Nanami Style */
    .mentor-box {
        background: linear-gradient(135deg, #365314 0%, #4d7c0f 100%);
        border: 3px solid #84cc16;
        border-radius: 15px;
        padding: 25px;
        margin: 20px 0;
        box-shadow: 0 8px 32px rgba(132, 204, 22, 0.4);
    }

    /* Buttons */
    .stButton > button {
        font-family: 'Cinzel', serif;
        font-size: 18px;
        font-weight: 600;
        background: linear-gradient(135deg, var(--primary-pink) 0%, var(--secondary-purple) 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 12px 30px;
        box-shadow: 0 4px 15px rgba(255, 107, 157, 0.4);
        transition: all 0.3s ease;
        cursor: pointer;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 107, 157, 0.6);
    }

    /* Stats Display */
    .stat-card {
        background: rgba(139, 92, 246, 0.1);
        border: 2px solid var(--secondary-purple);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        text-align: center;
    }

    .stat-value {
        font-size: 32px;
        font-weight: 700;
        color: var(--accent-gold);
        text-shadow: 0 0 10px rgba(255, 215, 0, 0.5);
    }

    .stat-label {
        font-size: 14px;
        color: var(--text-light);
        opacity: 0.8;
    }

    /* Input Box */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: rgba(45, 27, 61, 0.8);
        border: 2px solid var(--secondary-purple);
        border-radius: 10px;
        color: var(--text-light);
        font-family: 'Crimson Text', serif;
        font-size: 16px;
        padding: 12px;
    }

    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: var(--primary-pink);
        box-shadow: 0 0 15px rgba(255, 107, 157, 0.3);
    }

    /* Progress Bar */
    .progress-container {
        background: rgba(45, 27, 61, 0.6);
        border-radius: 20px;
        padding: 5px;
        margin: 10px 0;
    }

    .progress-bar {
        background: linear-gradient(90deg, var(--primary-pink) 0%, var(--secondary-purple) 100%);
        height: 20px;
        border-radius: 15px;
        transition: width 0.5s ease;
    }

    /* Title */
    .main-title {
        font-family: 'Cinzel', serif;
        font-size: 48px;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, var(--primary-pink) 0%, var(--secondary-purple) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(255, 107, 157, 0.5);
        margin: 20px 0;
        animation: glow 2s infinite;
    }

    @keyframes glow {
        0%, 100% { filter: drop-shadow(0 0 10px rgba(255, 107, 157, 0.5)); }
        50% { filter: drop-shadow(0 0 20px rgba(139, 92, 246, 0.8)); }
    }

    /* Character Image Container */
    .character-tachie {
        border: 5px solid var(--primary-pink);
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(255, 107, 157, 0.4);
        overflow: hidden;
        animation: fadeIn 0.8s ease-in;
    }

    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def display_character_name(name: str, role: str):
    """Display character name tag."""
    st.markdown(f"""
    <div class="character-name">
        {name} - {role}
    </div>
    """, unsafe_allow_html=True)


def display_dialogue(text: str, box_class: str = "dialogue-box"):
    """Display dialogue in styled box."""
    st.markdown(f"""
    <div class="{box_class}">
        {text}
    </div>
    """, unsafe_allow_html=True)


def display_question(question: Question):
    """Display question in Sukuna-styled box."""
    st.markdown(f"""
    <div class="question-box">
        <div class="question-text">
            <strong>{t("question_label").format(question.id[-8:])}:</strong><br>
            {question.content}
        </div>
        <div style="margin-top: 15px; font-size: 14px; opacity: 0.8;">
            {t("difficulty")}: {'⭐' * question.difficulty} ({question.difficulty}/5) |
            {t("category")}: {question.category}
        </div>
    </div>
    """, unsafe_allow_html=True)


def display_stats(stats: Dict[str, Any]):
    """Display progress statistics."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{stats['total_questions']}</div>
            <div class="stat-label">{t("total_questions")}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{stats['due_for_review']}</div>
            <div class="stat-label">{t("due_review")}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        success_rate = stats['overall_success_rate'] * 100
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{success_rate:.1f}%</div>
            <div class="stat-label">{t("success_rate")}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        mastered = stats['questions_by_box'].get('box_5', 0)
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{mastered}</div>
            <div class="stat-label">{t("mastered")}</div>
        </div>
        """, unsafe_allow_html=True)


# ============================================================================
# Session State Initialization
# ============================================================================

def initialize_session_state():
    """Initialize Streamlit session state."""
    if "language" not in st.session_state:
        st.session_state.language = "zh"  # 默认中文

    if "knowledge_engine" not in st.session_state:
        st.session_state.knowledge_engine = KnowledgeEngine()

    if "persona_manager" not in st.session_state:
        st.session_state.persona_manager = PersonaManager()

    if "current_question" not in st.session_state:
        st.session_state.current_question = None

    if "user_answer" not in st.session_state:
        st.session_state.user_answer = ""

    if "evaluation_result" not in st.session_state:
        st.session_state.evaluation_result = None

    if "feedback" not in st.session_state:
        st.session_state.feedback = None

    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    if "jd_context" not in st.session_state:
        st.session_state.jd_context = ""

    if "interviewer_persona" not in st.session_state:
        st.session_state.interviewer_persona = "sukuna"

    if "tutor_persona" not in st.session_state:
        st.session_state.tutor_persona = "gojo"

    if "mentor_persona" not in st.session_state:
        st.session_state.mentor_persona = "nanami"

    if "current_speaker" not in st.session_state:
        st.session_state.current_speaker = "sukuna"


# ============================================================================
# Core Functions
# ============================================================================

def load_next_question():
    """Load next question from knowledge engine."""
    question = st.session_state.knowledge_engine.get_next_question()

    if question:
        st.session_state.current_question = question
        st.session_state.user_answer = ""
        st.session_state.evaluation_result = None
        st.session_state.feedback = None
        st.session_state.current_speaker = st.session_state.interviewer_persona
        return True
    else:
        st.warning(t("no_questions"))
        return False


async def process_answer():
    """Process user's answer through the interview graph."""
    if not st.session_state.current_question:
        st.error(t("no_question_loaded"))
        return

    if not st.session_state.user_answer.strip():
        st.warning(t("empty_answer"))
        return

    with st.spinner(t("evaluating")):
        try:
            # Prepare state for interview graph
            initial_state: InterviewState = {
                "question": st.session_state.current_question,
                "user_answer": st.session_state.user_answer,
                "jd_context": st.session_state.jd_context,
                "evaluation": {},
                "history": st.session_state.conversation_history,
                "feedback": "",
                "persona_mode": st.session_state.interviewer_persona,
                "next_action": None
            }

            # Run through interview graph
            result = await interview_graph.ainvoke(initial_state)

            # Store results
            st.session_state.evaluation_result = result["evaluation"]
            st.session_state.feedback = result["feedback"]
            st.session_state.conversation_history = result["history"]

            # Update progress in knowledge engine
            is_correct = result["evaluation"].get("is_correct", False)
            st.session_state.knowledge_engine.update_progress(
                st.session_state.current_question.id,
                is_correct=is_correct
            )

            # Update current speaker for character display
            if is_correct:
                st.session_state.current_speaker = st.session_state.mentor_persona
            else:
                st.session_state.current_speaker = st.session_state.tutor_persona

            st.success(t("eval_complete"))

        except Exception as e:
            st.error(t("eval_error").format(e))
            import traceback
            st.code(traceback.format_exc())


async def ingest_knowledge_base(uploaded_file, jd_context: str):
    """Ingest knowledge base file."""
    if uploaded_file:
        with st.spinner(t("processing_kb")):
            try:
                # Read file content
                text_content = uploaded_file.read().decode("utf-8")

                # Generate questions
                questions = await st.session_state.knowledge_engine.ingest_data(
                    text=text_content,
                    jd_context=jd_context
                )

                st.success(t("kb_success").format(len(questions)))
                return True

            except Exception as e:
                st.error(t("kb_error").format(e))
                import traceback
                st.code(traceback.format_exc())
                return False
    return False


# ============================================================================
# Main App
# ============================================================================

def main():
    """Main Streamlit app."""
    # Page config
    st.set_page_config(
        page_title="Interview Prep - Visual Novel",
        page_icon="💖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Load custom CSS
    load_custom_css()

    # Initialize session state
    initialize_session_state()

    # Title
    st.markdown(f'<h1 class="main-title">{t("title")}</h1>', unsafe_allow_html=True)
    st.markdown(f'<p style="text-align: center; font-family: Cinzel, serif; font-size: 18px; color: #ff6b9d; margin-bottom: 30px;">{t("subtitle")}</p>', unsafe_allow_html=True)

    # ========================================================================
    # Sidebar - Configuration Area
    # ========================================================================
    with st.sidebar:
        # Language Selector at the top
        st.markdown(f"## {t('language')}")
        lang_option = st.selectbox(
            "",
            options=["中文", "English"],
            index=0 if st.session_state.language == "zh" else 1,
            key="lang_selector"
        )
        if lang_option == "中文" and st.session_state.language != "zh":
            st.session_state.language = "zh"
            st.rerun()
        elif lang_option == "English" and st.session_state.language != "en":
            st.session_state.language = "en"
            st.rerun()

        st.markdown("---")
        st.markdown(f"## {t('sidebar_config')}")

        # Job Description Upload
        st.markdown(f"### {t('jd_section')}")
        jd_text = st.text_area(
            t("jd_label"),
            value=st.session_state.jd_context,
            height=150,
            help=t("jd_help")
        )
        if jd_text != st.session_state.jd_context:
            st.session_state.jd_context = jd_text

        st.markdown("---")

        # Knowledge Base Upload
        st.markdown(f"### {t('kb_section')}")
        kb_file = st.file_uploader(
            t("kb_upload"),
            type=["txt", "md"],
            help=t("kb_help")
        )

        if kb_file and st.button(t("kb_process")):
            success = asyncio.run(ingest_knowledge_base(kb_file, jd_text))
            if success:
                st.balloons()

        st.markdown("---")

        # Character Card Upload
        st.markdown(f"### {t('card_section')}")
        card_file = st.file_uploader(
            t("card_upload"),
            type=["json", "png"],
            help=t("card_help")
        )

        if card_file and st.button(t("card_load")):
            try:
                # Save uploaded file temporarily
                temp_path = Path(f"temp_{card_file.name}")
                with open(temp_path, "wb") as f:
                    f.write(card_file.read())

                # Load character
                persona = st.session_state.persona_manager.load_card(temp_path)
                st.success(t("card_success").format(persona.name, persona.role))

                # Clean up
                temp_path.unlink()

            except Exception as e:
                st.error(t("card_error").format(e))

        st.markdown("---")

        # Persona Slot Selection
        st.markdown(f"### {t('persona_section')}")

        available_personas = st.session_state.persona_manager.list_personas()

        st.session_state.interviewer_persona = st.selectbox(
            t("persona_interviewer"),
            options=[p.lower().replace(" ", "_") for p in available_personas],
            index=0,
            help=t("persona_interviewer_help")
        )

        st.session_state.tutor_persona = st.selectbox(
            t("persona_tutor"),
            options=[p.lower().replace(" ", "_") for p in available_personas],
            index=1 if len(available_personas) > 1 else 0,
            help=t("persona_tutor_help")
        )

        st.session_state.mentor_persona = st.selectbox(
            t("persona_mentor"),
            options=[p.lower().replace(" ", "_") for p in available_personas],
            index=2 if len(available_personas) > 2 else 0,
            help=t("persona_mentor_help")
        )

        st.markdown("---")

        # Stats
        st.markdown(f"### {t('progress_section')}")
        stats = st.session_state.knowledge_engine.get_progress_statistics()

        st.metric(t("total_questions"), stats['total_questions'])
        st.metric(t("due_review"), stats['due_for_review'])
        success_rate = stats['overall_success_rate'] * 100
        st.metric(t("success_rate"), f"{success_rate:.1f}%")

    # ========================================================================
    # Main Area
    # ========================================================================

    # Load initial question if none exists
    if st.session_state.current_question is None:
        if st.button(t("start_button")):
            load_next_question()

    # Display current session
    if st.session_state.current_question:
        # Create two columns for character and dialogue
        char_col, dialogue_col = st.columns([1, 2])

        with char_col:
            st.markdown(f"### {t('current_speaker')}")

            # Get current persona
            current_persona = st.session_state.persona_manager.get_persona(
                st.session_state.current_speaker
            )

            if current_persona:
                # Display character info
                display_character_name(current_persona.name, current_persona.role)

                st.markdown(f"""
                <div class="character-card">
                    <p style="font-family: 'Crimson Text', serif; font-size: 16px; line-height: 1.6;">
                        {current_persona.description[:200]}...
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # Placeholder for character image (tachie)
                st.info(t("character_tachie"))

        with dialogue_col:
            st.markdown(f"### {t('interview_session')}")

            # Display question
            if st.session_state.evaluation_result is None:
                display_question(st.session_state.current_question)

                # Show expected answer hint (optional)
                with st.expander(t("answer_key")):
                    st.info(st.session_state.current_question.answer_key)

            # Display evaluation if exists
            if st.session_state.evaluation_result:
                is_correct = st.session_state.evaluation_result.get("is_correct", False)
                score = st.session_state.evaluation_result.get("score", 0)
                comment = st.session_state.evaluation_result.get("comment", "")

                # Evaluation box
                eval_class = "evaluation-box evaluation-correct" if is_correct else "evaluation-box"

                st.markdown(f"""
                <div class="{eval_class}">
                    <h3 style="margin-top: 0; color: white;">
                        {t('correct') if is_correct else t('incorrect')}
                    </h3>
                    <p style="font-size: 18px; margin: 10px 0;">
                        <strong>{t('score')}:</strong> {score}/10
                    </p>
                    <p style="font-size: 16px; line-height: 1.6;">
                        {comment}
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # Feedback box
                if st.session_state.feedback:
                    feedback_class = "mentor-box" if is_correct else "tutor-box"
                    feedback_persona = st.session_state.mentor_persona if is_correct else st.session_state.tutor_persona

                    persona_obj = st.session_state.persona_manager.get_persona(feedback_persona)
                    persona_name = persona_obj.name if persona_obj else feedback_persona

                    st.markdown(f"""
                    <div class="{feedback_class}">
                        <h3 style="margin-top: 0; color: white;">
                            {persona_name}{t('feedback_label')}
                        </h3>
                        <p style="font-size: 16px; line-height: 1.8;">
                            {st.session_state.feedback}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                # Next question button
                st.markdown("<br>", unsafe_allow_html=True)
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button(t("next_question"), use_container_width=True):
                        load_next_question()
                        st.rerun()

    # ========================================================================
    # Bottom - User Input Area
    # ========================================================================

    if st.session_state.current_question and st.session_state.evaluation_result is None:
        st.markdown("---")
        st.markdown(f"### {t('your_answer')}")

        # Text input for answer
        user_answer = st.text_area(
            t("answer_placeholder"),
            value=st.session_state.user_answer,
            height=150,
            key="answer_input",
            help=t("answer_help")
        )

        # Update session state
        st.session_state.user_answer = user_answer

        # Submit button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button(t("submit_answer"), use_container_width=True, type="primary"):
                asyncio.run(process_answer())
                st.rerun()

    # ========================================================================
    # Progress Display
    # ========================================================================

    st.markdown("---")
    st.markdown(f"### {t('session_stats')}")

    stats = st.session_state.knowledge_engine.get_progress_statistics()
    display_stats(stats)


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    main()
