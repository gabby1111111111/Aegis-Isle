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

import streamlit as st

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aegis_isle.interview import (
    KnowledgeEngine,
    PersonaManager,
    Question,
    app as interview_graph,
    InterviewState
)


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
            <strong>Question {question.id[-8:]}:</strong><br>
            {question.content}
        </div>
        <div style="margin-top: 15px; font-size: 14px; opacity: 0.8;">
            Difficulty: {'⭐' * question.difficulty} ({question.difficulty}/5) |
            Category: {question.category}
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
            <div class="stat-label">Total Questions</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{stats['due_for_review']}</div>
            <div class="stat-label">Due for Review</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        success_rate = stats['overall_success_rate'] * 100
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{success_rate:.1f}%</div>
            <div class="stat-label">Success Rate</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        mastered = stats['questions_by_box'].get('box_5', 0)
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{mastered}</div>
            <div class="stat-label">Mastered</div>
        </div>
        """, unsafe_allow_html=True)


# ============================================================================
# Session State Initialization
# ============================================================================

def initialize_session_state():
    """Initialize Streamlit session state."""
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
        st.warning("No questions available! Please add questions via the knowledge base.")
        return False


async def process_answer():
    """Process user's answer through the interview graph."""
    if not st.session_state.current_question:
        st.error("No question loaded!")
        return

    if not st.session_state.user_answer.strip():
        st.warning("Please provide an answer before submitting!")
        return

    with st.spinner("Evaluating your answer..."):
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

            st.success("Evaluation complete!")

        except Exception as e:
            st.error(f"Error during evaluation: {e}")
            import traceback
            st.code(traceback.format_exc())


async def ingest_knowledge_base(uploaded_file, jd_context: str):
    """Ingest knowledge base file."""
    if uploaded_file:
        with st.spinner("Processing knowledge base..."):
            try:
                # Read file content
                text_content = uploaded_file.read().decode("utf-8")

                # Generate questions
                questions = await st.session_state.knowledge_engine.ingest_data(
                    text=text_content,
                    jd_context=jd_context
                )

                st.success(f"Generated {len(questions)} questions from knowledge base!")
                return True

            except Exception as e:
                st.error(f"Error processing knowledge base: {e}")
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
    st.markdown('<h1 class="main-title">✨ Interview Prep Academy ✨</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-family: Cinzel, serif; font-size: 18px; color: #ff6b9d; margin-bottom: 30px;">Master Your Skills with AI-Powered Personas</p>', unsafe_allow_html=True)

    # ========================================================================
    # Sidebar - Configuration Area
    # ========================================================================
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")

        # Job Description Upload
        st.markdown("### 📄 Job Description")
        jd_text = st.text_area(
            "Enter Job Description",
            value=st.session_state.jd_context,
            height=150,
            help="Paste the job description to generate relevant questions"
        )
        if jd_text != st.session_state.jd_context:
            st.session_state.jd_context = jd_text

        st.markdown("---")

        # Knowledge Base Upload
        st.markdown("### 📚 Knowledge Base")
        kb_file = st.file_uploader(
            "Upload Study Material",
            type=["txt", "md"],
            help="Upload text file to generate interview questions"
        )

        if kb_file and st.button("📥 Process Knowledge Base"):
            success = asyncio.run(ingest_knowledge_base(kb_file, jd_text))
            if success:
                st.balloons()

        st.markdown("---")

        # Character Card Upload
        st.markdown("### 🎭 Character Cards")
        card_file = st.file_uploader(
            "Upload Character Card",
            type=["json", "png"],
            help="Upload SillyTavern character card (JSON or PNG)"
        )

        if card_file and st.button("📥 Load Character"):
            try:
                # Save uploaded file temporarily
                temp_path = Path(f"temp_{card_file.name}")
                with open(temp_path, "wb") as f:
                    f.write(card_file.read())

                # Load character
                persona = st.session_state.persona_manager.load_card(temp_path)
                st.success(f"Loaded character: {persona.name} ({persona.role})")

                # Clean up
                temp_path.unlink()

            except Exception as e:
                st.error(f"Error loading character: {e}")

        st.markdown("---")

        # Persona Slot Selection
        st.markdown("### 🎯 Persona Assignments")

        available_personas = st.session_state.persona_manager.list_personas()

        st.session_state.interviewer_persona = st.selectbox(
            "Interviewer (Evaluator)",
            options=[p.lower().replace(" ", "_") for p in available_personas],
            index=0,
            help="Who evaluates your answers?"
        )

        st.session_state.tutor_persona = st.selectbox(
            "Tutor (Wrong Answer)",
            options=[p.lower().replace(" ", "_") for p in available_personas],
            index=1 if len(available_personas) > 1 else 0,
            help="Who teaches you when wrong?"
        )

        st.session_state.mentor_persona = st.selectbox(
            "Mentor (Correct Answer)",
            options=[p.lower().replace(" ", "_") for p in available_personas],
            index=2 if len(available_personas) > 2 else 0,
            help="Who encourages you when correct?"
        )

        st.markdown("---")

        # Stats
        st.markdown("### 📊 Your Progress")
        stats = st.session_state.knowledge_engine.get_progress_statistics()

        st.metric("Total Questions", stats['total_questions'])
        st.metric("Due for Review", stats['due_for_review'])
        success_rate = stats['overall_success_rate'] * 100
        st.metric("Success Rate", f"{success_rate:.1f}%")

    # ========================================================================
    # Main Area
    # ========================================================================

    # Load initial question if none exists
    if st.session_state.current_question is None:
        if st.button("🎬 Start Interview Session"):
            load_next_question()

    # Display current session
    if st.session_state.current_question:
        # Create two columns for character and dialogue
        char_col, dialogue_col = st.columns([1, 2])

        with char_col:
            st.markdown("### 👤 Current Speaker")

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
                st.info("🎨 Character Tachie\n\n(Place character images here based on persona)")

        with dialogue_col:
            st.markdown("### 💬 Interview Session")

            # Display question
            if st.session_state.evaluation_result is None:
                display_question(st.session_state.current_question)

                # Show expected answer hint (optional)
                with st.expander("💡 Answer Key (for reference)"):
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
                        {'✅ Correct!' if is_correct else '❌ Incorrect'}
                    </h3>
                    <p style="font-size: 18px; margin: 10px 0;">
                        <strong>Score:</strong> {score}/10
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
                            {persona_name}'s Feedback:
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
                    if st.button("⏭️ Next Question", use_container_width=True):
                        load_next_question()
                        st.rerun()

    # ========================================================================
    # Bottom - User Input Area
    # ========================================================================

    if st.session_state.current_question and st.session_state.evaluation_result is None:
        st.markdown("---")
        st.markdown("### ✍️ Your Answer")

        # Text input for answer
        user_answer = st.text_area(
            "Type your answer here...",
            value=st.session_state.user_answer,
            height=150,
            key="answer_input",
            help="Provide your answer to the question above"
        )

        # Update session state
        st.session_state.user_answer = user_answer

        # Submit button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("📤 Submit Answer", use_container_width=True, type="primary"):
                asyncio.run(process_answer())
                st.rerun()

    # ========================================================================
    # Progress Display
    # ========================================================================

    st.markdown("---")
    st.markdown("### 📈 Session Statistics")

    stats = st.session_state.knowledge_engine.get_progress_statistics()
    display_stats(stats)


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    main()
