# Interview Prep System - Frontend UI

## 🎨 Visual Novel Style Interface

A beautiful Otome Game-inspired UI for the gamified interview preparation system.

![Theme](https://img.shields.io/badge/Theme-Visual_Novel-ff6b9d)
![Framework](https://img.shields.io/badge/Framework-Streamlit-red)
![Style](https://img.shields.io/badge/Style-Otome_Game-purple)

---

## ✨ Features

### 🎭 Character-Based Learning
- **Sukuna** - Strict interviewer who evaluates your answers
- **Gojo Satoru** - Playful tutor with ELI5 explanations
- **Nanami Kento** - Professional mentor who encourages growth

### 🎮 Game-Like Interface
- Visual Novel aesthetic with pink/purple dark theme
- Character tachie display area
- Elegant dialogue boxes with character names
- Smooth animations and transitions
- Rounded corners and custom fonts (Cinzel & Crimson Text)

### 📚 Knowledge Management
- Upload job descriptions for context
- Import study materials to generate questions
- Load custom SillyTavern character cards
- Assign personas to different roles

### 📊 Progress Tracking
- Real-time statistics display
- Spaced repetition algorithm
- Success rate tracking
- Mastery level indicators

---

## 🚀 Quick Start

### Prerequisites

```bash
# Ensure you have Streamlit installed
pip install streamlit

# Or install from requirements.txt
pip install -r requirements.txt
```

### Running the App

**Option 1: Using the startup script (Recommended)**

```bash
python run_interview_app.py
```

**Option 2: Direct Streamlit command**

```bash
streamlit run frontend/interview_app.py
```

**Option 3: From frontend directory**

```bash
cd frontend
streamlit run interview_app.py
```

The app will automatically open in your default browser at `http://localhost:8501`

---

## 📱 User Interface Guide

### Layout Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    ✨ Interview Prep Academy ✨             │
│            Master Your Skills with AI-Powered Personas       │
├───────────┬─────────────────────────────────────────────────┤
│           │                                                   │
│  SIDEBAR  │              MAIN INTERVIEW AREA                 │
│           │                                                   │
│  Config   │  ┌──────────────┐  ┌────────────────────────┐  │
│  Area:    │  │              │  │                        │  │
│           │  │  Character   │  │    Chat Interface      │  │
│  • JD     │  │   Tachie     │  │    - Question Display  │  │
│  • KB     │  │   (Image)    │  │    - Evaluation        │  │
│  • Cards  │  │              │  │    - Feedback          │  │
│  • Slots  │  │              │  │                        │  │
│  • Stats  │  └──────────────┘  └────────────────────────┘  │
│           │                                                   │
├───────────┼───────────────────────────────────────────────────┤
│           │              USER INPUT AREA                     │
│           │  ┌───────────────────────────────────────────┐  │
│           │  │  Type your answer here...                  │  │
│           │  │                                            │  │
│           │  └───────────────────────────────────────────┘  │
│           │           [📤 Submit Answer]                    │
└───────────┴───────────────────────────────────────────────────┘
```

### Sidebar - Configuration Area

#### 📄 Job Description
- **Purpose**: Provide context for question generation
- **Usage**: Paste job description in text area
- **Effect**: Questions will be tailored to JD requirements

#### 📚 Knowledge Base
- **Purpose**: Generate interview questions from study material
- **Usage**:
  1. Upload `.txt` or `.md` file
  2. Click "📥 Process Knowledge Base"
  3. System will use LLM to extract questions
- **Result**: Multiple questions added to database

#### 🎭 Character Cards
- **Purpose**: Add custom personas
- **Usage**:
  1. Upload SillyTavern character card (`.json` or `.png`)
  2. Click "📥 Load Character"
  3. Character appears in persona selection
- **Format**: Supports SillyTavern V2 Spec

#### 🎯 Persona Assignments
- **Interviewer**: Who evaluates your answers (default: Sukuna)
- **Tutor**: Who teaches when you're wrong (default: Gojo)
- **Mentor**: Who encourages when you're right (default: Nanami)

#### 📊 Your Progress
- **Total Questions**: Questions in knowledge base
- **Due for Review**: Questions that need review today
- **Success Rate**: Overall correct answer percentage

### Main Area - Interview Session

#### Left Column: Character Display
- Shows current speaking character
- Character name with role
- Character description
- Placeholder for tachie image

#### Right Column: Dialogue Interface

**Question Phase**:
- Question displayed in red Sukuna-styled box
- Difficulty stars (⭐ 1-5)
- Category badge
- Answer key available in expander

**Evaluation Phase**:
- Evaluation box (red for wrong, green for correct)
- Score display (0-10)
- Sukuna's harsh but fair comment

**Feedback Phase**:
- Tutor box (blue) for wrong answers - Gojo's ELI5 explanation
- Mentor box (green) for correct answers - Nanami's encouragement
- Next Question button

### Bottom Area - User Input
- Large text area for typing answer
- Submit Answer button (primary action)
- Only shown when question is active

---

## 🎨 Visual Design

### Color Scheme

| Element | Color | Hex |
|---------|-------|-----|
| Primary Pink | Background gradients | `#ff6b9d` |
| Secondary Purple | Borders, accents | `#8b5cf6` |
| Dark Background | Main BG | `#1a1625` |
| Card Background | Dialogue boxes | `#2d1b3d` |
| Accent Gold | Text highlights | `#ffd700` |
| Success Green | Correct answers | `#4ade80` |
| Error Red | Wrong answers | `#ef4444` |

### Typography

- **Headings**: Cinzel (serif, elegant)
- **Body Text**: Crimson Text (serif, readable)
- **UI Elements**: Default Streamlit fonts

### Animations

- **Fade In**: Elements smoothly appear
- **Pulse**: Question boxes gently pulse
- **Glow**: Title has animated glow effect
- **Hover**: Buttons lift on hover

---

## 🔧 Configuration

### Environment Variables

Required in `.env`:

```env
# LLM Configuration
LLM_PROVIDER=openai
DEFAULT_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
OPENAI_API_KEY=your-key-here
OPENAI_BASE_URL=https://api.siliconflow.cn/v1

# Optional: Adjust generation parameters
TEMPERATURE=0.7
MAX_TOKENS=1500
```

### Streamlit Configuration

Theme settings (applied automatically by `run_interview_app.py`):

```toml
[theme]
base = "dark"
primaryColor = "#ff6b9d"
backgroundColor = "#1a1625"
secondaryBackgroundColor = "#2d1b3d"
textColor = "#f0e6ff"
```

---

## 📖 Usage Workflow

### 1. Initial Setup

```
1. Start the app: python run_interview_app.py
2. (Optional) Upload job description in sidebar
3. (Optional) Upload knowledge base file
4. (Optional) Adjust persona assignments
5. Click "🎬 Start Interview Session"
```

### 2. Interview Loop

```
1. Question appears in Sukuna's red box
2. Read the question carefully
3. Type your answer in the input area
4. Click "📤 Submit Answer"
5. Wait for evaluation (spinner shows progress)
6. View Sukuna's evaluation (score + comment)
7. Read feedback:
   - If wrong: Gojo explains in simple terms
   - If right: Nanami encourages you
8. Click "⏭️ Next Question"
9. Repeat!
```

### 3. Progress Tracking

- Stats update automatically after each question
- Spaced repetition schedules reviews
- Mastery level increases with correct answers
- Success rate calculated in real-time

---

## 🎭 Persona Customization

### Using Custom Characters

1. **Create Character Card**:
   - Use SillyTavern format
   - Include: name, description, personality, first_mes, mes_example
   - Save as `.json` or `.png` with metadata

2. **Upload to App**:
   - Use "🎭 Character Cards" section in sidebar
   - Click upload and select file
   - Character loads immediately

3. **Assign to Role**:
   - Select character in persona dropdowns
   - Choose appropriate role based on personality
   - Changes take effect on next question

### Default Personas

**Sukuna (Interviewer)**:
- Style: Strict, demanding, uncompromising
- Use: Realistic interview pressure
- Temperature: 0.3 (consistent evaluation)

**Gojo Satoru (Tutor)**:
- Style: Playful, encouraging, ELI5
- Use: Learning and understanding
- Temperature: 0.8 (creative analogies)

**Nanami Kento (Mentor)**:
- Style: Professional, patient, methodical
- Use: Long-term growth guidance
- Temperature: 0.6 (balanced feedback)

---

## 🛠️ Technical Details

### Architecture

```
Frontend (Streamlit)
    ↓
Interview Graph (LangGraph)
    ↓ ↓ ↓
PersonaManager | KnowledgeEngine | TextGenerator
    ↓              ↓                    ↓
3 Personas    Question DB          LLM API
```

### State Management

Streamlit session state stores:
- Current question
- User answer
- Evaluation results
- Feedback text
- Conversation history
- Persona assignments
- Knowledge engine instance
- Persona manager instance

### Async Operations

- Question generation (LLM call)
- Answer evaluation (LLM call)
- Feedback generation (LLM call)
- Knowledge base ingestion (LLM call)

All handled via `asyncio.run()` in Streamlit.

---

## 🐛 Troubleshooting

### App Won't Start

**Error**: `ModuleNotFoundError: No module named 'streamlit'`

**Solution**:
```bash
pip install streamlit
```

**Error**: `No module named 'aegis_isle'`

**Solution**:
```bash
# Make sure you're in the project root
cd AegisIsle_root_directory

# Run from root, not from frontend folder
python run_interview_app.py
```

### LLM Errors

**Error**: `Error code: 401 - Incorrect API key`

**Solution**:
```bash
# Check .env file has valid API key
OPENAI_API_KEY=your-valid-key-here
OPENAI_BASE_URL=https://api.siliconflow.cn/v1
```

**Error**: `LLM generation failed: timeout`

**Solution**:
- Check internet connection
- Verify API endpoint is accessible
- Try increasing timeout in code

### No Questions Available

**Symptom**: "No questions available!" message

**Solution**:
1. Upload knowledge base file
2. Or manually add questions via KnowledgeEngine
3. Or generate questions from text

### Character Cards Won't Load

**Error**: `Error loading character: ...`

**Solution**:
- Verify file format (JSON or PNG)
- Check SillyTavern V2 format
- Ensure 'name' and 'description' fields exist

---

## 📚 Example Usage

### Sample Job Description

```
Software Engineer - Full Stack
Requirements:
- 3+ years Python/JavaScript experience
- Strong algorithms and data structures
- REST API design and implementation
- Database design (SQL/NoSQL)
- Cloud deployment (AWS/GCP)
```

### Sample Knowledge Base File

```
# Python Fundamentals

## Data Structures
- Lists are mutable, tuples are immutable
- Dictionaries use hash tables for O(1) lookup
- Sets are unordered collections of unique elements

## Algorithms
- Binary search has O(log n) complexity
- Quicksort average case is O(n log n)
- Dynamic programming uses memoization
```

---

## 🎯 Tips for Best Experience

1. **Start Easy**: Begin with easier questions to build confidence
2. **Be Thorough**: Write complete answers, not just keywords
3. **Learn from Mistakes**: Read Gojo's explanations carefully
4. **Track Progress**: Monitor your success rate regularly
5. **Use JD Context**: Upload job description for relevant questions
6. **Customize Personas**: Try different character combinations
7. **Regular Practice**: Use spaced repetition consistently

---

## 🔮 Future Enhancements

Potential features (not yet implemented):

- [ ] Character tachie image support (PNG display)
- [ ] Voice synthesis for character dialogue
- [ ] Achievement system with badges
- [ ] Session replay and review
- [ ] Multi-player collaborative mode
- [ ] Difficulty adaptation based on performance
- [ ] Export progress reports
- [ ] Mobile-responsive design
- [ ] Dark/Light theme toggle
- [ ] Custom CSS themes

---

## 📞 Support

For issues or questions:

1. Check this README first
2. Review main documentation (`INTERVIEW_MODULE.md`)
3. Check workflow docs (`INTERVIEW_WORKFLOW.md`)
4. Examine test files for examples

---

## 📄 License

Part of the AegisIsle Interview Prep System.

---

**Built with ❤️ using Streamlit, LangGraph, and AI**

Enjoy your visual novel-style interview preparation! ✨
