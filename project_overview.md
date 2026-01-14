# Aegis Isle Project Overview

## Project Structure

The project is organized into a clear frontend-backend architecture, with the core logic residing in `src/aegis_isle` and the user interface in `frontend`.

### Core Components (`src/aegis_isle`)

#### Interview Module (`src/aegis_isle/interview`)
This module contains the heart of the "Infinite Interview" system.

- **`generator.py`**: The "Polyphonic Questioning" engine. It orchestrates the interaction with the LLM to generate:
    - **In-Character Questions**: Technical questions wrapped in the persona's lore and tone.
    - **Three-Fold Judgment**: Feedback consisting of the Character's Verdict, a Standard Technical Answer, and a "Servitor" (ELI5) Explanation.
    - **Story Nodes**: Narrative segments triggered by progress.

- **`knowledge_engine.py`**: Manages the technical content using a **Spaced Repetition System (SRS)**.
    - **`Question` Model**: Tracks difficulty, review history, and "Review Box" (Leitner system).
    - **`KnowledgeEngine`**: Handles question selection based on the user's learning progress.

- **`persona_manager.py`**: Handles the "Soul" of the interviewer.
    - **`Persona` Model**: Defines the character's name, role, personality, and world lore.
    - **SillyTavern Support**: Capable of loading character cards (JSON/PNG) to import custom characters.
    - **System Prompts**: Generates immersive prompts to keep the LLM in character.

- **`story_manager.py`**: Manages the narrative arc.
    - Tracks "Story Nodes" (milestones) based on the user's mastery of questions (Review Box progress).
    - Triggers special story events (e.g., "Gene Surgery", "Warp Contact") when criteria are met.

#### Frontend (`frontend`)

- **`interview_app.py`**: The main entry point for the Streamlit application.
    - **Visual Novel UI**: Implements a highly stylized, manga-inspired interface.
    - **Key Features**:
        - **Immersive Dialogue**: Character nameplates, typing effects, and manga-style backgrounds.
        - **Interactive Elements**: Custom-styled input boxes, buttons, and progress bars.
        - **"Emperor's Satisfaction"**: A gamified progress tracking system with visual feedback (shake/fireworks).
        - **Configuration Panel**: A hidden sidebar for uploading characters and adjusting settings.

### Key Workflows

1. **Initialization**: The app loads the `KnowledgeEngine` and `PersonaManager`. The user selects or uploads a character (e.g., The Emperor).
2. **Question Generation**: `Generator` fetches a question from `KnowledgeEngine` and uses the `Persona` to rewrite it into an in-character challenge.
3. **User Interaction**: The user answers in the custom UI.
4. **Feedback & Judgment**: `Generator` evaluates the answer, providing the "Three-Fold Judgment".
5. **Progression**:
    - `KnowledgeEngine` updates the question's review schedule.
    - `StoryManager` checks for narrative milestones.
    - "Emperor's Satisfaction" score is updated visually.

## Recent Updates

- **UI Overhaul**: Transformed the standard Streamlit interface into a "Visual Novel" style with black-and-white manga aesthetics.
- **Feedback System**: Implemented the "Three-Fold Judgment" display (Verdict, Standard Answer, ELI5).
- **Gamification**: Added the "Emperor's Satisfaction" progress bar with dynamic visual effects.
- **Navigation**: Added custom "Menu" and "Settings" buttons for better UX.
