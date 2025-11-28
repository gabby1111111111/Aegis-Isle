"""
Knowledge Engine for Interview Prep System

Implements spaced repetition learning algorithm for interview questions.
Manages question database and integrates with LLM for content generation.
"""

import json
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict
from enum import IntEnum

from pydantic import BaseModel, Field, validator

from ..core.logging import logger
from ..core.config import settings


class Difficulty(IntEnum):
    """Question difficulty levels."""
    VERY_EASY = 1
    EASY = 2
    MEDIUM = 3
    HARD = 4
    VERY_HARD = 5


class ReviewBox(IntEnum):
    """Spaced repetition review boxes (0 = new, 1-5 = increasing intervals)."""
    NEW = 0
    BOX_1 = 1  # 1 day
    BOX_2 = 2  # 3 days
    BOX_3 = 3  # 7 days
    BOX_4 = 4  # 14 days
    BOX_5 = 5  # 30 days


class Question(BaseModel):
    """
    Interview question data model with spaced repetition support.

    Attributes:
        id: Unique question identifier
        content: Question text
        answer_key: Optional reference answer/key points
        difficulty: Difficulty level (1-5)
        review_box: Current spaced repetition box (0-5)
        next_review: DateTime string for next review (ISO format)
        created_at: Creation timestamp
        category: Question category (e.g., "algorithms", "system_design")
        tags: Associated tags for filtering
        source: Source of the question (e.g., job description, study material)
        attempts: Number of times question was attempted
        correct_answers: Number of correct answers
    """

    id: str = Field(..., description="Unique question identifier")
    content: str = Field(..., min_length=10, description="Question text")
    answer_key: Optional[str] = Field(None, description="Reference answer or key points")
    difficulty: int = Field(..., ge=1, le=5, description="Difficulty level (1-5)")
    review_box: int = Field(default=0, ge=0, le=5, description="Spaced repetition box (0-5)")
    next_review: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="Next review datetime (ISO format)"
    )
    created_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="Creation timestamp"
    )
    category: str = Field(default="general", description="Question category")
    tags: List[str] = Field(default_factory=list, description="Associated tags")
    source: str = Field(default="unknown", description="Source of question")
    attempts: int = Field(default=0, ge=0, description="Number of attempts")
    correct_answers: int = Field(default=0, ge=0, description="Number of correct answers")

    @validator('next_review', 'created_at')
    def validate_datetime_string(cls, v):
        """Validate datetime string format."""
        try:
            datetime.fromisoformat(v.replace('Z', '+00:00'))
            return v
        except ValueError:
            raise ValueError("DateTime must be in ISO format")

    @property
    def next_review_datetime(self) -> datetime:
        """Get next_review as datetime object."""
        return datetime.fromisoformat(self.next_review.replace('Z', '+00:00'))

    @property
    def created_at_datetime(self) -> datetime:
        """Get created_at as datetime object."""
        return datetime.fromisoformat(self.created_at.replace('Z', '+00:00'))

    @property
    def success_rate(self) -> float:
        """Calculate success rate (0.0 to 1.0)."""
        if self.attempts == 0:
            return 0.0
        return self.correct_answers / self.attempts

    def is_due_for_review(self) -> bool:
        """Check if question is due for review."""
        return datetime.utcnow() >= self.next_review_datetime

    def update_review_schedule(self, is_correct: bool):
        """
        Update review schedule based on answer correctness.

        Args:
            is_correct: Whether the answer was correct
        """
        self.attempts += 1

        if is_correct:
            self.correct_answers += 1
            # Move to next box (increase interval)
            if self.review_box < ReviewBox.BOX_5:
                self.review_box += 1

            # Calculate next review time based on box
            intervals = {
                ReviewBox.BOX_1: timedelta(days=1),
                ReviewBox.BOX_2: timedelta(days=3),
                ReviewBox.BOX_3: timedelta(days=7),
                ReviewBox.BOX_4: timedelta(days=14),
                ReviewBox.BOX_5: timedelta(days=30)
            }

            next_interval = intervals.get(self.review_box, timedelta(days=1))
            self.next_review = (datetime.utcnow() + next_interval).isoformat()
        else:
            # Reset to immediate review
            self.review_box = ReviewBox.NEW
            self.next_review = datetime.utcnow().isoformat()


class KnowledgeEngine:
    """
    Manages interview questions database with spaced repetition learning.

    Features:
    - JSON-based question database
    - LLM-powered question generation from text/job descriptions
    - Spaced repetition algorithm for optimal learning
    - Progress tracking and analytics
    """

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize KnowledgeEngine.

        Args:
            db_path: Path to question database JSON file
        """
        self.db_path = db_path or Path("data/interview_db.json")
        self.questions: Dict[str, Question] = {}
        self.load_database()

        logger.info(f"KnowledgeEngine initialized with {len(self.questions)} questions")

    def load_database(self):
        """Load questions from JSON database file."""
        try:
            if self.db_path.exists():
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Convert dict data to Question objects
                for q_id, q_data in data.get('questions', {}).items():
                    try:
                        question = Question(**q_data)
                        self.questions[q_id] = question
                    except Exception as e:
                        logger.warning(f"Failed to load question {q_id}: {e}")

                logger.info(f"Loaded {len(self.questions)} questions from database")
            else:
                # Create empty database
                self.save_database()
                logger.info("Created new question database")

        except Exception as e:
            logger.error(f"Failed to load question database: {e}")
            self.questions = {}

    def save_database(self):
        """Save questions to JSON database file."""
        try:
            # Ensure directory exists
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert questions to serializable format
            data = {
                'questions': {q_id: q.dict() for q_id, q in self.questions.items()},
                'metadata': {
                    'total_questions': len(self.questions),
                    'last_updated': datetime.utcnow().isoformat()
                }
            }

            with open(self.db_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.debug(f"Saved {len(self.questions)} questions to database")

        except Exception as e:
            logger.error(f"Failed to save question database: {e}")

    async def ingest_data(self, text: str, jd_context: Optional[str] = None) -> List[Question]:
        """
        Generate interview questions from text using LLM.

        Args:
            text: Source text (study material, documentation, etc.)
            jd_context: Optional job description for contextual questions

        Returns:
            List of generated Question objects

        Raises:
            Exception: If LLM generation fails
        """
        try:
            from ..rag.generator import LLMGenerator, GenerationConfig

            # Prepare prompt for question generation
            prompt = self._build_question_generation_prompt(text, jd_context)

            # Initialize text generator
            config = GenerationConfig(
                model=settings.default_llm_model,
                max_tokens=2000,
                temperature=0.7
            )

            generator = LLMGenerator(config, provider=settings.llm_provider)

            # Generate questions
            logger.info("Generating interview questions from text...")
            result = await generator.generate(prompt)

            # Parse generated questions
            questions = self._parse_generated_questions(result.generated_text, jd_context)

            # Add to database
            for question in questions:
                self.questions[question.id] = question

            # Save database
            self.save_database()

            logger.info(f"Generated and added {len(questions)} questions to database")
            return questions

        except Exception as e:
            logger.error(f"Failed to ingest data: {e}")
            raise

    def _build_question_generation_prompt(self, text: str, jd_context: Optional[str] = None) -> str:
        """Build prompt for LLM question generation."""

        context_section = ""
        if jd_context:
            context_section = f"""

Job Description Context:
{jd_context[:1500]}  # Limit context to prevent token overflow

Focus questions on skills and requirements mentioned in this job description.
"""

        prompt = f"""Based on the following text, generate relevant interview questions that could be asked about this topic.

Source Text:
{text[:2000]}  # Limit input text to prevent token overflow
{context_section}

Please generate 5-10 interview questions in the following JSON format:

```json
{{
  "questions": [
    {{
      "content": "What is the time complexity of binary search?",
      "answer_key": "O(log n) - because we eliminate half the search space in each iteration",
      "difficulty": 3,
      "category": "algorithms",
      "tags": ["binary_search", "time_complexity", "algorithms"]
    }},
    {{
      "content": "Explain the difference between REST and GraphQL APIs",
      "answer_key": "REST uses multiple endpoints with HTTP verbs; GraphQL uses single endpoint with query language for flexible data fetching",
      "difficulty": 4,
      "category": "api_design",
      "tags": ["rest", "graphql", "api", "web_development"]
    }}
  ]
}}
```

Guidelines:
1. Difficulty scale: 1=Very Easy, 2=Easy, 3=Medium, 4=Hard, 5=Very Hard
2. Questions should be specific and answerable
3. Include diverse difficulty levels (mix of 2-4 mostly)
4. Provide concise but accurate answer_key
5. Use relevant categories and tags
6. Focus on practical knowledge and understanding
7. Make questions realistic for actual interviews

Return only the JSON format, no additional text."""

        return prompt

    def _parse_generated_questions(self, llm_output: str, source_context: Optional[str] = None) -> List[Question]:
        """
        Parse LLM-generated questions from JSON output.

        Args:
            llm_output: Raw LLM response containing JSON
            source_context: Original source context for metadata

        Returns:
            List of Question objects
        """
        questions = []

        try:
            # Extract JSON from LLM output
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', llm_output, re.DOTALL)
            if not json_match:
                # Try to find JSON without code blocks
                json_match = re.search(r'\{.*\}', llm_output, re.DOTALL)

            if not json_match:
                raise ValueError("No JSON found in LLM output")

            json_text = json_match.group(1) if json_match.lastindex else json_match.group(0)
            data = json.loads(json_text)

            # Parse questions from JSON
            for i, q_data in enumerate(data.get('questions', [])):
                try:
                    # Generate unique ID
                    question_id = f"gen_{datetime.utcnow().timestamp()}_{i:02d}"

                    # Create Question object
                    question = Question(
                        id=question_id,
                        content=q_data['content'],
                        answer_key=q_data.get('answer_key', ''),
                        difficulty=q_data.get('difficulty', 3),
                        category=q_data.get('category', 'general'),
                        tags=q_data.get('tags', []),
                        source=source_context or "llm_generated"
                    )

                    questions.append(question)

                except Exception as e:
                    logger.warning(f"Failed to parse question {i}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Failed to parse LLM-generated questions: {e}")
            # Return empty list rather than crash
            return []

        return questions

    def get_next_question(self) -> Optional[Question]:
        """
        Get next question based on spaced repetition algorithm.

        Priority:
        1. Questions due for review (next_review <= now)
        2. New questions (review_box == 0), sorted by difficulty (easy to hard)

        Returns:
            Next Question object or None if no questions available
        """
        if not self.questions:
            return None

        now = datetime.utcnow()

        # Get questions due for review
        due_questions = [
            q for q in self.questions.values()
            if q.next_review_datetime <= now and q.review_box > ReviewBox.NEW
        ]

        if due_questions:
            # Sort by review urgency (oldest due first)
            due_questions.sort(key=lambda q: q.next_review_datetime)
            logger.debug(f"Selected review question: {due_questions[0].content[:50]}...")
            return due_questions[0]

        # Get new questions (never reviewed)
        new_questions = [
            q for q in self.questions.values()
            if q.review_box == ReviewBox.NEW
        ]

        if new_questions:
            # Sort by difficulty (easy to hard)
            new_questions.sort(key=lambda q: (q.difficulty, q.created_at))
            logger.debug(f"Selected new question: {new_questions[0].content[:50]}...")
            return new_questions[0]

        # No questions available (all questions are scheduled for future review)
        logger.info("No questions currently available for review")
        return None

    def update_progress(self, question_id: str, is_correct: bool) -> bool:
        """
        Update question progress after answering.

        Args:
            question_id: ID of the answered question
            is_correct: Whether the answer was correct

        Returns:
            True if update successful, False if question not found
        """
        question = self.questions.get(question_id)
        if not question:
            logger.warning(f"Question not found for progress update: {question_id}")
            return False

        # Update review schedule
        question.update_review_schedule(is_correct)

        # Save database
        self.save_database()

        logger.info(
            f"Updated progress for question {question_id}: "
            f"{'correct' if is_correct else 'incorrect'}, "
            f"box={question.review_box}, "
            f"next_review={question.next_review_datetime.strftime('%Y-%m-%d %H:%M')}"
        )

        return True

    def get_questions_by_category(self, category: str) -> List[Question]:
        """Get all questions in a specific category."""
        return [q for q in self.questions.values() if q.category == category]

    def get_questions_by_difficulty(self, difficulty: int) -> List[Question]:
        """Get all questions of specific difficulty level."""
        return [q for q in self.questions.values() if q.difficulty == difficulty]

    def get_questions_due_for_review(self) -> List[Question]:
        """Get all questions currently due for review."""
        now = datetime.utcnow()
        return [q for q in self.questions.values() if q.next_review_datetime <= now]

    def get_progress_statistics(self) -> Dict[str, Any]:
        """
        Get learning progress statistics.

        Returns:
            Dictionary with various progress metrics
        """
        total_questions = len(self.questions)

        if total_questions == 0:
            return {
                'total_questions': 0,
                'questions_by_box': {},
                'due_for_review': 0,
                'overall_success_rate': 0.0,
                'questions_by_difficulty': {},
                'questions_by_category': {}
            }

        # Count questions by review box
        box_counts = {}
        for i in range(6):  # Boxes 0-5
            box_counts[f'box_{i}'] = len([q for q in self.questions.values() if q.review_box == i])

        # Count due questions
        due_count = len(self.get_questions_due_for_review())

        # Calculate overall success rate
        total_attempts = sum(q.attempts for q in self.questions.values())
        total_correct = sum(q.correct_answers for q in self.questions.values())
        overall_success_rate = (total_correct / total_attempts) if total_attempts > 0 else 0.0

        # Count by difficulty
        difficulty_counts = {}
        for i in range(1, 6):
            difficulty_counts[f'difficulty_{i}'] = len([q for q in self.questions.values() if q.difficulty == i])

        # Count by category
        categories = set(q.category for q in self.questions.values())
        category_counts = {cat: len([q for q in self.questions.values() if q.category == cat]) for cat in categories}

        return {
            'total_questions': total_questions,
            'questions_by_box': box_counts,
            'due_for_review': due_count,
            'overall_success_rate': round(overall_success_rate, 3),
            'questions_by_difficulty': difficulty_counts,
            'questions_by_category': category_counts,
            'last_updated': datetime.utcnow().isoformat()
        }

    def search_questions(self, query: str, limit: int = 10) -> List[Question]:
        """
        Search questions by content, tags, or category.

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of matching Question objects
        """
        query_lower = query.lower()
        matches = []

        for question in self.questions.values():
            # Search in content, category, tags
            if (query_lower in question.content.lower() or
                query_lower in question.category.lower() or
                any(query_lower in tag.lower() for tag in question.tags) or
                (question.answer_key and query_lower in question.answer_key.lower())):
                matches.append(question)

        # Sort by relevance (simple scoring)
        matches.sort(key=lambda q: (
            query_lower in q.content.lower(),
            query_lower in q.category.lower(),
            any(query_lower in tag.lower() for tag in q.tags)
        ), reverse=True)

        return matches[:limit]

    def add_question(self, content: str, answer_key: str = "", difficulty: int = 3,
                    category: str = "general", tags: List[str] = None) -> Question:
        """
        Manually add a question to the database.

        Args:
            content: Question text
            answer_key: Reference answer
            difficulty: Difficulty level (1-5)
            category: Question category
            tags: List of tags

        Returns:
            Created Question object
        """
        if tags is None:
            tags = []

        question_id = f"manual_{datetime.utcnow().timestamp()}_{len(self.questions):04d}"

        question = Question(
            id=question_id,
            content=content,
            answer_key=answer_key,
            difficulty=difficulty,
            category=category,
            tags=tags,
            source="manual_entry"
        )

        self.questions[question_id] = question
        self.save_database()

        logger.info(f"Added manual question: {content[:50]}...")
        return question

    def delete_question(self, question_id: str) -> bool:
        """
        Delete a question from the database.

        Args:
            question_id: ID of question to delete

        Returns:
            True if deleted successfully, False if not found
        """
        if question_id in self.questions:
            del self.questions[question_id]
            self.save_database()
            logger.info(f"Deleted question: {question_id}")
            return True

        logger.warning(f"Question not found for deletion: {question_id}")
        return False