import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from aegis_isle.interview.generator import Generator
from aegis_isle.interview.persona_manager import Persona
from aegis_isle.interview.knowledge_engine import Question


@pytest.fixture
def mock_llm():
    with patch("aegis_isle.interview.generator.LLMGenerator") as MockLLM:
        mock_instance = MockLLM.return_value
        yield mock_instance


@pytest.fixture
def sample_persona():
    return Persona(
        name="TestBot",
        role="Interviewer",
        description="A test bot.",
        personality="Robotic.",
        first_message="Hello.",
        example_messages="User: Hi\nBot: Hello.",
        scenario="Testing lab.",
    )


@pytest.fixture
def sample_question():
    return Question(
        id="q1", content="What is 2+2?", answer_key="4", difficulty=1, category="math"
    )


@pytest.mark.asyncio
async def test_generate_question_interaction(mock_llm, sample_persona, sample_question):
    # Mock LLM response
    mock_response = MagicMock()
    mock_response.generated_text = """
    ```json
    {
        "role_flavor": {
            "scenario": "Lab setting.",
            "in_character_question": "Compute the sum of two pairs.",
            "encouragement": "Do it."
        },
        "original_question": "What is 2+2?",
        "hints": {
            "tech_keywords": ["addition"],
            "eli5_analogy": "Counting fingers."
        }
    }
    ```
    """
    mock_llm.generate = AsyncMock(return_value=mock_response)

    generator = Generator()
    # Inject mock
    generator.llm = mock_llm

    result = await generator.generate_question_interaction(
        sample_persona, sample_question
    )

    assert result["role_flavor"]["scenario"] == "Lab setting."
    assert result["original_question"] == "What is 2+2?"
    assert "addition" in result["hints"]["tech_keywords"]


@pytest.mark.asyncio
async def test_generate_feedback(mock_llm, sample_persona, sample_question):
    # Mock LLM response
    mock_response = MagicMock()
    mock_response.generated_text = """
    {
        "character_verdict": {
            "status": "correct",
            "comment": "Good job."
        },
        "standard_answer": "4",
        "eli5_explanation": "1, 2, 3, 4."
    }
    """
    mock_llm.generate = AsyncMock(return_value=mock_response)

    generator = Generator()
    generator.llm = mock_llm

    result = await generator.generate_feedback(
        sample_persona, sample_question, "It is 4", {}
    )

    assert result["character_verdict"]["status"] == "correct"
    assert result["standard_answer"] == "4"
