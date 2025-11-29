import sys
import os
import asyncio
from unittest.mock import AsyncMock, MagicMock

# Add src to path
sys.path.insert(0, os.path.abspath("src"))

# Mock dependencies
sys.modules["PIL"] = MagicMock()
sys.modules["PIL.Image"] = MagicMock()
sys.modules["openai"] = MagicMock()
sys.modules["anthropic"] = MagicMock()

# Mock internal modules to avoid external dependencies
sys.modules["aegis_isle.interview.graph"] = MagicMock()

# Mock LLMGenerator to prevent instantiation issues
mock_rag_generator = MagicMock()
sys.modules["aegis_isle.rag.generator"] = mock_rag_generator

# Create a class that returns an instance with AsyncMock methods
class MockLLMGeneratorClass:
    def __init__(self, *args, **kwargs):
        self.generate = AsyncMock()

mock_rag_generator.LLMGenerator = MockLLMGeneratorClass
mock_rag_generator.GenerationConfig = MagicMock()

try:
    # Import Generator from package root to verify __init__.py export
    from aegis_isle.interview import Generator
    from aegis_isle.interview.persona_manager import Persona
    from aegis_isle.interview.knowledge_engine import Question
    print("Imports successful!")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

async def test_generator():
    print("Starting test...")
    
    # Mock LLM
    mock_llm = AsyncMock()
    mock_response = MagicMock()
    mock_response.generated_text = """
    {
        "role_flavor": {
            "scenario": "Lab setting.",
            "in_character_question": "Compute the sum.",
            "encouragement": "Do it."
        },
        "original_question": "What is 2+2?",
        "hints": {
            "tech_keywords": ["addition"],
            "eli5_analogy": "Counting."
        }
    }
    """
    mock_llm.generate.return_value = mock_response

    # Patch Generator's LLM
    generator = Generator()
    generator.llm = mock_llm
    
    persona = Persona(
        name="Test", role="Bot", description="Desc", personality="Pers",
        first_message="Hi", example_messages="Ex"
    )
    question = Question(
        id="q1", content="What is 2+2? Please explain.", difficulty=1
    )
    
    print("Testing generate_question_interaction with language='zh'...")
    # Update mock to return Chinese-like response (simulated)
    mock_response.generated_text = """
    {
        "role_flavor": {
            "scenario": "实验室环境。",
            "in_character_question": "计算和。",
            "encouragement": "做吧。"
        },
        "original_question": "What is 2+2?",
        "hints": {
            "tech_keywords": ["加法"],
            "eli5_analogy": "数数。"
        }
    }
    """
    
    result = await generator.generate_question_interaction(persona, question, language="zh")
    print(f"Result: {result}")
    
    # Verify that the prompt contained the language instruction
    call_args = mock_llm.generate.call_args[0][0]
    if "Chinese (Simplified)" in call_args:
        print("SUCCESS: Language instruction found in prompt")
    else:
        print("FAILURE: Language instruction NOT found in prompt")
        print(f"Prompt was: {call_args}")

    if result["role_flavor"]["scenario"] == "实验室环境。":
        print("SUCCESS: Scenario matches Chinese output")
    else:
        print("FAILURE: Scenario mismatch")

if __name__ == "__main__":
    asyncio.run(test_generator())
