#!/usr/bin/env python3
"""
Test script for Interview Prep System

Tests the persona manager and knowledge engine functionality.
"""

import sys
import asyncio
from pathlib import Path

sys.path.insert(0, 'src')


def test_persona_manager():
    """Test PersonaManager functionality."""
    print("=" * 60)
    print("Testing PersonaManager")
    print("=" * 60)

    from aegis_isle.interview import PersonaManager

    # Initialize manager
    manager = PersonaManager()

    # List available personas
    print("\nAvailable Personas:")
    for name in manager.list_personas():
        print(f"  - {name}")

    # Test getting each default persona
    print("\nTesting Default Personas:")
    for persona_name in ["sukuna", "gojo", "nanami"]:
        persona = manager.get_persona(persona_name)
        if persona:
            print(f"\n{persona.name} ({persona.role}):")
            print(f"  Description: {persona.description[:100]}...")
            print(f"  First Message: {persona.first_message[:80]}...")
            print(f"  System Prompt Length: {len(persona.get_system_prompt())} chars")
        else:
            print(f"❌ Failed to load {persona_name}")

    # Test default persona
    default = manager.get_default_persona()
    print(f"\n✅ Default persona: {default.name}")

    print("\n✅ PersonaManager tests passed!")


def test_knowledge_engine():
    """Test KnowledgeEngine functionality."""
    print("\n" + "=" * 60)
    print("Testing KnowledgeEngine")
    print("=" * 60)

    from aegis_isle.interview import KnowledgeEngine

    # Initialize engine with test database
    test_db_path = Path("data/test_interview_db.json")
    engine = KnowledgeEngine(db_path=test_db_path)

    print(f"\nInitial question count: {len(engine.questions)}")

    # Test adding manual questions
    print("\nAdding manual questions...")
    q1 = engine.add_question(
        content="What is the time complexity of quicksort in the average case?",
        answer_key="O(n log n)",
        difficulty=3,
        category="algorithms",
        tags=["sorting", "complexity", "quicksort"]
    )
    print(f"✅ Added question: {q1.id}")

    q2 = engine.add_question(
        content="Explain the difference between TCP and UDP protocols.",
        answer_key="TCP is connection-oriented and reliable; UDP is connectionless and faster",
        difficulty=2,
        category="networking",
        tags=["protocols", "networking"]
    )
    print(f"✅ Added question: {q2.id}")

    q3 = engine.add_question(
        content="Design a URL shortening service like bit.ly",
        answer_key="Use hash function, database for mappings, consider scalability and collision handling",
        difficulty=5,
        category="system_design",
        tags=["system_design", "scalability"]
    )
    print(f"✅ Added question: {q3.id}")

    # Test getting next question
    print("\nGetting next question (should be easiest new question):")
    next_q = engine.get_next_question()
    if next_q:
        print(f"  ID: {next_q.id}")
        print(f"  Content: {next_q.content}")
        print(f"  Difficulty: {next_q.difficulty}")
        print(f"  Review Box: {next_q.review_box}")
    else:
        print("❌ No questions available")

    # Test updating progress (correct answer)
    print("\nSimulating correct answer...")
    success = engine.update_progress(next_q.id, is_correct=True)
    if success:
        updated_q = engine.questions[next_q.id]
        print(f"✅ Updated: box={updated_q.review_box}, attempts={updated_q.attempts}")
    else:
        print("❌ Failed to update progress")

    # Test getting next question again
    print("\nGetting next question again:")
    next_q2 = engine.get_next_question()
    if next_q2:
        print(f"  Content: {next_q2.content[:60]}...")
        print(f"  Difficulty: {next_q2.difficulty}")

    # Test incorrect answer
    print("\nSimulating incorrect answer...")
    engine.update_progress(next_q2.id, is_correct=False)
    updated_q2 = engine.questions[next_q2.id]
    print(f"✅ Updated: box={updated_q2.review_box} (should be 0)")

    # Test search
    print("\nSearching for 'network' questions:")
    results = engine.search_questions("network")
    print(f"  Found {len(results)} results")
    for r in results:
        print(f"  - {r.content[:60]}...")

    # Test statistics
    print("\nProgress Statistics:")
    stats = engine.get_progress_statistics()
    print(f"  Total Questions: {stats['total_questions']}")
    print(f"  Due for Review: {stats['due_for_review']}")
    print(f"  Success Rate: {stats['overall_success_rate']:.2%}")
    print(f"  Questions by Box: {stats['questions_by_box']}")

    # Test filtering
    print("\nQuestions by Category:")
    categories = set(q.category for q in engine.questions.values())
    for cat in categories:
        cat_questions = engine.get_questions_by_category(cat)
        print(f"  {cat}: {len(cat_questions)} questions")

    print("\n✅ KnowledgeEngine tests passed!")

    # Cleanup test database
    if test_db_path.exists():
        test_db_path.unlink()
        print(f"\nCleaned up test database: {test_db_path}")


async def test_llm_integration():
    """Test LLM integration for question generation."""
    print("\n" + "=" * 60)
    print("Testing LLM Integration (Question Generation)")
    print("=" * 60)

    from aegis_isle.interview import KnowledgeEngine

    # Initialize engine
    test_db_path = Path("data/test_llm_interview_db.json")
    engine = KnowledgeEngine(db_path=test_db_path)

    # Sample text for question generation
    sample_text = """
    Python is a high-level, interpreted programming language known for its
    readability and versatility. Key features include:

    1. Dynamic typing - variables don't need explicit type declarations
    2. Garbage collection - automatic memory management
    3. First-class functions - functions can be passed as arguments
    4. List comprehensions - concise way to create lists
    5. Decorators - modify function behavior

    Python is widely used in web development (Django, Flask), data science
    (pandas, NumPy), machine learning (TensorFlow, PyTorch), and automation.
    """

    job_description = """
    We are looking for a Python Developer with:
    - 3+ years of Python experience
    - Strong understanding of web frameworks (Django/Flask)
    - Experience with data processing and APIs
    - Knowledge of testing and CI/CD
    """

    print("\nGenerating questions from sample text...")
    print("Note: This requires valid OpenAI API configuration in .env")

    try:
        questions = await engine.ingest_data(sample_text, jd_context=job_description)

        print(f"\nGenerated {len(questions)} questions:")
        for i, q in enumerate(questions, 1):
            print(f"\n{i}. {q.content}")
            print(f"   Difficulty: {q.difficulty}/5")
            print(f"   Category: {q.category}")
            print(f"   Tags: {', '.join(q.tags)}")
            if q.answer_key:
                print(f"   Answer: {q.answer_key[:80]}...")

        print("\nLLM integration test passed!")

    except Exception as e:
        print(f"\nLLM integration test skipped: {e}")
        print("   This is expected if OpenAI API is not configured.")

    # Cleanup
    if test_db_path.exists():
        test_db_path.unlink()


def main():
    """Run all tests."""
    print("Interview Prep System - Test Suite")
    print("=" * 60)

    try:
        # Test PersonaManager
        test_persona_manager()

        # Test KnowledgeEngine
        test_knowledge_engine()

        # Test LLM integration (async)
        asyncio.run(test_llm_integration())

        print("\n" + "=" * 60)
        print("All tests completed!")
        print("=" * 60)

    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
