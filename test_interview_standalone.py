#!/usr/bin/env python3
"""
Standalone test for Interview Prep System core functionality.
Tests persona_manager and knowledge_engine without full project dependencies.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, 'src')


def test_persona_imports():
    """Test that persona_manager can be imported."""
    print("=" * 60)
    print("Testing Persona Manager Imports")
    print("=" * 60)

    try:
        from aegis_isle.interview.persona_manager import PersonaManager, Persona
        print("Successfully imported PersonaManager and Persona")
        return PersonaManager, Persona
    except Exception as e:
        print(f"Failed to import: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_persona_functionality():
    """Test PersonaManager without full project dependencies."""
    print("\n" + "=" * 60)
    print("Testing PersonaManager Functionality")
    print("=" * 60)

    # Import just the persona module
    try:
        from aegis_isle.interview.persona_manager import PersonaManager, Persona

        # Initialize manager (without custom personas dir for now)
        manager = PersonaManager(persona_dir=Path("data/personas_test"))

        print(f"\nLoaded {len(manager.personas)} personas")

        # List personas
        print("\nAvailable Personas:")
        for name in manager.list_personas():
            print(f"  - {name}")

        # Test default personas
        print("\nTesting Default Personas:")
        for name in ["sukuna", "gojo", "nanami"]:
            persona = manager.get_persona(name)
            if persona:
                print(f"\n{persona.name} ({persona.role}):")
                print(f"  Personality: {persona.personality[:60]}...")
                print(f"  First Message: {persona.first_message[:60]}...")

                # Test system prompt generation
                prompt = persona.get_system_prompt()
                print(f"  System Prompt: {len(prompt)} characters")
                assert len(prompt) > 100, "System prompt should be substantial"
            else:
                raise Exception(f"Failed to load persona: {name}")

        # Test default persona
        default = manager.get_default_persona()
        assert default.name == "Gojo Satoru", "Default should be Gojo"
        print(f"\nDefault persona: {default.name}")

        # Test role inference
        assert manager._infer_role("Test", "strict interview", "") == "Interviewer"
        assert manager._infer_role("Test", "teaching and tutoring", "") == "Tutor"
        print("\nRole inference working correctly")

        print("\nPersonaManager tests PASSED!")
        return True

    except Exception as e:
        print(f"\nPersonaManager test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_knowledge_engine_imports():
    """Test that knowledge_engine can be imported."""
    print("\n" + "=" * 60)
    print("Testing Knowledge Engine Imports")
    print("=" * 60)

    try:
        from aegis_isle.interview.knowledge_engine import (
            KnowledgeEngine, Question, Difficulty, ReviewBox
        )
        print("Successfully imported KnowledgeEngine, Question, Difficulty, ReviewBox")
        return True
    except Exception as e:
        print(f"Failed to import: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_question_model():
    """Test Question model functionality."""
    print("\n" + "=" * 60)
    print("Testing Question Model")
    print("=" * 60)

    try:
        from aegis_isle.interview.knowledge_engine import Question

        # Create a question
        q = Question(
            id="test_001",
            content="What is the difference between a list and tuple in Python?",
            answer_key="Lists are mutable, tuples are immutable",
            difficulty=2,
            category="python",
            tags=["data_structures", "python"]
        )

        print(f"Created question: {q.content[:50]}...")
        print(f"  Difficulty: {q.difficulty}")
        print(f"  Review Box: {q.review_box}")
        print(f"  Success Rate: {q.success_rate}")

        # Test update_review_schedule - correct answer
        print("\nTesting correct answer update...")
        q.update_review_schedule(is_correct=True)
        assert q.review_box == 1, "Box should increase to 1"
        assert q.attempts == 1, "Attempts should be 1"
        assert q.correct_answers == 1, "Correct answers should be 1"
        print(f"  Box: {q.review_box}, Attempts: {q.attempts}, Correct: {q.correct_answers}")

        # Test update_review_schedule - incorrect answer
        print("\nTesting incorrect answer update...")
        q.update_review_schedule(is_correct=False)
        assert q.review_box == 0, "Box should reset to 0"
        assert q.attempts == 2, "Attempts should be 2"
        assert q.correct_answers == 1, "Correct answers should still be 1"
        print(f"  Box: {q.review_box}, Attempts: {q.attempts}, Correct: {q.correct_answers}")

        # Test is_due_for_review
        assert q.is_due_for_review(), "Should be due immediately after reset"
        print("\nDue for review check: PASS")

        print("\nQuestion model tests PASSED!")
        return True

    except Exception as e:
        print(f"\nQuestion model test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_knowledge_engine_basic():
    """Test KnowledgeEngine basic functionality."""
    print("\n" + "=" * 60)
    print("Testing KnowledgeEngine Basic Operations")
    print("=" * 60)

    try:
        from aegis_isle.interview.knowledge_engine import KnowledgeEngine

        # Create test database
        test_db = Path("data/test_standalone_db.json")
        test_db.parent.mkdir(exist_ok=True)

        # Initialize engine
        engine = KnowledgeEngine(db_path=test_db)
        print(f"Initialized KnowledgeEngine with {len(engine.questions)} questions")

        # Add manual questions
        print("\nAdding manual questions...")
        q1 = engine.add_question(
            content="Explain the difference between processes and threads",
            answer_key="Processes have separate memory space; threads share memory",
            difficulty=3,
            category="operating_systems",
            tags=["concurrency", "processes", "threads"]
        )
        print(f"  Added: {q1.content[:50]}...")

        q2 = engine.add_question(
            content="What is Big O notation?",
            answer_key="Mathematical notation to describe algorithm complexity",
            difficulty=2,
            category="algorithms",
            tags=["complexity", "theory"]
        )
        print(f"  Added: {q2.content[:50]}...")

        q3 = engine.add_question(
            content="Design a distributed cache system",
            answer_key="Consider consistency, partitioning, replication",
            difficulty=5,
            category="system_design",
            tags=["distributed_systems", "caching"]
        )
        print(f"  Added: {q3.content[:50]}...")

        assert len(engine.questions) == 3, "Should have 3 questions"

        # Test get_next_question (should return easiest new question)
        print("\nTesting get_next_question...")
        next_q = engine.get_next_question()
        assert next_q is not None, "Should return a question"
        assert next_q.difficulty == 2, "Should return easiest question first"
        print(f"  Got: {next_q.content[:50]}... (difficulty={next_q.difficulty})")

        # Test update_progress - correct
        print("\nTesting update_progress (correct)...")
        success = engine.update_progress(next_q.id, is_correct=True)
        assert success, "Update should succeed"
        updated_q = engine.questions[next_q.id]
        assert updated_q.review_box == 1, "Box should be 1"
        print(f"  Updated: box={updated_q.review_box}")

        # Test get_next_question again (should skip the one we just answered)
        print("\nGetting next question again...")
        next_q2 = engine.get_next_question()
        assert next_q2.id != next_q.id, "Should get different question"
        print(f"  Got: {next_q2.content[:50]}...")

        # Test update_progress - incorrect
        print("\nTesting update_progress (incorrect)...")
        engine.update_progress(next_q2.id, is_correct=False)
        updated_q2 = engine.questions[next_q2.id]
        assert updated_q2.review_box == 0, "Box should reset to 0"
        print(f"  Updated: box={updated_q2.review_box}")

        # Test search
        print("\nTesting search...")
        results = engine.search_questions("cache")
        assert len(results) > 0, "Should find cache question"
        print(f"  Found {len(results)} results for 'cache'")

        # Test filter by category
        print("\nTesting category filter...")
        algo_questions = engine.get_questions_by_category("algorithms")
        assert len(algo_questions) == 1, "Should have 1 algorithm question"
        print(f"  Found {len(algo_questions)} algorithm questions")

        # Test filter by difficulty
        print("\nTesting difficulty filter...")
        hard_questions = engine.get_questions_by_difficulty(5)
        assert len(hard_questions) == 1, "Should have 1 very hard question"
        print(f"  Found {len(hard_questions)} very hard questions")

        # Test statistics
        print("\nTesting statistics...")
        stats = engine.get_progress_statistics()
        assert stats['total_questions'] == 3, "Should have 3 questions"
        print(f"  Total: {stats['total_questions']}")
        print(f"  Due: {stats['due_for_review']}")
        print(f"  Success Rate: {stats['overall_success_rate']:.2%}")

        # Test delete question
        print("\nTesting delete question...")
        deleted = engine.delete_question(q3.id)
        assert deleted, "Delete should succeed"
        assert len(engine.questions) == 2, "Should have 2 questions left"
        print(f"  Deleted question, {len(engine.questions)} remaining")

        print("\nKnowledgeEngine basic tests PASSED!")

        # Cleanup
        if test_db.exists():
            test_db.unlink()
            print(f"\nCleaned up test database")

        return True

    except Exception as e:
        print(f"\nKnowledgeEngine test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all standalone tests."""
    print("Interview Prep System - Standalone Test Suite")
    print("=" * 60)
    print("Testing core functionality without full project dependencies")
    print("=" * 60)

    results = []

    # Test persona manager
    if test_persona_functionality():
        results.append(("PersonaManager", True))
    else:
        results.append(("PersonaManager", False))

    # Test knowledge engine imports
    if test_knowledge_engine_imports():
        results.append(("KnowledgeEngine Imports", True))
    else:
        results.append(("KnowledgeEngine Imports", False))
        return False  # Can't continue without imports

    # Test question model
    if test_question_model():
        results.append(("Question Model", True))
    else:
        results.append(("Question Model", False))

    # Test knowledge engine basic operations
    if test_knowledge_engine_basic():
        results.append(("KnowledgeEngine Operations", True))
    else:
        results.append(("KnowledgeEngine Operations", False))

    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for test_name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"  {test_name}: {status}")

    all_passed = all(result[1] for result in results)
    print("\n" + "=" * 60)
    if all_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
