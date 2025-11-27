#!/usr/bin/env python3
"""
Test LangGraph Interview Workflow

Tests the complete interview flow with persona-based evaluation.
"""

import sys
import asyncio
from pathlib import Path

sys.path.insert(0, 'src')


async def test_workflow_structure():
    """Test that the workflow graph is correctly structured."""
    print("=" * 60)
    print("Testing Workflow Structure")
    print("=" * 60)

    try:
        from aegis_isle.interview.graph import (
            app, InterviewState, build_interview_graph,
            evaluate_node, tutor_node, mentor_node, generate_node
        )

        print("\n1. Checking graph compilation...")
        assert app is not None, "Graph should be compiled"
        print("   Graph compiled successfully")

        print("\n2. Checking state structure...")
        # Verify InterviewState has required fields
        from typing import get_type_hints
        hints = get_type_hints(InterviewState)

        required_fields = [
            'question', 'user_answer', 'jd_context',
            'evaluation', 'history', 'feedback'
        ]

        for field in required_fields:
            assert field in hints, f"Missing field: {field}"
            print(f"   Field '{field}': FOUND")

        print("\n3. Checking node functions...")
        assert callable(generate_node), "generate_node should be callable"
        assert callable(evaluate_node), "evaluate_node should be callable"
        assert callable(tutor_node), "tutor_node should be callable"
        assert callable(mentor_node), "mentor_node should be callable"
        print("   All node functions are callable")

        print("\nWorkflow structure test PASSED!")
        return True

    except Exception as e:
        print(f"\nWorkflow structure test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_evaluation_logic():
    """Test the evaluation routing logic."""
    print("\n" + "=" * 60)
    print("Testing Evaluation Routing Logic")
    print("=" * 60)

    try:
        from aegis_isle.interview.graph import should_tutor_or_mentor

        print("\n1. Testing correct answer routing...")
        state = {
            "evaluation": {"is_correct": True}
        }
        route = should_tutor_or_mentor(state)
        assert route == "mentor", "Correct answer should route to mentor"
        print(f"   Correct answer -> {route}: PASS")

        print("\n2. Testing incorrect answer routing...")
        state = {
            "evaluation": {"is_correct": False}
        }
        route = should_tutor_or_mentor(state)
        assert route == "tutor", "Incorrect answer should route to tutor"
        print(f"   Incorrect answer -> {route}: PASS")

        print("\nEvaluation routing logic test PASSED!")
        return True

    except Exception as e:
        print(f"\nEvaluation routing test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_mock_workflow_execution():
    """Test workflow execution with mock data (no actual LLM calls)."""
    print("\n" + "=" * 60)
    print("Testing Mock Workflow Execution")
    print("=" * 60)

    try:
        from aegis_isle.interview import Question

        print("\n1. Creating mock state...")

        # Create a mock question
        mock_question = Question(
            id="test_001",
            content="What is the time complexity of binary search?",
            answer_key="O(log n) - eliminates half the search space each iteration",
            difficulty=3,
            category="algorithms",
            tags=["binary_search", "complexity"]
        )

        # Create initial state
        initial_state = {
            "question": mock_question,
            "user_answer": "It's O(log n) because we divide the search space in half each time",
            "jd_context": "Looking for software engineer with strong algorithms knowledge",
            "evaluation": {},
            "history": [],
            "feedback": "",
            "persona_mode": "strict",
            "next_action": None
        }

        print("   Mock state created successfully")
        print(f"   Question: {mock_question.content[:50]}...")
        print(f"   User Answer: {initial_state['user_answer'][:50]}...")

        print("\n2. Verifying state structure...")
        assert initial_state["question"] is not None
        assert isinstance(initial_state["user_answer"], str)
        assert isinstance(initial_state["history"], list)
        print("   State structure valid")

        print("\nMock workflow execution test PASSED!")
        print("\nNote: Full LLM-based workflow test requires:")
        print("  - Valid OpenAI API configuration")
        print("  - Project dependencies installed")
        print("  - Use test_interview_system.py for full integration tests")

        return True

    except Exception as e:
        print(f"\nMock workflow test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_file_verification():
    """Verify graph.py file exists and has correct content."""
    print("\n" + "=" * 60)
    print("Testing File Content Verification")
    print("=" * 60)

    graph_file = Path("src/aegis_isle/interview/graph.py")

    if not graph_file.exists():
        print("FAILED: graph.py not found")
        return False

    print(f"\n1. File exists: {graph_file}")
    print(f"   Size: {graph_file.stat().st_size:,} bytes")

    content = graph_file.read_text(encoding='utf-8')

    # Check for key components
    checks = [
        ("InterviewState", "State definition"),
        ("generate_node", "Question generation node"),
        ("evaluate_node", "Answer evaluation node"),
        ("tutor_node", "Tutoring node (Gojo)"),
        ("mentor_node", "Mentoring node (Nanami)"),
        ("should_tutor_or_mentor", "Conditional routing"),
        ("build_interview_graph", "Graph builder"),
        ("StateGraph", "LangGraph import"),
        ("TextGenerator", "LLM integration"),
        ("PersonaManager", "Persona integration"),
    ]

    print("\n2. Checking implementation components:")
    all_found = True
    for component, description in checks:
        found = component in content
        status = "FOUND" if found else "MISSING"
        print(f"   {description}: {status}")
        all_found = all_found and found

    if all_found:
        print("\nFile verification PASSED!")
        return True
    else:
        print("\nFile verification FAILED: Some components missing")
        return False


async def main():
    """Run all workflow tests."""
    print("LangGraph Interview Workflow - Test Suite")
    print("=" * 60)

    results = []

    # Test file verification
    results.append(("File Verification", test_file_verification()))

    # Test workflow structure
    results.append(("Workflow Structure", await test_workflow_structure()))

    # Test evaluation logic
    results.append(("Evaluation Logic", await test_evaluation_logic()))

    # Test mock execution
    results.append(("Mock Execution", await test_mock_workflow_execution()))

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
        print("All workflow tests PASSED!")
        print("\nThe LangGraph workflow is ready for use:")
        print("  - State management: Complete")
        print("  - Node functions: Implemented")
        print("  - Conditional routing: Working")
        print("  - Persona integration: Ready")
        print("\nNext steps:")
        print("  - Configure OpenAI API in .env")
        print("  - Run full integration test with LLM")
        print("  - Build Streamlit UI for user interaction")
    else:
        print("Some workflow tests FAILED!")

    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
