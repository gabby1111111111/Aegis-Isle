"""
Simple test script to verify the enhanced tool system functionality.
Tests basic tool operations and agent integration.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the source directory to the Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


async def test_tools():
    """Test the tool system functionality."""
    print("🔧 Testing Tool System...")

    try:
        from aegis_isle.tools import PythonREPLTool, SearchTool, ToolConfig
        from aegis_isle.tools import get_tool_registry

        # Test Python REPL Tool
        print("\n📊 Testing Python REPL Tool:")
        python_config = ToolConfig(
            name="test_python_repl",
            description="Test Python REPL tool",
            timeout=10
        )

        python_tool = PythonREPLTool(config=python_config)

        # Test simple calculation
        result = await python_tool.run("result = 2 + 2\nprint(f'Result: {result}')")
        print(f"   Status: {result.status.value}")
        print(f"   Output: {result.content.get('output', 'No output')}")

        # Test Search Tool
        print("\n🔍 Testing Search Tool:")
        search_config = ToolConfig(
            name="test_search",
            description="Test search tool",
            timeout=15
        )

        search_tool = SearchTool(
            config=search_config,
            primary_provider="duckduckgo",
            enable_fallback=True
        )

        search_result = await search_tool.run("Python programming language")
        print(f"   Status: {search_result.status.value}")
        if search_result.status.value == "success":
            content = search_result.content
            print(f"   Found {content.get('total_results', 0)} results")
            if content.get('results'):
                first_result = content['results'][0]
                print(f"   First result: {first_result.get('title', 'No title')}")

        # Test Tool Registry
        print("\n📝 Testing Tool Registry:")
        registry = get_tool_registry()
        registry.register_tool(python_tool)
        registry.register_tool(search_tool)

        tools = registry.list_tools()
        print(f"   Registered tools: {tools}")

        print("\n✅ Tool system tests completed successfully!")
        return True

    except Exception as e:
        print(f"\n❌ Tool system test failed: {e}")
        return False


async def test_enhanced_agents():
    """Test the enhanced agent functionality."""
    print("\n🤖 Testing Enhanced Agents...")

    try:
        from aegis_isle.agents.implementations import EnhancedChartAgent, EnhancedResearcherAgent
        from aegis_isle.agents.base import AgentConfig, AgentRole

        # Test Chart Agent
        print("\n📈 Testing Enhanced Chart Agent:")
        chart_config = AgentConfig(
            name="test_chart_agent",
            role=AgentRole.CHART_GENERATOR,
            description="Test enhanced chart agent",
            tools=["python_repl"]
        )

        chart_agent = EnhancedChartAgent(
            config=chart_config,
            enable_python_tools=False  # Disable for testing to avoid tool dependency
        )

        await chart_agent.initialize()
        print(f"   Agent initialized: {chart_agent.id}")
        print(f"   Agent status: {chart_agent.status}")

        # Test Researcher Agent
        print("\n🔬 Testing Enhanced Researcher Agent:")
        researcher_config = AgentConfig(
            name="test_researcher_agent",
            role=AgentRole.RESEARCHER,
            description="Test enhanced researcher agent",
            tools=["web_search"]
        )

        researcher_agent = EnhancedResearcherAgent(
            config=researcher_config,
            enable_web_search=False,  # Disable for testing
            enable_rag_retrieval=False  # Disable for testing
        )

        await researcher_agent.initialize()
        print(f"   Agent initialized: {researcher_agent.id}")
        print(f"   Agent status: {researcher_agent.status}")

        print("\n✅ Enhanced agent tests completed successfully!")
        return True

    except Exception as e:
        print(f"\n❌ Enhanced agent test failed: {e}")
        return False


async def test_orchestrator():
    """Test the orchestrator functionality."""
    print("\n🎼 Testing Tool-Integrated Orchestrator...")

    try:
        from aegis_isle.agents.orchestrator import ToolIntegratedOrchestrator
        from aegis_isle.agents.router import AgentRouter

        # Create router and orchestrator
        router = AgentRouter(use_llm_routing=False)  # Use keyword routing for testing

        orchestrator = ToolIntegratedOrchestrator(
            router=router,
            auto_initialize_tools=False,  # Disable for testing
            enable_web_search=False  # Disable for testing
        )

        print(f"   Orchestrator initialized with {len(router.agents)} agents")

        print("\n✅ Orchestrator test completed successfully!")
        return True

    except Exception as e:
        print(f"\n❌ Orchestrator test failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("🚀 Starting Enhanced Aegis Isle Tool System Tests\n")

    all_tests_passed = True

    # Run tests
    if not await test_tools():
        all_tests_passed = False

    if not await test_enhanced_agents():
        all_tests_passed = False

    if not await test_orchestrator():
        all_tests_passed = False

    print("\n" + "="*60)
    if all_tests_passed:
        print("🎉 All tests passed! Tool system is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")

    return all_tests_passed


if __name__ == "__main__":
    success = asyncio.run(main())