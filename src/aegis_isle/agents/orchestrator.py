"""
Agent Orchestrator - Coordinates complex workflows between multiple agents using LangGraph.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Union, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langchain_core.runnables import RunnableLambda

from .base import AgentMessage, AgentResponse, BaseAgent, AgentRole
from .router import AgentRouter
from .implementations import EnhancedChartAgent, EnhancedResearcherAgent
from ..tools import get_tool_registry, PythonREPLTool, SearchTool, ToolConfig
from ..core.logging import logger


class AgentState(TypedDict):
    """Global state class for LangGraph multi-agent orchestration.

    Contains all shared information between agents during workflow execution.
    """
    messages: List[BaseMessage]
    context: Dict[str, Any]
    next_step: Optional[str]
    current_query: str
    agent_results: Dict[str, Any]
    final_answer: Optional[str]
    execution_metadata: Dict[str, Any]


class WorkflowStep:
    """Legacy WorkflowStep class for backward compatibility."""

    def __init__(self, name: str, agent_roles: List[str], input_message: str, **kwargs):
        """Initialize legacy workflow step."""
        self.name = name
        self.agent_roles = agent_roles
        self.input_message = input_message
        logger.warning(f"WorkflowStep '{name}' created - consider migrating to LangGraph")


class Workflow:
    """Legacy Workflow class for backward compatibility."""

    def __init__(self, name: str, description: str = ""):
        """Initialize legacy workflow."""
        self.name = name
        self.description = description
        logger.warning(f"Workflow '{name}' created - consider migrating to LangGraph")

    def add_step(self, step: WorkflowStep) -> None:
        """Legacy add_step method."""
        pass


class ToolIntegratedOrchestrator:
    """Enhanced LangGraph orchestrator with integrated tool system.

    Provides advanced multi-agent coordination with:
    - Automatic tool registration and initialization
    - Enhanced agent implementations with tool support
    - Advanced routing with tool-aware decision making
    - Performance monitoring and error handling
    """

    def __init__(
        self,
        router: Optional[AgentRouter] = None,
        auto_initialize_tools: bool = True,
        enable_web_search: bool = True,
        search_providers_config: Optional[Dict[str, Dict[str, str]]] = None
    ):
        """Initialize the tool-integrated orchestrator.

        Args:
            router: Optional router (creates default if None)
            auto_initialize_tools: Whether to automatically initialize tools
            enable_web_search: Whether to enable web search for researcher agent
            search_providers_config: Configuration for search providers
        """
        self.router = router or AgentRouter(use_llm_routing=True)
        self.enable_web_search = enable_web_search
        self.search_providers_config = search_providers_config or {}
        self.graph: Optional[CompiledStateGraph] = None
        self.tool_registry = get_tool_registry()

        # Initialize tools and enhanced agents
        if auto_initialize_tools:
            self._initialize_tools()
        self._initialize_enhanced_agents()
        self._build_graph()

        logger.info("Initialized Tool-Integrated Orchestrator with enhanced agents")

    def _initialize_tools(self) -> None:
        """Initialize and register tools in the tool registry."""
        try:
            # Initialize Python REPL tool
            python_config = ToolConfig(
                name="orchestrator_python_repl",
                description="Python REPL for code execution and data analysis",
                timeout=60,
                max_retries=2
            )
            python_tool = PythonREPLTool(config=python_config)
            self.tool_registry.register_tool(python_tool)

            # Initialize search tool if enabled
            if self.enable_web_search:
                search_config = ToolConfig(
                    name="orchestrator_web_search",
                    description="Web search for real-time information retrieval",
                    timeout=30,
                    max_retries=3
                )
                search_tool = SearchTool(
                    config=search_config,
                    primary_provider="duckduckgo",
                    providers_config=self.search_providers_config,
                    enable_fallback=True
                )
                self.tool_registry.register_tool(search_tool)

            logger.info(f"Initialized {len(self.tool_registry.list_tools())} tools")

        except Exception as e:
            logger.error(f"Failed to initialize tools: {e}")

    def _initialize_enhanced_agents(self) -> None:
        """Initialize enhanced agents with tool integration."""
        try:
            # Create enhanced chart agent
            chart_config = AgentConfig(
                name="enhanced_chart_agent",
                role=AgentRole.CHART_GENERATOR,
                description="Enhanced chart generation with Python tool integration",
                temperature=0.3,
                tools=["python_repl"]
            )
            chart_agent = EnhancedChartAgent(
                config=chart_config,
                enable_python_tools=True
            )
            self.router.register_agent(chart_agent)

            # Create enhanced researcher agent
            researcher_config = AgentConfig(
                name="enhanced_researcher_agent",
                role=AgentRole.RESEARCHER,
                description="Enhanced research with web search and RAG integration",
                temperature=0.7,
                tools=["web_search", "rag_retrieval"] if self.enable_web_search else ["rag_retrieval"]
            )
            researcher_agent = EnhancedResearcherAgent(
                config=researcher_config,
                enable_web_search=self.enable_web_search,
                search_providers_config=self.search_providers_config
            )
            self.router.register_agent(researcher_agent)

            logger.info("Initialized enhanced agents with tool integration")

        except Exception as e:
            logger.error(f"Failed to initialize enhanced agents: {e}")

    def _build_graph(self) -> None:
        """Build the LangGraph StateGraph with enhanced multi-agent workflow."""
        # Create the StateGraph
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("router", self._router_node)
        workflow.add_node("researcher", self._researcher_node)
        workflow.add_node("retriever", self._retriever_node)
        workflow.add_node("summarizer", self._summarizer_node)
        workflow.add_node("chart_generator", self._chart_generator_node)
        workflow.add_node("finalizer", self._finalizer_node)

        # Set entry point
        workflow.set_entry_point("router")

        # Add conditional routing from router to appropriate agents
        workflow.add_conditional_edges(
            "router",
            self._route_to_agents,
            {
                "researcher": "researcher",
                "retriever": "retriever",
                "summarizer": "summarizer",
                "chart_generator": "chart_generator",
                "finalizer": "finalizer"
            }
        )

        # All agent nodes flow to finalizer
        for agent_node in ["researcher", "retriever", "summarizer", "chart_generator"]:
            workflow.add_edge(agent_node, "finalizer")

        # Finalizer ends the workflow
        workflow.add_edge("finalizer", END)

        # Compile the graph
        self.graph = workflow.compile()
        logger.info("Built enhanced LangGraph workflow with tool integration")

    async def initialize_all(self) -> Dict[str, bool]:
        """Initialize all tools and agents.

        Returns:
            Dictionary mapping component names to initialization success
        """
        results = {}

        # Initialize tools
        tool_results = await self.tool_registry.initialize_all()
        results.update({f"tool_{name}": success for name, success in tool_results.items()})

        # Initialize agents
        agent_results = {}
        for agent_id, agent in self.router.agents.items():
            try:
                success = await agent.initialize()
                agent_results[agent_id] = success
            except Exception as e:
                logger.error(f"Failed to initialize agent {agent_id}: {e}")
                agent_results[agent_id] = False

        results.update({f"agent_{name}": success for name, success in agent_results.items()})

        return results

    async def cleanup_all(self) -> Dict[str, bool]:
        """Clean up all tools and agents.

        Returns:
            Dictionary mapping component names to cleanup success
        """
        results = {}

        # Cleanup agents
        agent_results = {}
        for agent_id, agent in self.router.agents.items():
            try:
                success = await agent.cleanup()
                agent_results[agent_id] = success
            except Exception as e:
                logger.error(f"Failed to cleanup agent {agent_id}: {e}")
                agent_results[agent_id] = False

        results.update({f"agent_{name}": success for name, success in agent_results.items()})

        # Cleanup tools
        tool_results = await self.tool_registry.cleanup_all()
        results.update({f"tool_{name}": success for name, success in tool_results.items()})

        return results


class LangGraphAgentOrchestrator(ToolIntegratedOrchestrator):
    """Enhanced Agent Orchestrator using LangGraph for state-based multi-agent coordination.

    Now inherits from ToolIntegratedOrchestrator to provide tool integration
    while maintaining backward compatibility.
    """

    def __init__(self, router: AgentRouter, enable_tool_integration: bool = True):
        """Initialize the LangGraph-based orchestrator.

        Args:
            router: The agent router for agent management and routing logic
            enable_tool_integration: Whether to enable tool integration features
        """
        if enable_tool_integration:
            super().__init__(router=router)
        else:
            # Legacy mode without tool integration
            self.router = router
            self.graph: Optional[CompiledStateGraph] = None
            self._build_legacy_graph()

        logger.info("Initialized LangGraph Agent Orchestrator")

    def _build_legacy_graph(self) -> None:
        """Build legacy graph without tool integration."""
        # Create the StateGraph
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("router", self._router_node)
        workflow.add_node("researcher", self._researcher_node)
        workflow.add_node("retriever", self._retriever_node)
        workflow.add_node("summarizer", self._summarizer_node)
        workflow.add_node("chart_generator", self._chart_generator_node)
        workflow.add_node("finalizer", self._finalizer_node)

        # Set entry point
        workflow.set_entry_point("router")

        # Add conditional routing from router to appropriate agents
        workflow.add_conditional_edges(
            "router",
            self._route_to_agents,
            {
                "researcher": "researcher",
                "retriever": "retriever",
                "summarizer": "summarizer",
                "chart_generator": "chart_generator",
                "finalizer": "finalizer"
            }
        )

        # All agent nodes flow to finalizer
        for agent_node in ["researcher", "retriever", "summarizer", "chart_generator"]:
            workflow.add_edge(agent_node, "finalizer")

        # Finalizer ends the workflow
        workflow.add_edge("finalizer", END)

        # Compile the graph
        self.graph = workflow.compile()
        logger.info("Built legacy LangGraph workflow")

    async def _router_node(self, state: AgentState) -> AgentState:
        """Router node that determines which agent should handle the request.

        Args:
            state: Current workflow state

        Returns:
            Updated state with routing decision
        """
        try:
            query = state["current_query"]
            logger.info(f"Router processing query: {query[:100]}...")

            # Use the router to determine target agents
            message = AgentMessage(sender_id="orchestrator", content=query)
            target_agents = await self.router.route_message(message)

            # Determine the primary agent based on routing logic
            if target_agents:
                # Map agent IDs to node names
                agent_mapping = {
                    "researcher": "researcher",
                    "retriever": "retriever",
                    "summarizer": "summarizer",
                    "chart_generator": "chart_generator"
                }

                # Find first matching agent type
                for agent_id in target_agents:
                    agent = self.router.agents.get(agent_id)
                    if agent and agent.role.value in agent_mapping:
                        state["next_step"] = agent_mapping[agent.role.value]
                        break
                else:
                    # Default to retriever if no specific match
                    state["next_step"] = "retriever"
            else:
                # Default to finalizer if no agents available
                state["next_step"] = "finalizer"

            state["messages"].append(SystemMessage(
                content=f"Router decision: Route to {state['next_step']}"
            ))

            return state

        except Exception as e:
            logger.error(f"Router node error: {e}")
            state["next_step"] = "finalizer"
            state["context"]["error"] = str(e)
            return state

    def _route_to_agents(self, state: AgentState) -> str:
        """Conditional routing function to determine next node.

        Args:
            state: Current workflow state

        Returns:
            Name of the next node to execute
        """
        return state.get("next_step", "finalizer")

    async def _researcher_node(self, state: AgentState) -> AgentState:
        """Researcher agent node for information research and analysis.

        Args:
            state: Current workflow state

        Returns:
            Updated state with research results
        """
        return await self._execute_agent_node(state, AgentRole.RESEARCHER, "researcher")

    async def _retriever_node(self, state: AgentState) -> AgentState:
        """Retriever agent node for document retrieval.

        Args:
            state: Current workflow state

        Returns:
            Updated state with retrieval results
        """
        return await self._execute_agent_node(state, AgentRole.RETRIEVER, "retriever")

    async def _summarizer_node(self, state: AgentState) -> AgentState:
        """Summarizer agent node for content summarization.

        Args:
            state: Current workflow state

        Returns:
            Updated state with summarization results
        """
        return await self._execute_agent_node(state, AgentRole.SUMMARIZER, "summarizer")

    async def _chart_generator_node(self, state: AgentState) -> AgentState:
        """Chart generator agent node for visualization creation.

        Args:
            state: Current workflow state

        Returns:
            Updated state with chart generation results
        """
        return await self._execute_agent_node(state, AgentRole.CHART_GENERATOR, "chart_generator")

    async def _execute_agent_node(
        self,
        state: AgentState,
        agent_role: AgentRole,
        node_name: str
    ) -> AgentState:
        """Execute a specific agent node with error handling.

        Args:
            state: Current workflow state
            agent_role: Role of the agent to execute
            node_name: Name of the current node for logging

        Returns:
            Updated state with agent execution results
        """
        try:
            logger.info(f"Executing {node_name} node")
            start_time = time.time()

            # Find agents with the specified role
            target_agents = [
                agent_id for agent_id, agent in self.router.agents.items()
                if agent.role == agent_role and agent.config.enabled
            ]

            if not target_agents:
                logger.warning(f"No available {agent_role.value} agents found")
                state["agent_results"][node_name] = {
                    "error": f"No {agent_role.value} agents available",
                    "execution_time": 0.0
                }
                return state

            # Execute agent
            message = AgentMessage(
                sender_id="orchestrator",
                content=state["current_query"]
            )

            results = await self.router.send_to_agents(message, target_agents[:1])  # Use first agent
            execution_time = time.time() - start_time

            # Store results in state
            state["agent_results"][node_name] = {
                "results": results,
                "execution_time": execution_time,
                "agent_count": len(target_agents)
            }

            # Add message to conversation
            if results and len(results) > 0:
                agent_result = list(results.values())[0]
                if agent_result.get("success"):
                    content = agent_result.get("content", "")
                    state["messages"].append(AIMessage(
                        content=f"{agent_role.value} result: {content}"
                    ))

            logger.info(f"Completed {node_name} node in {execution_time:.2f}s")
            return state

        except Exception as e:
            logger.error(f"{node_name} node execution failed: {e}")
            state["agent_results"][node_name] = {
                "error": str(e),
                "execution_time": time.time() - start_time if 'start_time' in locals() else 0.0
            }
            return state

    async def _finalizer_node(self, state: AgentState) -> AgentState:
        """Finalizer node that compiles results and generates final answer.

        Args:
            state: Current workflow state with all agent results

        Returns:
            Updated state with final answer
        """
        try:
            logger.info("Executing finalizer node")
            start_time = time.time()

            # Compile results from all agents
            all_results = []
            for agent_name, result in state["agent_results"].items():
                if "results" in result:
                    agent_results = result["results"]
                    for agent_id, agent_response in agent_results.items():
                        if agent_response.get("success") and agent_response.get("content"):
                            all_results.append(f"{agent_name}: {agent_response['content']}")

            # Generate final consolidated answer
            if all_results:
                final_answer = "Based on multi-agent analysis:\n\n" + "\n\n".join(all_results)
            else:
                final_answer = "Unable to generate response - no successful agent results"

            state["final_answer"] = final_answer
            state["execution_metadata"]["finalization_time"] = time.time() - start_time
            state["execution_metadata"]["total_agents"] = len(state["agent_results"])

            # Add final message
            state["messages"].append(AIMessage(content=final_answer))

            logger.info("Finalization completed successfully")
            return state

        except Exception as e:
            logger.error(f"Finalizer node failed: {e}")
            state["final_answer"] = f"Finalization error: {str(e)}"
            state["execution_metadata"]["error"] = str(e)
            return state

    async def execute_workflow(
        self,
        query: str,
        initial_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute the LangGraph workflow for a given query.

        Args:
            query: User query to process
            initial_context: Optional initial context

        Returns:
            Dictionary containing execution results and metadata
        """
        if not self.graph:
            raise RuntimeError("Graph not properly initialized")

        logger.info(f"Starting LangGraph workflow execution for query: {query[:100]}...")
        start_time = time.time()

        try:
            # Initialize state
            initial_state: AgentState = {
                "messages": [HumanMessage(content=query)],
                "context": initial_context or {},
                "next_step": None,
                "current_query": query,
                "agent_results": {},
                "final_answer": None,
                "execution_metadata": {
                    "start_time": start_time,
                    "workflow_type": "langgraph_multiagent"
                }
            }

            # Execute the graph
            final_state = await self.graph.ainvoke(initial_state)

            execution_time = time.time() - start_time
            final_state["execution_metadata"]["total_time"] = execution_time

            logger.info(f"LangGraph workflow completed in {execution_time:.2f}s")

            return {
                "success": True,
                "final_answer": final_state.get("final_answer"),
                "agent_results": final_state.get("agent_results", {}),
                "messages": [msg.content for msg in final_state.get("messages", [])],
                "execution_time": execution_time,
                "metadata": final_state.get("execution_metadata", {})
            }

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"LangGraph workflow failed after {execution_time:.2f}s: {e}")

            return {
                "success": False,
                "error": str(e),
                "execution_time": execution_time,
                "metadata": {"error": str(e)}
            }


# Legacy compatibility class - maintains backward compatibility
class AgentOrchestrator(LangGraphAgentOrchestrator):
    """Legacy wrapper for LangGraphAgentOrchestrator to maintain API compatibility.

    Inherits all LangGraph functionality while maintaining the original interface.
    """

    def __init__(self, router: AgentRouter):
        """Initialize with LangGraph orchestrator as base."""
        super().__init__(router)
        self.active_workflows: Dict[str, Any] = {}  # For compatibility
        self.workflow_templates: Dict[str, Any] = {}  # For compatibility

        # Create basic workflow templates for backward compatibility
        self._create_legacy_templates()

    def _create_legacy_templates(self) -> None:
        """Create legacy workflow templates for compatibility."""
        # Create a basic RAG workflow template
        self.workflow_templates["rag_query"] = {
            "name": "rag_query",
            "description": "Standard RAG query processing workflow",
            "type": "langgraph_multiagent"
        }

    async def execute_workflow(
        self,
        workflow_name: str = "rag_query",
        initial_input: Union[str, Dict[str, Any]] = "",
        workflow_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute workflow with legacy compatibility.

        Args:
            workflow_name: Name of workflow template (now unused, all use LangGraph)
            initial_input: Query string or input dict
            workflow_id: Optional workflow ID for tracking

        Returns:
            Workflow execution results
        """
        # Convert input to query string
        if isinstance(initial_input, dict):
            query = initial_input.get("query", str(initial_input))
        else:
            query = str(initial_input)

        # Use the new LangGraph execution
        result = await super().execute_workflow(
            query=query,
            initial_context=initial_input if isinstance(initial_input, dict) else {}
        )

        # Add legacy workflow fields
        result["workflow_id"] = workflow_id or f"langgraph_{int(time.time())}"

        return result

    def register_workflow_template(self, workflow: Any) -> None:
        """Register workflow template (legacy compatibility - now no-op)."""
        logger.info(f"Legacy workflow registration for '{workflow}' - using LangGraph instead")

    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow status (legacy compatibility - simplified)."""
        return {
            "workflow_id": workflow_id,
            "name": "langgraph_workflow",
            "description": "LangGraph multi-agent workflow",
            "total_steps": 4,  # router, agent, finalizer, end
            "completed_steps": 4,
            "failed_steps": 0,
            "is_complete": True,
            "has_errors": False
        }

    def create_rag_workflow(self) -> Dict[str, Any]:
        """Create RAG workflow (legacy compatibility)."""
        return {
            "name": "rag_query",
            "description": "LangGraph-based RAG workflow",
            "type": "langgraph_multiagent"
        }