"""
Agent Router - Routes messages between agents based on content and strategy.
Enhanced with LLMRouter for semantic routing using LLM intent analysis.
"""

import asyncio
import json
from typing import Any, Dict, List, Optional, Union

from .base import AgentMessage, AgentRole, BaseAgent
from ..rag.generator import get_generator, GenerationResult
from ..core.logging import logger


class RoutingStrategy:
    """Base class for routing strategies."""

    async def route(
        self,
        message: AgentMessage,
        agents: Dict[str, BaseAgent],
        context: Dict[str, Any]
    ) -> List[str]:
        """Route a message to appropriate agents."""
        raise NotImplementedError


class LLMRoutingStrategy(RoutingStrategy):
    """LLM-based semantic routing strategy that uses language model to analyze user intent.

    Replaces simple keyword matching with intelligent intent understanding.
    Uses the existing LLM generator interface to determine appropriate agent routing.
    """

    def __init__(self, provider: str = "openai", model: str = "gpt-4"):
        """Initialize LLM routing strategy.

        Args:
            provider: LLM provider (openai, anthropic, etc.)
            model: Model name to use for intent analysis
        """
        self.provider = provider
        self.model = model
        self.generator = get_generator(provider=provider, model=model)

        # Agent role descriptions for context
        self.agent_descriptions = {
            AgentRole.RESEARCHER: "Conducts research, searches for information, investigates topics, and explores complex subjects",
            AgentRole.RETRIEVER: "Retrieves and fetches documents, loads knowledge from databases, and accesses stored information",
            AgentRole.SUMMARIZER: "Summarizes content, creates abstracts, provides overviews, and digests information",
            AgentRole.CHART_GENERATOR: "Creates charts, graphs, plots, visualizations, diagrams, and figures",
            AgentRole.TOOL_CALLER: "Executes functions, runs tools, calls APIs, and performs specific operations",
            AgentRole.COORDINATOR: "Coordinates tasks, manages workflows, and provides general assistance"
        }

        logger.info(f"Initialized LLM routing strategy with {provider} {model}")

    async def route(
        self,
        message: AgentMessage,
        agents: Dict[str, BaseAgent],
        context: Dict[str, Any]
    ) -> List[str]:
        """Route message using LLM intent analysis.

        Args:
            message: The message to route
            agents: Available agents
            context: Additional context

        Returns:
            List of target agent IDs
        """
        try:
            # Get available agent roles
            available_roles = [agent.role for agent in agents.values() if agent.config.enabled]
            if not available_roles:
                logger.warning("No enabled agents available for routing")
                return []

            # Build LLM prompt for intent analysis
            prompt = self._build_routing_prompt(message.content, available_roles)

            # Get LLM routing decision
            result = await self.generator.generate(prompt)
            routing_decision = self._parse_routing_response(result.generated_text)

            # Map to actual agent IDs
            target_agents = self._map_to_agent_ids(routing_decision, agents)

            logger.info(f"LLM router decision: {routing_decision.get('target_agent', 'unknown')} - "
                       f"Reason: {routing_decision.get('reason', 'No reason provided')}")

            return target_agents

        except Exception as e:
            logger.error(f"LLM routing failed: {e}")
            # Fallback to keyword-based routing
            fallback_strategy = KeywordRoutingStrategy()
            return await fallback_strategy.route(message, agents, context)

    def _build_routing_prompt(self, user_message: str, available_roles: List[AgentRole]) -> str:
        """Build the routing prompt for LLM analysis.

        Args:
            user_message: The user's message
            available_roles: List of available agent roles

        Returns:
            Formatted prompt for LLM
        """
        # Create role descriptions for available agents
        role_descriptions = []
        for role in available_roles:
            description = self.agent_descriptions.get(role, f"Handles {role.value} tasks")
            role_descriptions.append(f"- {role.value}: {description}")

        prompt = f"""You are an intelligent routing system for a multi-agent AI system.

Available agents and their capabilities:
{chr(10).join(role_descriptions)}

User message: "{user_message}"

Analyze the user's intent and determine which agent would be best suited to handle this request.

Respond with a JSON object in this format:
{{
    "target_agent": "agent_role_name",
    "reason": "brief explanation of why this agent is appropriate"
}}

Where target_agent should be one of: {", ".join([role.value for role in available_roles])}

If the request requires multiple agents or is unclear, choose the most appropriate primary agent.
If no agent seems suitable, choose "coordinator" as a fallback.

JSON Response:"""

        return prompt

    def _parse_routing_response(self, llm_response: str) -> Dict[str, str]:
        """Parse the LLM response to extract routing decision.

        Args:
            llm_response: Raw LLM response text

        Returns:
            Dictionary with target_agent and reason
        """
        try:
            # Clean the response and extract JSON
            response_text = llm_response.strip()

            # Find JSON object in response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1

            if start_idx >= 0 and end_idx > start_idx:
                json_text = response_text[start_idx:end_idx]
                routing_data = json.loads(json_text)

                return {
                    "target_agent": routing_data.get("target_agent", "coordinator"),
                    "reason": routing_data.get("reason", "LLM routing decision")
                }
            else:
                logger.warning(f"No valid JSON found in LLM response: {response_text}")
                return {"target_agent": "coordinator", "reason": "Failed to parse LLM response"}

        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing failed: {e}. Response: {llm_response}")
            return {"target_agent": "coordinator", "reason": "JSON parsing error"}
        except Exception as e:
            logger.error(f"Error parsing LLM routing response: {e}")
            return {"target_agent": "coordinator", "reason": "Parsing error"}

    def _map_to_agent_ids(self, routing_decision: Dict[str, str], agents: Dict[str, BaseAgent]) -> List[str]:
        """Map routing decision to actual agent IDs.

        Args:
            routing_decision: The LLM routing decision
            agents: Available agents

        Returns:
            List of matching agent IDs
        """
        target_role = routing_decision.get("target_agent", "coordinator")

        # Find agents with the target role
        matching_agents = []
        for agent_id, agent in agents.items():
            if agent.config.enabled and agent.role.value == target_role:
                matching_agents.append(agent_id)

        # If no exact match, try coordinator as fallback
        if not matching_agents and target_role != "coordinator":
            for agent_id, agent in agents.items():
                if agent.config.enabled and agent.role == AgentRole.COORDINATOR:
                    matching_agents.append(agent_id)

        # If still no match, return any available agent
        if not matching_agents:
            enabled_agents = [aid for aid, a in agents.items() if a.config.enabled]
            if enabled_agents:
                matching_agents = enabled_agents[:1]

        return matching_agents


class KeywordRoutingStrategy(RoutingStrategy):
    """Route based on keywords in the message."""

    def __init__(self):
        self.role_keywords = {
            AgentRole.RESEARCHER: [
                "research", "search", "find", "investigate", "explore", "study"
            ],
            AgentRole.RETRIEVER: [
                "retrieve", "fetch", "get", "load", "document", "knowledge"
            ],
            AgentRole.SUMMARIZER: [
                "summarize", "summary", "brief", "overview", "digest", "abstract"
            ],
            AgentRole.CHART_GENERATOR: [
                "chart", "graph", "plot", "visualization", "diagram", "figure"
            ],
            AgentRole.TOOL_CALLER: [
                "execute", "run", "tool", "function", "api", "call"
            ],
        }

    async def route(
        self,
        message: AgentMessage,
        agents: Dict[str, BaseAgent],
        context: Dict[str, Any]
    ) -> List[str]:
        """Route message based on keywords."""
        content = message.content.lower()
        target_agents = []

        for agent_id, agent in agents.items():
            if not agent.config.enabled:
                continue

            role_keywords = self.role_keywords.get(agent.role, [])
            if any(keyword in content for keyword in role_keywords):
                target_agents.append(agent_id)

        # If no specific agents found, route to coordinator
        if not target_agents:
            coordinator_agents = [
                agent_id for agent_id, agent in agents.items()
                if agent.role == AgentRole.COORDINATOR and agent.config.enabled
            ]
            target_agents.extend(coordinator_agents)

        logger.debug(f"Routed message to agents: {target_agents}")
        return target_agents


class PriorityRoutingStrategy(RoutingStrategy):
    """Route based on agent priority and availability."""

    def __init__(self):
        self.role_priority = {
            AgentRole.ROUTER: 1,
            AgentRole.COORDINATOR: 2,
            AgentRole.RETRIEVER: 3,
            AgentRole.RESEARCHER: 4,
            AgentRole.SUMMARIZER: 5,
            AgentRole.CHART_GENERATOR: 6,
            AgentRole.TOOL_CALLER: 7,
        }

    async def route(
        self,
        message: AgentMessage,
        agents: Dict[str, BaseAgent],
        context: Dict[str, Any]
    ) -> List[str]:
        """Route message based on priority."""
        available_agents = [
            (agent_id, agent) for agent_id, agent in agents.items()
            if agent.config.enabled and agent.status != "busy"
        ]

        # Sort by priority
        available_agents.sort(key=lambda x: self.role_priority.get(x[1].role, 99))

        # Return top 3 agents or all available if less than 3
        target_count = min(3, len(available_agents))
        target_agents = [agent_id for agent_id, _ in available_agents[:target_count]]

        return target_agents


class AgentRouter:
    """Routes messages between agents using configurable strategies.

    Enhanced with LLM-based semantic routing by default, with keyword fallback.
    """

    def __init__(self, strategy: Optional[RoutingStrategy] = None, use_llm_routing: bool = True):
        """Initialize the agent router.

        Args:
            strategy: Optional custom routing strategy
            use_llm_routing: Whether to use LLM routing by default (True for semantic routing)
        """
        if strategy is not None:
            self.strategy = strategy
        elif use_llm_routing:
            try:
                self.strategy = LLMRoutingStrategy()
                logger.info("Initialized router with LLM semantic routing")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM routing, falling back to keywords: {e}")
                self.strategy = KeywordRoutingStrategy()
        else:
            self.strategy = KeywordRoutingStrategy()

        self.agents: Dict[str, BaseAgent] = {}
        self.message_history: List[AgentMessage] = []

    def upgrade_to_llm_routing(self, provider: str = "openai", model: str = "gpt-4") -> bool:
        """Upgrade to LLM-based routing strategy.

        Args:
            provider: LLM provider to use
            model: Model name to use

        Returns:
            True if upgrade successful, False if fallback to keywords
        """
        try:
            self.strategy = LLMRoutingStrategy(provider=provider, model=model)
            logger.info(f"Upgraded to LLM routing with {provider} {model}")
            return True
        except Exception as e:
            logger.error(f"Failed to upgrade to LLM routing: {e}")
            return False

    def downgrade_to_keyword_routing(self) -> None:
        """Downgrade to keyword-based routing strategy."""
        self.strategy = KeywordRoutingStrategy()
        logger.info("Downgraded to keyword-based routing")

    def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent with the router."""
        self.agents[agent.id] = agent
        logger.info(f"Registered agent {agent.id} ({agent.role.value}) with router")

    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from the router."""
        if agent_id in self.agents:
            del self.agents[agent_id]
            logger.info(f"Unregistered agent {agent_id} from router")

    async def route_message(
        self,
        message: Union[str, AgentMessage],
        context: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Route a message to appropriate agents."""
        if isinstance(message, str):
            message = AgentMessage(
                sender_id="user",
                content=message
            )

        context = context or {}
        self.message_history.append(message)

        # Keep only the last 1000 messages
        if len(self.message_history) > 1000:
            self.message_history = self.message_history[-1000:]

        target_agents = await self.strategy.route(message, self.agents, context)

        logger.info(
            f"Routed message '{message.content[:100]}...' to {len(target_agents)} agents"
        )

        return target_agents

    async def broadcast_message(
        self,
        message: Union[str, AgentMessage],
        exclude_agents: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Broadcast a message to all registered agents."""
        if isinstance(message, str):
            message = AgentMessage(
                sender_id="router",
                content=message
            )

        exclude_agents = exclude_agents or []
        results = {}

        tasks = []
        for agent_id, agent in self.agents.items():
            if agent_id not in exclude_agents and agent.config.enabled:
                tasks.append(self._send_to_agent(agent, message))

        if tasks:
            responses = await asyncio.gather(*tasks, return_exceptions=True)

            for i, response in enumerate(responses):
                agent_id = list(self.agents.keys())[i]
                if isinstance(response, Exception):
                    results[agent_id] = {"error": str(response)}
                else:
                    results[agent_id] = response

        logger.info(f"Broadcast message to {len(results)} agents")
        return results

    async def send_to_agents(
        self,
        message: Union[str, AgentMessage],
        target_agents: List[str]
    ) -> Dict[str, Any]:
        """Send a message to specific agents."""
        if isinstance(message, str):
            message = AgentMessage(
                sender_id="router",
                content=message
            )

        results = {}
        tasks = []

        for agent_id in target_agents:
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                if agent.config.enabled:
                    tasks.append(self._send_to_agent(agent, message))

        if tasks:
            responses = await asyncio.gather(*tasks, return_exceptions=True)

            for i, response in enumerate(responses):
                agent_id = target_agents[i]
                if isinstance(response, Exception):
                    results[agent_id] = {"error": str(response)}
                else:
                    results[agent_id] = response

        return results

    async def _send_to_agent(self, agent: BaseAgent, message: AgentMessage) -> Dict[str, Any]:
        """Send a message to a specific agent."""
        try:
            agent.add_to_memory(message)
            response = await agent.process(message)
            return {
                "success": response.success,
                "content": response.content,
                "metadata": response.metadata,
                "execution_time": response.execution_time
            }
        except Exception as e:
            logger.error(f"Error sending message to agent {agent.id}: {e}")
            return {"error": str(e), "success": False}

    def get_agent_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all registered agents."""
        return {
            agent_id: agent.get_info()
            for agent_id, agent in self.agents.items()
        }

    def set_routing_strategy(self, strategy: RoutingStrategy) -> None:
        """Set a new routing strategy."""
        self.strategy = strategy
        logger.info(f"Updated routing strategy to {strategy.__class__.__name__}")


# Convenient alias for LLM-based routing
class LLMRouter(AgentRouter):
    """LLM-based Agent Router with semantic intent analysis.

    This is the enhanced router that uses LLM to understand user intent
    and route to appropriate agents instead of simple keyword matching.
    """

    def __init__(self, provider: str = "openai", model: str = "gpt-4"):
        """Initialize LLMRouter with specific LLM configuration.

        Args:
            provider: LLM provider (openai, anthropic, etc.)
            model: Model name for intent analysis
        """
        strategy = LLMRoutingStrategy(provider=provider, model=model)
        super().__init__(strategy=strategy, use_llm_routing=False)  # Skip auto-init since we provide strategy
        logger.info(f"Initialized LLMRouter with {provider} {model}")


def create_llm_router(provider: str = "openai", model: str = "gpt-4") -> LLMRouter:
    """Factory function to create an LLM-based router.

    Args:
        provider: LLM provider to use
        model: Model name to use

    Returns:
        Configured LLMRouter instance
    """
    return LLMRouter(provider=provider, model=model)