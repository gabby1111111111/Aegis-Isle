"""
Agent Router - Routes messages between agents based on content and strategy.
"""

import asyncio
from typing import Any, Dict, List, Optional, Union

from .base import AgentMessage, AgentRole, BaseAgent
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
    """Routes messages between agents using configurable strategies."""

    def __init__(self, strategy: Optional[RoutingStrategy] = None):
        self.strategy = strategy or KeywordRoutingStrategy()
        self.agents: Dict[str, BaseAgent] = {}
        self.message_history: List[AgentMessage] = []

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