"""
Multi-Agent Orchestration System

This module provides the core infrastructure for coordinating multiple AI agents
in the AegisIsle RAG system. Enhanced with LangGraph-based orchestration and
LLM-powered semantic routing.
"""

from .base import BaseAgent, AgentConfig, AgentRole, AgentMessage, AgentResponse
from .router import (
    AgentRouter,
    LLMRouter,
    LLMRoutingStrategy,
    KeywordRoutingStrategy,
    create_llm_router
)
from .orchestrator import (
    AgentOrchestrator,
    LangGraphAgentOrchestrator,
    AgentState
)
from .memory import AgentMemory

__all__ = [
    # Base classes
    "BaseAgent",
    "AgentConfig",
    "AgentRole",
    "AgentMessage",
    "AgentResponse",

    # Routing
    "AgentRouter",
    "LLMRouter",
    "LLMRoutingStrategy",
    "KeywordRoutingStrategy",
    "create_llm_router",

    # Orchestration
    "AgentOrchestrator",
    "LangGraphAgentOrchestrator",
    "AgentState",

    # Memory
    "AgentMemory",
]