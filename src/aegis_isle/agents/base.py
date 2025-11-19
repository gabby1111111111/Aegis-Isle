"""
Base agent classes and configurations for the multi-agent system.
"""

import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from ..core.logging import logger


class AgentRole(Enum):
    """Enumeration of agent roles in the system."""

    RESEARCHER = "researcher"
    RETRIEVER = "retriever"
    SUMMARIZER = "summarizer"
    CHART_GENERATOR = "chart_generator"
    TOOL_CALLER = "tool_caller"
    COORDINATOR = "coordinator"
    ROUTER = "router"


class AgentConfig(BaseModel):
    """Configuration for an agent."""

    name: str
    role: AgentRole
    description: str
    max_iterations: int = Field(default=10, gt=0)
    timeout: int = Field(default=300, gt=0)  # seconds
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1000, gt=0)
    model_name: Optional[str] = None
    tools: List[str] = Field(default_factory=list)
    prompt_template: Optional[str] = None
    enabled: bool = True
    memory_enabled: bool = True


class AgentMessage(BaseModel):
    """Message between agents."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str
    receiver_id: Optional[str] = None  # None for broadcast
    content: str
    message_type: str = "text"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)


class AgentResponse(BaseModel):
    """Response from an agent."""

    agent_id: str
    content: str
    success: bool
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    execution_time: float = 0.0
    token_usage: Dict[str, int] = Field(default_factory=dict)


class BaseAgent(ABC):
    """Base class for all agents in the system."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.id = f"{config.role.value}_{uuid.uuid4().hex[:8]}"
        self.created_at = datetime.now()
        self.status = "initialized"
        self._memory: List[AgentMessage] = []

        logger.info(f"Initialized agent {self.id} with role {config.role.value}")

    @property
    def name(self) -> str:
        """Get agent name."""
        return self.config.name

    @property
    def role(self) -> AgentRole:
        """Get agent role."""
        return self.config.role

    @abstractmethod
    async def process(self, message: Union[str, AgentMessage]) -> AgentResponse:
        """Process a message and return a response."""
        pass

    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the agent resources."""
        pass

    @abstractmethod
    async def cleanup(self) -> bool:
        """Clean up agent resources."""
        pass

    def add_to_memory(self, message: AgentMessage) -> None:
        """Add a message to the agent's memory."""
        if self.config.memory_enabled:
            self._memory.append(message)
            # Keep only the last 100 messages to avoid memory overflow
            if len(self._memory) > 100:
                self._memory = self._memory[-100:]

    def get_memory(self, limit: int = 10) -> List[AgentMessage]:
        """Get recent messages from memory."""
        return self._memory[-limit:] if self._memory else []

    def clear_memory(self) -> None:
        """Clear the agent's memory."""
        self._memory.clear()
        logger.debug(f"Cleared memory for agent {self.id}")

    def update_status(self, status: str) -> None:
        """Update agent status."""
        old_status = self.status
        self.status = status
        logger.debug(f"Agent {self.id} status changed from {old_status} to {status}")

    def get_info(self) -> Dict[str, Any]:
        """Get agent information."""
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role.value,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "config": self.config.dict(),
            "memory_size": len(self._memory) if self.config.memory_enabled else 0,
        }