"""
Base tool interface for Aegis Isle agent tools.
Provides unified interface for tool execution and error handling.
"""

import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from ..core.logging import logger


class ToolStatus(Enum):
    """Tool execution status."""

    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class ToolResult(BaseModel):
    """Result of tool execution."""

    tool_name: str
    status: ToolStatus
    content: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)


class ToolError(Exception):
    """Base exception for tool-related errors."""

    def __init__(self, message: str, tool_name: str = "", details: Optional[Dict] = None):
        super().__init__(message)
        self.tool_name = tool_name
        self.details = details or {}


class ToolConfig(BaseModel):
    """Configuration for a tool."""

    name: str
    description: str
    timeout: int = Field(default=30, gt=0)  # seconds
    max_retries: int = Field(default=3, ge=0)
    enabled: bool = True
    rate_limit: Optional[int] = None  # requests per minute
    custom_params: Dict[str, Any] = Field(default_factory=dict)


class BaseTool(ABC):
    """Base class for all tools in the Aegis Isle system.

    Provides unified interface for tool execution with error handling,
    timeout management, and result standardization.
    """

    def __init__(self, config: ToolConfig):
        """Initialize the tool with configuration.

        Args:
            config: Tool configuration including name, description, and parameters
        """
        self.config = config
        self.id = f"{config.name}_{uuid.uuid4().hex[:8]}"
        self.created_at = datetime.now()
        self.execution_count = 0
        self.last_execution = None
        self._rate_limiter = {}  # Simple rate limiting tracking

        logger.info(f"Initialized tool {self.id}: {config.name}")

    @property
    def name(self) -> str:
        """Get tool name."""
        return self.config.name

    @property
    def description(self) -> str:
        """Get tool description."""
        return self.config.description

    @property
    def is_enabled(self) -> bool:
        """Check if tool is enabled."""
        return self.config.enabled

    async def run(self, tool_input: Union[str, Dict[str, Any]]) -> ToolResult:
        """Execute the tool with given input.

        This is the main entry point for tool execution that provides:
        - Input validation
        - Rate limiting
        - Timeout handling
        - Error handling and logging
        - Result standardization

        Args:
            tool_input: Input for the tool (string or dictionary)

        Returns:
            ToolResult: Standardized tool execution result
        """
        if not self.is_enabled:
            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.ERROR,
                content=None,
                error="Tool is disabled"
            )

        # Check rate limiting
        if not self._check_rate_limit():
            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.ERROR,
                content=None,
                error="Rate limit exceeded"
            )

        start_time = time.time()
        execution_id = str(uuid.uuid4())

        logger.debug(f"Starting execution {execution_id} for tool {self.name}")

        try:
            # Validate input
            validated_input = await self._validate_input(tool_input)

            # Execute tool with timeout
            result = await self._execute_with_timeout(validated_input)

            execution_time = time.time() - start_time
            self.execution_count += 1
            self.last_execution = datetime.now()

            # Create success result
            tool_result = ToolResult(
                tool_name=self.name,
                status=ToolStatus.SUCCESS,
                content=result,
                execution_time=execution_time,
                metadata={
                    "execution_id": execution_id,
                    "execution_count": self.execution_count,
                    "input_type": type(tool_input).__name__
                }
            )

            logger.debug(
                f"Tool {self.name} execution {execution_id} completed in {execution_time:.2f}s"
            )

            return tool_result

        except TimeoutError:
            execution_time = time.time() - start_time
            error_msg = f"Tool execution timed out after {self.config.timeout}s"

            logger.warning(f"Tool {self.name} execution {execution_id}: {error_msg}")

            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.TIMEOUT,
                content=None,
                error=error_msg,
                execution_time=execution_time,
                metadata={"execution_id": execution_id}
            )

        except ToolError as e:
            execution_time = time.time() - start_time
            error_msg = str(e)

            logger.error(f"Tool {self.name} execution {execution_id} failed: {error_msg}")

            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.ERROR,
                content=None,
                error=error_msg,
                execution_time=execution_time,
                metadata={
                    "execution_id": execution_id,
                    "error_details": e.details
                }
            )

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Unexpected error: {str(e)}"

            logger.error(
                f"Tool {self.name} execution {execution_id} failed with unexpected error: {e}",
                exc_info=True
            )

            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.ERROR,
                content=None,
                error=error_msg,
                execution_time=execution_time,
                metadata={
                    "execution_id": execution_id,
                    "error_type": type(e).__name__
                }
            )

    @abstractmethod
    async def _execute(self, validated_input: Any) -> Any:
        """Execute the tool's core functionality.

        This method should be implemented by each specific tool to perform
        its primary function. Input validation has already been performed.

        Args:
            validated_input: Validated and processed input

        Returns:
            Tool execution result

        Raises:
            ToolError: For tool-specific errors
        """
        pass

    async def _validate_input(self, tool_input: Union[str, Dict[str, Any]]) -> Any:
        """Validate and process tool input.

        Default implementation accepts both string and dictionary inputs.
        Override this method for tool-specific input validation.

        Args:
            tool_input: Raw input for the tool

        Returns:
            Validated and processed input

        Raises:
            ToolError: If input validation fails
        """
        if tool_input is None:
            raise ToolError("Tool input cannot be None", self.name)

        if isinstance(tool_input, (str, dict)):
            return tool_input

        # Try to convert to string as fallback
        try:
            return str(tool_input)
        except Exception as e:
            raise ToolError(
                f"Invalid input type: {type(tool_input).__name__}",
                self.name,
                {"input_type": type(tool_input).__name__, "conversion_error": str(e)}
            )

    async def _execute_with_timeout(self, validated_input: Any) -> Any:
        """Execute tool with timeout handling.

        Args:
            validated_input: Validated input

        Returns:
            Tool execution result

        Raises:
            TimeoutError: If execution exceeds timeout
        """
        import asyncio

        try:
            return await asyncio.wait_for(
                self._execute(validated_input),
                timeout=self.config.timeout
            )
        except asyncio.TimeoutError:
            raise TimeoutError(f"Tool execution exceeded {self.config.timeout}s timeout")

    def _check_rate_limit(self) -> bool:
        """Check if tool execution is within rate limits.

        Returns:
            True if execution is allowed, False if rate limited
        """
        if self.config.rate_limit is None:
            return True

        now = time.time()
        minute_key = int(now // 60)

        # Clean old entries
        old_keys = [k for k in self._rate_limiter.keys() if k < minute_key - 5]
        for old_key in old_keys:
            del self._rate_limiter[old_key]

        # Check current minute
        current_count = self._rate_limiter.get(minute_key, 0)
        if current_count >= self.config.rate_limit:
            logger.warning(f"Rate limit exceeded for tool {self.name}: {current_count}/{self.config.rate_limit}")
            return False

        # Update counter
        self._rate_limiter[minute_key] = current_count + 1
        return True

    def get_info(self) -> Dict[str, Any]:
        """Get tool information and statistics.

        Returns:
            Dictionary with tool information
        """
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "enabled": self.is_enabled,
            "execution_count": self.execution_count,
            "last_execution": self.last_execution.isoformat() if self.last_execution else None,
            "created_at": self.created_at.isoformat(),
            "config": self.config.dict()
        }

    async def test_connection(self) -> ToolResult:
        """Test tool connectivity and basic functionality.

        Default implementation returns success. Override for tool-specific tests.

        Returns:
            ToolResult indicating test status
        """
        return ToolResult(
            tool_name=self.name,
            status=ToolStatus.SUCCESS,
            content="Tool connection test passed",
            metadata={"test": "basic_connectivity"}
        )

    async def initialize(self) -> bool:
        """Initialize tool resources.

        Override this method for tools that need initialization.

        Returns:
            True if initialization successful, False otherwise
        """
        logger.debug(f"Initializing tool {self.name}")
        return True

    async def cleanup(self) -> bool:
        """Clean up tool resources.

        Override this method for tools that need cleanup.

        Returns:
            True if cleanup successful, False otherwise
        """
        logger.debug(f"Cleaning up tool {self.name}")
        return True


class ToolRegistry:
    """Registry for managing available tools."""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._tool_configs: Dict[str, ToolConfig] = {}

    def register_tool(self, tool: BaseTool) -> None:
        """Register a tool in the registry.

        Args:
            tool: Tool instance to register
        """
        tool_name = tool.name
        if tool_name in self._tools:
            logger.warning(f"Overriding existing tool registration: {tool_name}")

        self._tools[tool_name] = tool
        self._tool_configs[tool_name] = tool.config

        logger.info(f"Registered tool: {tool_name}")

    def unregister_tool(self, tool_name: str) -> bool:
        """Unregister a tool from the registry.

        Args:
            tool_name: Name of tool to unregister

        Returns:
            True if tool was found and unregistered, False otherwise
        """
        if tool_name in self._tools:
            del self._tools[tool_name]
            del self._tool_configs[tool_name]
            logger.info(f"Unregistered tool: {tool_name}")
            return True
        return False

    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """Get a tool by name.

        Args:
            tool_name: Name of the tool

        Returns:
            Tool instance if found, None otherwise
        """
        return self._tools.get(tool_name)

    def list_tools(self) -> List[str]:
        """List all registered tool names.

        Returns:
            List of tool names
        """
        return list(self._tools.keys())

    def get_enabled_tools(self) -> List[str]:
        """Get list of enabled tool names.

        Returns:
            List of enabled tool names
        """
        return [name for name, tool in self._tools.items() if tool.is_enabled]

    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Tool information dictionary if found, None otherwise
        """
        tool = self.get_tool(tool_name)
        return tool.get_info() if tool else None

    async def initialize_all(self) -> Dict[str, bool]:
        """Initialize all registered tools.

        Returns:
            Dictionary mapping tool names to initialization success status
        """
        results = {}
        for tool_name, tool in self._tools.items():
            try:
                results[tool_name] = await tool.initialize()
            except Exception as e:
                logger.error(f"Failed to initialize tool {tool_name}: {e}")
                results[tool_name] = False

        return results

    async def cleanup_all(self) -> Dict[str, bool]:
        """Clean up all registered tools.

        Returns:
            Dictionary mapping tool names to cleanup success status
        """
        results = {}
        for tool_name, tool in self._tools.items():
            try:
                results[tool_name] = await tool.cleanup()
            except Exception as e:
                logger.error(f"Failed to cleanup tool {tool_name}: {e}")
                results[tool_name] = False

        return results


# Global tool registry instance
_global_registry = ToolRegistry()


def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry instance.

    Returns:
        Global ToolRegistry instance
    """
    return _global_registry


def register_tool(tool: BaseTool) -> None:
    """Register a tool in the global registry.

    Args:
        tool: Tool instance to register
    """
    _global_registry.register_tool(tool)


def get_tool(tool_name: str) -> Optional[BaseTool]:
    """Get a tool from the global registry.

    Args:
        tool_name: Name of the tool

    Returns:
        Tool instance if found, None otherwise
    """
    return _global_registry.get_tool(tool_name)