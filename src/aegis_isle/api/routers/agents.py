"""
Agent management and orchestration endpoints.
"""

from typing import Dict, Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from ..dependencies import get_agent_router, get_agent_orchestrator, get_request_id
from ...core.logging import logger
from ...agents.router import AgentRouter
from ...agents.orchestrator import AgentOrchestrator
from ...agents.base import AgentConfig, AgentRole

agents_router = APIRouter()


class AgentConfigRequest(BaseModel):
    """Request model for agent configuration."""
    name: str
    role: str  # Will be converted to AgentRole
    description: str
    max_iterations: Optional[int] = 10
    timeout: Optional[int] = 300
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000
    model_name: Optional[str] = None
    tools: Optional[List[str]] = None
    prompt_template: Optional[str] = None
    enabled: bool = True
    memory_enabled: bool = True


class WorkflowExecutionRequest(BaseModel):
    """Request model for workflow execution."""
    workflow_name: str
    input_data: Any
    workflow_id: Optional[str] = None


class MessageRequest(BaseModel):
    """Request model for sending messages to agents."""
    message: str
    target_agents: Optional[List[str]] = None  # None for auto-routing
    broadcast: bool = False


@agents_router.get("/")
async def list_agents(
    router: AgentRouter = Depends(get_agent_router)
) -> Dict[str, Any]:
    """List all registered agents."""

    try:
        agent_status = router.get_agent_status()

        return {
            "total_agents": len(agent_status),
            "agents": agent_status
        }

    except Exception as e:
        logger.error(f"Error listing agents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@agents_router.get("/{agent_id}")
async def get_agent_info(
    agent_id: str,
    router: AgentRouter = Depends(get_agent_router)
) -> Dict[str, Any]:
    """Get detailed information about a specific agent."""

    try:
        if agent_id not in router.agents:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent not found: {agent_id}"
            )

        agent = router.agents[agent_id]
        return agent.get_info()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@agents_router.post("/message")
async def send_message_to_agents(
    request: MessageRequest,
    router: AgentRouter = Depends(get_agent_router),
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Send a message to agents."""

    try:
        if not request.message.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Message cannot be empty"
            )

        logger.info(f"Sending message to agents: {request.message[:100]}...")

        if request.broadcast:
            # Broadcast to all agents
            results = await router.broadcast_message(request.message)
        elif request.target_agents:
            # Send to specific agents
            results = await router.send_to_agents(request.message, request.target_agents)
        else:
            # Auto-route based on message content
            target_agents = await router.route_message(request.message)
            results = await router.send_to_agents(request.message, target_agents)

        return {
            "success": True,
            "message": "Message sent successfully",
            "request_id": request_id,
            "results": results
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error sending message to agents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@agents_router.get("/workflows/templates")
async def list_workflow_templates(
    orchestrator: AgentOrchestrator = Depends(get_agent_orchestrator)
) -> Dict[str, Any]:
    """List available workflow templates."""

    try:
        templates = list(orchestrator.workflow_templates.keys())

        return {
            "total_templates": len(templates),
            "templates": templates,
            "details": {
                name: {
                    "name": workflow.name,
                    "description": workflow.description,
                    "steps": list(workflow.steps.keys())
                }
                for name, workflow in orchestrator.workflow_templates.items()
            }
        }

    except Exception as e:
        logger.error(f"Error listing workflow templates: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@agents_router.post("/workflows/execute")
async def execute_workflow(
    request: WorkflowExecutionRequest,
    orchestrator: AgentOrchestrator = Depends(get_agent_orchestrator),
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Execute a workflow."""

    try:
        if request.workflow_name not in orchestrator.workflow_templates:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workflow template not found: {request.workflow_name}"
            )

        logger.info(f"Executing workflow: {request.workflow_name}")

        result = await orchestrator.execute_workflow(
            workflow_name=request.workflow_name,
            initial_input=request.input_data,
            workflow_id=request.workflow_id
        )

        result["request_id"] = request_id
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing workflow: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@agents_router.get("/workflows/{workflow_id}/status")
async def get_workflow_status(
    workflow_id: str,
    orchestrator: AgentOrchestrator = Depends(get_agent_orchestrator)
) -> Dict[str, Any]:
    """Get the status of a running workflow."""

    try:
        status = orchestrator.get_workflow_status(workflow_id)

        if status is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workflow not found: {workflow_id}"
            )

        return status

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting workflow status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@agents_router.get("/active-workflows")
async def list_active_workflows(
    orchestrator: AgentOrchestrator = Depends(get_agent_orchestrator)
) -> Dict[str, Any]:
    """List all currently active workflows."""

    try:
        active_workflows = {}

        for workflow_id in orchestrator.active_workflows.keys():
            status = orchestrator.get_workflow_status(workflow_id)
            if status:
                active_workflows[workflow_id] = status

        return {
            "total_active": len(active_workflows),
            "workflows": active_workflows
        }

    except Exception as e:
        logger.error(f"Error listing active workflows: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@agents_router.get("/roles")
async def list_agent_roles() -> Dict[str, Any]:
    """List available agent roles."""

    roles = [role.value for role in AgentRole]

    return {
        "available_roles": roles,
        "role_descriptions": {
            "researcher": "Performs research and investigation tasks",
            "retriever": "Retrieves documents and information from knowledge base",
            "summarizer": "Summarizes and synthesizes information",
            "chart_generator": "Generates charts and visualizations",
            "tool_caller": "Executes tools and external functions",
            "coordinator": "Coordinates tasks between multiple agents",
            "router": "Routes messages and tasks to appropriate agents"
        }
    }