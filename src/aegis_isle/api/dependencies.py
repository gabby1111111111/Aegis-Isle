"""
Dependency injection for FastAPI endpoints.
"""

from typing import Optional

from fastapi import Depends, HTTPException, Request, status

from ..agents.orchestrator import AgentOrchestrator
from ..agents.router import AgentRouter
from ..core.logging import logger
from ..rag.pipeline import RAGPipeline


def get_rag_pipeline(request: Request) -> RAGPipeline:
    """Get the RAG pipeline from the application state."""
    pipeline = getattr(request.app.state, "rag_pipeline", None)
    if pipeline is None:
        logger.error("RAG pipeline not initialized")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG pipeline not available"
        )
    return pipeline


def get_agent_router(request: Request) -> AgentRouter:
    """Get the agent router from the application state."""
    router = getattr(request.app.state, "agent_router", None)
    if router is None:
        logger.error("Agent router not initialized")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent router not available"
        )
    return router


def get_agent_orchestrator(request: Request) -> AgentOrchestrator:
    """Get the agent orchestrator from the application state."""
    orchestrator = getattr(request.app.state, "agent_orchestrator", None)
    if orchestrator is None:
        logger.error("Agent orchestrator not initialized")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent orchestrator not available"
        )
    return orchestrator


def get_metrics_middleware(request: Request):
    """Get the metrics middleware from the application state."""
    middleware = getattr(request.app.state, "metrics_middleware", None)
    if middleware is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Metrics not available"
        )
    return middleware


def get_request_id(request: Request) -> str:
    """Get the request ID from the request state."""
    return getattr(request.state, "request_id", "unknown")


# Optional authentication dependency (placeholder)
def get_current_user(request: Request) -> Optional[dict]:
    """
    Get current user from request.
    This is a placeholder for authentication implementation.
    """
    # TODO: Implement proper authentication
    # For now, return a default user
    return {
        "id": "default_user",
        "username": "anonymous",
        "roles": ["user"]
    }


def require_admin(current_user: dict = Depends(get_current_user)) -> dict:
    """Require admin role for access."""
    if "admin" not in current_user.get("roles", []):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user