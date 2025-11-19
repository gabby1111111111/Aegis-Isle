"""
Health check and system status endpoints.
"""

from typing import Dict, Any

from fastapi import APIRouter, Depends, Request

from ..dependencies import get_rag_pipeline, get_agent_orchestrator, get_metrics_middleware
from ...core.config import settings

health_router = APIRouter()


@health_router.get("/")
async def health_check() -> Dict[str, Any]:
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "service": "AegisIsle RAG API",
        "version": "0.1.0",
        "timestamp": "2024-01-01T00:00:00Z"
    }


@health_router.get("/detailed")
async def detailed_health_check(
    request: Request,
    pipeline=Depends(get_rag_pipeline),
    orchestrator=Depends(get_agent_orchestrator)
) -> Dict[str, Any]:
    """Detailed health check including all components."""

    # Get RAG pipeline health
    rag_health = await pipeline.health_check()

    # Get agent orchestrator status
    agent_status = orchestrator.router.get_agent_status()

    # Get system metrics if available
    metrics = {}
    try:
        metrics_middleware = get_metrics_middleware(request)
        metrics = metrics_middleware.get_metrics()
    except:
        pass

    return {
        "status": "healthy" if rag_health["status"] == "healthy" else "degraded",
        "service": "AegisIsle RAG API",
        "version": "0.1.0",
        "components": {
            "rag_pipeline": rag_health,
            "agents": {
                "status": "healthy" if agent_status else "unhealthy",
                "registered_agents": len(agent_status),
                "agents": agent_status
            }
        },
        "metrics": metrics,
        "configuration": {
            "environment": settings.environment,
            "debug": settings.debug,
            "features": {
                "multimodal": settings.enable_multimodal,
                "metrics": settings.enable_metrics,
                "agent_memory": settings.enable_memory
            }
        }
    }


@health_router.get("/ready")
async def readiness_check(pipeline=Depends(get_rag_pipeline)) -> Dict[str, Any]:
    """Kubernetes-style readiness probe."""
    try:
        health = await pipeline.health_check()
        ready = health["status"] == "healthy"

        return {
            "ready": ready,
            "status": "ready" if ready else "not_ready"
        }
    except Exception as e:
        return {
            "ready": False,
            "status": "not_ready",
            "error": str(e)
        }


@health_router.get("/live")
async def liveness_check() -> Dict[str, Any]:
    """Kubernetes-style liveness probe."""
    return {
        "alive": True,
        "status": "alive"
    }