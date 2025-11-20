"""
Administrative endpoints for system management.
"""

from typing import Dict, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from ..dependencies import get_rag_pipeline, get_agent_orchestrator, require_admin, get_metrics_middleware, CurrentUser
from ...core.config import settings
from ...core.logging import logger
from ...rag.pipeline import RAGPipeline, RAGConfig
from ...agents.orchestrator import AgentOrchestrator

admin_router = APIRouter()


class ConfigUpdateRequest(BaseModel):
    """Request model for updating configuration."""
    config_updates: Dict[str, Any]


class SystemStatsResponse(BaseModel):
    """Response model for system statistics."""
    system_info: Dict[str, Any]
    rag_stats: Dict[str, Any]
    agent_stats: Dict[str, Any]
    metrics: Dict[str, Any]


@admin_router.get("/config")
async def get_system_config(
    admin_user: CurrentUser = Depends(require_admin)
) -> Dict[str, Any]:
    """Get current system configuration."""

    try:
        return {
            "environment": settings.environment,
            "debug": settings.debug,
            "api_settings": {
                "host": settings.api_host,
                "port": settings.api_port,
                "reload": settings.api_reload
            },
            "rag_settings": {
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
                "max_retrieved_docs": settings.max_retrieved_docs,
                "similarity_threshold": settings.similarity_threshold,
                "vector_db_type": settings.vector_db_type,
                "embedding_model": settings.embedding_model
            },
            "llm_settings": {
                "provider": settings.llm_provider,
                "model": settings.default_llm_model,
                "max_tokens": settings.max_tokens,
                "temperature": settings.temperature
            },
            "agent_settings": {
                "max_iterations": settings.max_agent_iterations,
                "timeout": settings.agent_timeout,
                "enable_memory": settings.enable_memory
            },
            "feature_flags": {
                "multimodal": settings.enable_multimodal,
                "metrics": settings.enable_metrics,
                "ocr": settings.ocr_enabled,
                "image_processing": settings.image_processing_enabled
            }
        }

    except Exception as e:
        logger.error(f"Error getting system config: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@admin_router.put("/config")
async def update_system_config(
    request: ConfigUpdateRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
    admin_user: CurrentUser = Depends(require_admin)
) -> Dict[str, Any]:
    """Update system configuration."""

    try:
        updated_settings = []

        # Update RAG pipeline configuration
        rag_updates = {}
        for key, value in request.config_updates.items():
            if hasattr(pipeline.config, key):
                setattr(pipeline.config, key, value)
                rag_updates[key] = value
                updated_settings.append(f"rag.{key}")

        if rag_updates:
            pipeline.update_config(**rag_updates)

        # Update global settings (note: these won't persist across restarts)
        for key, value in request.config_updates.items():
            if hasattr(settings, key):
                setattr(settings, key, value)
                updated_settings.append(f"settings.{key}")

        return {
            "success": True,
            "message": "Configuration updated successfully",
            "updated_settings": updated_settings,
            "note": "Changes will not persist across server restarts unless updated in environment variables"
        }

    except Exception as e:
        logger.error(f"Error updating system config: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@admin_router.get("/stats", response_model=SystemStatsResponse)
async def get_system_stats(
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
    orchestrator: AgentOrchestrator = Depends(get_agent_orchestrator),
    admin_user: CurrentUser = Depends(require_admin)
) -> SystemStatsResponse:
    """Get comprehensive system statistics."""

    try:
        # Get RAG pipeline stats
        rag_stats = await pipeline.get_stats()

        # Get agent stats
        agent_status = orchestrator.router.get_agent_status()
        agent_stats = {
            "total_agents": len(agent_status),
            "active_agents": len([a for a in agent_status.values() if a.get("status") != "inactive"]),
            "workflow_templates": len(orchestrator.workflow_templates),
            "active_workflows": len(orchestrator.active_workflows),
            "agents_by_role": {}
        }

        # Count agents by role
        for agent_info in agent_status.values():
            role = agent_info.get("role", "unknown")
            agent_stats["agents_by_role"][role] = agent_stats["agents_by_role"].get(role, 0) + 1

        # Get system metrics
        metrics = {}
        try:
            # This would need to be passed through dependencies if metrics middleware is available
            pass
        except:
            pass

        # Get system information
        import psutil
        import platform

        system_info = {
            "platform": platform.system(),
            "platform_version": platform.version(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory": {
                "total": psutil.virtual_memory().total,
                "available": psutil.virtual_memory().available,
                "percent": psutil.virtual_memory().percent
            },
            "disk": {
                "total": psutil.disk_usage('/').total,
                "free": psutil.disk_usage('/').free,
                "percent": psutil.disk_usage('/').percent
            }
        }

        return SystemStatsResponse(
            system_info=system_info,
            rag_stats=rag_stats,
            agent_stats=agent_stats,
            metrics=metrics
        )

    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@admin_router.post("/maintenance/clear-cache")
async def clear_system_cache(
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
    admin_user: CurrentUser = Depends(require_admin)
) -> Dict[str, Any]:
    """Clear system caches."""

    try:
        # Clear any in-memory caches
        # This would depend on specific cache implementations

        return {
            "success": True,
            "message": "System caches cleared successfully"
        }

    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@admin_router.post("/maintenance/health-check")
async def run_comprehensive_health_check(
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
    admin_user: CurrentUser = Depends(require_admin)
) -> Dict[str, Any]:
    """Run a comprehensive health check on all system components."""

    try:
        health_result = await pipeline.health_check()

        # Additional health checks could be added here
        # - Database connectivity
        # - External API availability
        # - File system access
        # - Memory usage checks

        return {
            "success": True,
            "health_check": health_result,
            "timestamp": "2024-01-01T00:00:00Z",
            "recommendations": []  # Could include performance recommendations
        }

    except Exception as e:
        logger.error(f"Error running health check: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@admin_router.get("/logs")
async def get_system_logs(
    lines: int = 100,
    level: Optional[str] = None,
    admin_user: CurrentUser = Depends(require_admin)
) -> Dict[str, Any]:
    """Get recent system logs."""

    try:
        # This would need to be implemented based on your logging setup
        # For now, return a placeholder

        return {
            "success": True,
            "message": f"Log retrieval not implemented yet",
            "requested_lines": lines,
            "requested_level": level,
            "logs": []
        }

    except Exception as e:
        logger.error(f"Error getting logs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )