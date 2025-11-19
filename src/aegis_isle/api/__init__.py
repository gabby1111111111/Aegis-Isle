"""
FastAPI service layer for AegisIsle RAG system.

This module provides REST API endpoints for the multi-agent RAG system,
including document management, querying, and agent orchestration.
"""

from .main import create_app
from .routers import (
    documents_router,
    query_router,
    agents_router,
    health_router,
    admin_router
)
from .middleware import setup_middleware
from .dependencies import get_rag_pipeline, get_agent_orchestrator

__all__ = [
    "create_app",
    "documents_router",
    "query_router",
    "agents_router",
    "health_router",
    "admin_router",
    "setup_middleware",
    "get_rag_pipeline",
    "get_agent_orchestrator",
]