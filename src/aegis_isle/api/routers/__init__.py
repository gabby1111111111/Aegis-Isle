"""
API routers for AegisIsle RAG system.
"""

from .documents import documents_router
from .query import query_router
from .agents import agents_router
from .health import health_router
from .admin import admin_router
from .auth import router as auth_router

__all__ = [
    "documents_router",
    "query_router",
    "agents_router",
    "health_router",
    "admin_router",
    "auth_router"
]