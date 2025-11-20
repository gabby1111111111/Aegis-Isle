"""
Main FastAPI application setup.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..core.config import settings
from ..core.logging import logger
from .middleware import setup_middleware
from .routers import (
    documents_router,
    query_router,
    agents_router,
    health_router,
    admin_router,
    auth_router
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    logger.info("Starting AegisIsle API server")

    # Initialize RAG pipeline
    try:
        from ..rag.pipeline import initialize_default_pipeline
        pipeline = await initialize_default_pipeline()
        app.state.rag_pipeline = pipeline
        logger.info("RAG pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG pipeline: {e}")
        raise

    # Initialize agent orchestrator
    try:
        from ..agents.router import AgentRouter
        from ..agents.orchestrator import AgentOrchestrator

        router = AgentRouter()
        orchestrator = AgentOrchestrator(router)

        # Register default workflow templates
        rag_workflow = orchestrator.create_rag_workflow()
        orchestrator.register_workflow_template(rag_workflow)

        app.state.agent_router = router
        app.state.agent_orchestrator = orchestrator
        logger.info("Agent orchestrator initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize agent orchestrator: {e}")
        raise

    logger.info("AegisIsle API server started successfully")

    yield

    # Shutdown
    logger.info("Shutting down AegisIsle API server")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="AegisIsle RAG API",
        description="Multi-Agent Collaborative RAG System API",
        version="0.1.0",
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        lifespan=lifespan
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_hosts_list if settings.allowed_hosts != "*" else ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Setup custom middleware
    setup_middleware(app)

    # Include routers
    app.include_router(
        health_router,
        prefix="/api/v1/health",
        tags=["health"]
    )

    app.include_router(
        auth_router,
        prefix="/api/v1",
        tags=["authentication"]
    )

    app.include_router(
        documents_router,
        prefix="/api/v1/documents",
        tags=["documents"]
    )

    app.include_router(
        query_router,
        prefix="/api/v1/query",
        tags=["query"]
    )

    app.include_router(
        agents_router,
        prefix="/api/v1/agents",
        tags=["agents"]
    )

    app.include_router(
        admin_router,
        prefix="/api/v1/admin",
        tags=["admin"]
    )

    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "message": "Welcome to AegisIsle RAG API",
            "version": "0.1.0",
            "docs": "/docs" if settings.debug else "disabled",
            "health": "/api/v1/health"
        }

    @app.get("/info")
    async def info():
        """System information endpoint."""
        return {
            "system": "AegisIsle",
            "version": "0.1.0",
            "environment": settings.environment,
            "debug": settings.debug,
            "features": {
                "rag": True,
                "multi_agent": True,
                "multimodal": settings.enable_multimodal,
                "metrics": settings.enable_metrics
            }
        }

    return app


# Create the app instance
app = create_app()