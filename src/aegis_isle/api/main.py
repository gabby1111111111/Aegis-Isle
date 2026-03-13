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
    auth_router,
    openai_compat,
)
from .routers.memory import router as memory_router


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
        logger.warning(f"Failed to initialize RAG pipeline: {e}")
        app.state.rag_pipeline = None

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
        logger.warning(f"Failed to initialize agent orchestrator: {e}")
        app.state.agent_router = None
        app.state.agent_orchestrator = None

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
        lifespan=lifespan,
    )

    # Add CORS middleware
    # 允许来自所有本地来源的请求（包括 SillyTavern 在 8000 端口的跨域请求）
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Setup custom middleware
    setup_middleware(app)

    # Include routers
    app.include_router(health_router, prefix="/api/v1/health", tags=["health"])

    app.include_router(auth_router, prefix="/api/v1", tags=["authentication"])

    # OpenAI Compatible API
    app.include_router(openai_compat.router, prefix="/v1", tags=["openai"])

    app.include_router(documents_router, prefix="/api/v1/documents", tags=["documents"])

    app.include_router(query_router, prefix="/api/v1/query", tags=["query"])

    app.include_router(agents_router, prefix="/api/v1/agents", tags=["agents"])

    app.include_router(admin_router, prefix="/api/v1/admin", tags=["admin"])

    # 🌟 长线记忆 API（供 SillyTavern 插件调用）
    app.include_router(memory_router, prefix="/v1", tags=["memory"])

    # 🌌 世界线管理器 API
    from .routers.universe_manager import router as universe_router

    app.include_router(universe_router, prefix="/v1", tags=["universe"])

    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "message": "Welcome to AegisIsle RAG API",
            "version": "0.1.0",
            "docs": "/docs" if settings.debug else "disabled",
            "health": "/api/v1/health",
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
                "metrics": settings.enable_metrics,
            },
        }

    return app


# Create the app instance
app = create_app()
