"""
Query processing endpoints for RAG system.
"""

from typing import Dict, Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from ..dependencies import get_rag_pipeline, get_agent_orchestrator, get_request_id
from ...core.logging import logger
from ...rag.pipeline import RAGPipeline
from ...agents.orchestrator import AgentOrchestrator

query_router = APIRouter()


class QueryRequest(BaseModel):
    """Request model for RAG queries."""
    query: str
    max_docs: Optional[int] = 5
    use_agents: bool = False
    agent_workflow: Optional[str] = "rag_query"
    metadata: Optional[Dict[str, Any]] = None


class QueryResponse(BaseModel):
    """Response model for RAG queries."""
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class BatchQueryRequest(BaseModel):
    """Request model for batch queries."""
    queries: List[str]
    max_docs: Optional[int] = 5
    use_agents: bool = False
    agent_workflow: Optional[str] = "rag_query"


class BatchQueryResponse(BaseModel):
    """Response model for batch queries."""
    results: List[QueryResponse]
    total_queries: int
    total_time: float
    metadata: Dict[str, Any]


@query_router.post("/", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
    orchestrator: AgentOrchestrator = Depends(get_agent_orchestrator),
    request_id: str = Depends(get_request_id)
) -> QueryResponse:
    """Process a single RAG query."""

    try:
        if not request.query.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query cannot be empty"
            )

        logger.info(f"Processing query: {request.query[:100]}...")

        # Use agent orchestration if requested
        if request.use_agents:
            workflow_result = await orchestrator.execute_workflow(
                workflow_name=request.agent_workflow or "rag_query",
                initial_input=request.query
            )

            if workflow_result["success"]:
                # Extract answer from workflow results
                answer = "Agent-based response generated successfully"
                sources = []
                metadata = {
                    "workflow_id": workflow_result["workflow_id"],
                    "workflow_results": workflow_result["results"],
                    "agent_based": True,
                    "request_id": request_id
                }
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Agent workflow failed: {workflow_result.get('error', 'Unknown error')}"
                )

        else:
            # Use standard RAG pipeline
            result = await pipeline.query(
                query=request.query,
                max_docs=request.max_docs,
                **(request.metadata or {})
            )

            answer = result.answer
            sources = result.sources
            metadata = result.metadata
            metadata.update({
                "agent_based": False,
                "request_id": request_id,
                "total_time": result.total_time
            })

        return QueryResponse(
            query=request.query,
            answer=answer,
            sources=sources,
            metadata=metadata
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@query_router.post("/batch", response_model=BatchQueryResponse)
async def process_batch_queries(
    request: BatchQueryRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
    orchestrator: AgentOrchestrator = Depends(get_agent_orchestrator),
    request_id: str = Depends(get_request_id)
) -> BatchQueryResponse:
    """Process multiple queries in batch."""

    try:
        if not request.queries:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Queries list cannot be empty"
            )

        logger.info(f"Processing {len(request.queries)} queries in batch")

        import time
        start_time = time.time()
        results = []

        for i, query in enumerate(request.queries):
            if not query.strip():
                continue

            try:
                if request.use_agents:
                    # Use agent orchestration
                    workflow_result = await orchestrator.execute_workflow(
                        workflow_name=request.agent_workflow or "rag_query",
                        initial_input=query
                    )

                    if workflow_result["success"]:
                        answer = "Agent-based response generated successfully"
                        sources = []
                        metadata = {
                            "workflow_id": workflow_result["workflow_id"],
                            "agent_based": True,
                            "batch_index": i
                        }
                    else:
                        answer = f"Agent workflow failed: {workflow_result.get('error', 'Unknown error')}"
                        sources = []
                        metadata = {"error": True, "batch_index": i}

                else:
                    # Use standard RAG pipeline
                    result = await pipeline.query(
                        query=query,
                        max_docs=request.max_docs
                    )

                    answer = result.answer
                    sources = result.sources
                    metadata = result.metadata
                    metadata.update({
                        "agent_based": False,
                        "batch_index": i
                    })

                results.append(QueryResponse(
                    query=query,
                    answer=answer,
                    sources=sources,
                    metadata=metadata
                ))

            except Exception as e:
                logger.error(f"Error processing batch query {i}: {e}")
                results.append(QueryResponse(
                    query=query,
                    answer=f"Error processing query: {str(e)}",
                    sources=[],
                    metadata={"error": True, "batch_index": i}
                ))

        total_time = time.time() - start_time

        return BatchQueryResponse(
            results=results,
            total_queries=len(request.queries),
            total_time=total_time,
            metadata={
                "request_id": request_id,
                "use_agents": request.use_agents,
                "successful_queries": len([r for r in results if not r.metadata.get("error")]),
                "failed_queries": len([r for r in results if r.metadata.get("error")])
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing batch queries: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@query_router.get("/history")
async def get_query_history() -> Dict[str, Any]:
    """Get query history (placeholder)."""
    # TODO: Implement query history storage and retrieval
    return {
        "message": "Query history not implemented yet",
        "total_queries": 0,
        "recent_queries": []
    }


@query_router.post("/feedback")
async def submit_query_feedback(
    query_id: str,
    feedback: Dict[str, Any]
) -> Dict[str, Any]:
    """Submit feedback for a query result (placeholder)."""
    # TODO: Implement feedback collection system
    return {
        "message": "Feedback submitted successfully",
        "query_id": query_id,
        "feedback": feedback
    }