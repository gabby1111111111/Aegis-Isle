"""
Document management endpoints.
"""

import os
from typing import Dict, Any, List, Optional
from pathlib import Path

from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, status
from pydantic import BaseModel

from ..dependencies import get_rag_pipeline, get_request_id
from ...core.config import settings
from ...core.logging import logger
from ...rag.pipeline import RAGPipeline

documents_router = APIRouter()


class DocumentUploadResponse(BaseModel):
    """Response model for document upload."""
    success: bool
    message: str
    document_id: Optional[str] = None
    metadata: Dict[str, Any] = {}


class TextDocumentRequest(BaseModel):
    """Request model for adding text content."""
    content: str
    metadata: Optional[Dict[str, Any]] = None


class UrlDocumentRequest(BaseModel):
    """Request model for adding URL content."""
    url: str
    metadata: Optional[Dict[str, Any]] = None


@documents_router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
    request_id: str = Depends(get_request_id)
) -> DocumentUploadResponse:
    """Upload and process a document file."""

    try:
        # Validate file type
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No filename provided"
            )

        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in ['.pdf', '.docx', '.doc', '.txt', '.md', '.html', '.htm']:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file type: {file_ext}"
            )

        # Check file size
        content = await file.read()
        if len(content) > 50 * 1024 * 1024:  # 50MB limit
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="File too large (max 50MB)"
            )

        # Save temporary file
        temp_dir = settings.uploads_dir
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / f"{request_id}_{file.filename}"

        with open(temp_path, "wb") as temp_file:
            temp_file.write(content)

        logger.info(f"Saved uploaded file to {temp_path}")

        # Process document
        metadata = {
            "filename": file.filename,
            "content_type": file.content_type,
            "file_size": len(content),
            "request_id": request_id
        }

        success = await pipeline.add_document(str(temp_path), metadata)

        # Clean up temporary file
        try:
            os.unlink(temp_path)
        except:
            pass

        if success:
            return DocumentUploadResponse(
                success=True,
                message=f"Successfully processed document: {file.filename}",
                metadata=metadata
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to process document"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@documents_router.post("/text", response_model=DocumentUploadResponse)
async def add_text_content(
    request: TextDocumentRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
    request_id: str = Depends(get_request_id)
) -> DocumentUploadResponse:
    """Add raw text content to the knowledge base."""

    try:
        if not request.content.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Content cannot be empty"
            )

        metadata = request.metadata or {}
        metadata.update({
            "source": "text_input",
            "content_length": len(request.content),
            "request_id": request_id
        })

        success = await pipeline.add_text(request.content, metadata)

        if success:
            return DocumentUploadResponse(
                success=True,
                message="Successfully added text content",
                metadata=metadata
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to process text content"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding text content: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@documents_router.post("/url", response_model=DocumentUploadResponse)
async def add_url_content(
    request: UrlDocumentRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
    request_id: str = Depends(get_request_id)
) -> DocumentUploadResponse:
    """Add content from a URL to the knowledge base."""

    try:
        if not request.url.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="URL cannot be empty"
            )

        # Basic URL validation
        if not request.url.startswith(('http://', 'https://')):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="URL must start with http:// or https://"
            )

        metadata = request.metadata or {}
        metadata.update({
            "source": "url",
            "url": request.url,
            "request_id": request_id
        })

        success = await pipeline.add_url(request.url, metadata)

        if success:
            return DocumentUploadResponse(
                success=True,
                message=f"Successfully processed URL: {request.url}",
                metadata=metadata
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to process URL content"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding URL content: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@documents_router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    pipeline: RAGPipeline = Depends(get_rag_pipeline)
) -> Dict[str, Any]:
    """Delete a document from the knowledge base."""

    try:
        success = await pipeline.delete_document(document_id)

        if success:
            return {
                "success": True,
                "message": f"Successfully deleted document: {document_id}"
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document not found: {document_id}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@documents_router.get("/stats")
async def get_document_stats(
    pipeline: RAGPipeline = Depends(get_rag_pipeline)
) -> Dict[str, Any]:
    """Get statistics about the document collection."""

    try:
        stats = await pipeline.get_stats()
        return {
            "success": True,
            "stats": stats
        }

    except Exception as e:
        logger.error(f"Error getting document stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )