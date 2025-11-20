"""
Middleware setup for the FastAPI application.
"""

import time
import uuid
from typing import Callable

from fastapi import FastAPI, Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware

from ..core.config import settings
from ..core.logging import logger, audit_logger


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging HTTP requests with audit support."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and log details."""
        if not settings.log_requests:
            return await call_next(request)

        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        # Extract client information
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")

        # Log request
        start_time = time.time()
        logger.info(
            f"Request {request_id}: {request.method} {request.url} "
            f"from {client_ip}"
        )

        # Process request
        try:
            response = await call_next(request)
            duration = time.time() - start_time
            duration_ms = duration * 1000

            # Log response
            logger.info(
                f"Request {request_id}: {response.status_code} "
                f"({duration:.3f}s)"
            )

            # Get authenticated user info if available
            user_id = None
            username = None
            try:
                # Try to extract user info from request state if auth middleware has set it
                if hasattr(request.state, 'current_user'):
                    user_info = request.state.current_user
                    user_id = getattr(user_info, 'user_id', None)
                    username = getattr(user_info, 'username', None)
            except:
                pass

            # Log API access audit event (only for API endpoints)
            if str(request.url.path).startswith("/api/"):
                audit_logger.log_api_access(
                    method=request.method,
                    endpoint=request.url.path,
                    user_id=user_id,
                    username=username,
                    ip_address=client_ip,
                    user_agent=user_agent,
                    status_code=response.status_code,
                    response_time_ms=duration_ms,
                    request_id=request_id
                )

            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id

            return response

        except Exception as e:
            duration = time.time() - start_time
            duration_ms = duration * 1000

            logger.error(
                f"Request {request_id}: Error after {duration:.3f}s - {str(e)}"
            )

            # Log API access audit event for errors (only for API endpoints)
            if str(request.url.path).startswith("/api/"):
                audit_logger.log_api_access(
                    method=request.method,
                    endpoint=request.url.path,
                    ip_address=client_ip,
                    user_agent=user_agent,
                    status_code=500,  # Internal server error
                    response_time_ms=duration_ms,
                    request_id=request_id
                )

            raise


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for collecting metrics."""

    def __init__(self, app):
        super().__init__(app)
        self.request_count = 0
        self.request_duration_total = 0.0
        self.error_count = 0

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and collect metrics."""
        if not settings.enable_metrics:
            return await call_next(request)

        start_time = time.time()
        self.request_count += 1

        try:
            response = await call_next(request)
            duration = time.time() - start_time
            self.request_duration_total += duration

            # Add metrics headers
            response.headers["X-Request-Count"] = str(self.request_count)
            response.headers["X-Request-Duration"] = f"{duration:.3f}"

            return response

        except Exception as e:
            self.error_count += 1
            duration = time.time() - start_time
            self.request_duration_total += duration
            raise

    def get_metrics(self) -> dict:
        """Get collected metrics."""
        avg_duration = (
            self.request_duration_total / self.request_count
            if self.request_count > 0 else 0
        )

        return {
            "total_requests": self.request_count,
            "total_errors": self.error_count,
            "average_request_duration": avg_duration,
            "total_request_duration": self.request_duration_total,
            "error_rate": self.error_count / max(self.request_count, 1)
        }


def setup_middleware(app: FastAPI) -> None:
    """Setup all middleware for the application."""

    # Request logging middleware
    if settings.log_requests:
        app.add_middleware(RequestLoggingMiddleware)

    # Metrics middleware
    if settings.enable_metrics:
        metrics_middleware = MetricsMiddleware(app)
        app.add_middleware(MetricsMiddleware)

        # Store reference for metrics endpoint
        app.state.metrics_middleware = metrics_middleware

    logger.info("Middleware setup completed")