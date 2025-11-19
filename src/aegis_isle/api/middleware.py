"""
Middleware setup for the FastAPI application.
"""

import time
import uuid
from typing import Callable

from fastapi import FastAPI, Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware

from ..core.config import settings
from ..core.logging import logger


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging HTTP requests."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and log details."""
        if not settings.log_requests:
            return await call_next(request)

        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        # Log request
        start_time = time.time()
        logger.info(
            f"Request {request_id}: {request.method} {request.url} "
            f"from {request.client.host if request.client else 'unknown'}"
        )

        # Process request
        try:
            response = await call_next(request)
            duration = time.time() - start_time

            # Log response
            logger.info(
                f"Request {request_id}: {response.status_code} "
                f"({duration:.3f}s)"
            )

            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id

            return response

        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"Request {request_id}: Error after {duration:.3f}s - {str(e)}"
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