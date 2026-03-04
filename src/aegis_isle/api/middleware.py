"""
Middleware setup for the FastAPI application.
"""

import time
import uuid
from typing import Callable, Dict

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

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
    """
    增强版请求度量中间件。
    
    功能:
    - 按端点分组的请求计数和延迟统计
    - P50/P95/P99 延迟分位数计算
    - 超过 5 秒的慢请求自动告警
    - 错误率追踪
    """

    # 慢请求阈值 (毫秒)
    SLOW_REQUEST_THRESHOLD_MS = 5000

    def __init__(self, app):
        super().__init__(app)
        self.request_count = 0
        self.request_duration_total = 0.0
        self.error_count = 0
        # 每个端点独立的延迟历史 (最多保留 1000 条)
        self._endpoint_latencies: Dict[str, list] = {}
        # 全局延迟历史
        self._all_latencies: list = []
        # 每个端点的请求计数
        self._endpoint_counts: Dict[str, int] = {}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and collect metrics."""
        if not settings.enable_metrics:
            return await call_next(request)

        start_time = time.time()
        self.request_count += 1
        endpoint = request.url.path

        # 更新端点计数
        self._endpoint_counts[endpoint] = self._endpoint_counts.get(endpoint, 0) + 1

        try:
            response = await call_next(request)
            duration = time.time() - start_time
            duration_ms = duration * 1000
            self.request_duration_total += duration

            # 记录延迟
            self._record_latency(endpoint, duration_ms)

            # 慢请求检测
            if duration_ms > self.SLOW_REQUEST_THRESHOLD_MS:
                logger.warning(
                    f"[Metrics] ⚠️ 慢请求告警: {request.method} {endpoint} "
                    f"耗时 {duration_ms:.0f}ms (阈值: {self.SLOW_REQUEST_THRESHOLD_MS}ms)"
                )

            # Add metrics headers
            response.headers["X-Request-Count"] = str(self.request_count)
            response.headers["X-Request-Duration"] = f"{duration:.3f}"

            return response

        except Exception as e:
            self.error_count += 1
            duration = time.time() - start_time
            duration_ms = duration * 1000
            self.request_duration_total += duration
            self._record_latency(endpoint, duration_ms)
            raise

    def _record_latency(self, endpoint: str, latency_ms: float) -> None:
        """记录延迟数据到全局和端点级别"""
        # 全局
        self._all_latencies.append(latency_ms)
        if len(self._all_latencies) > 1000:
            self._all_latencies = self._all_latencies[-1000:]

        # 端点级别
        if endpoint not in self._endpoint_latencies:
            self._endpoint_latencies[endpoint] = []
        self._endpoint_latencies[endpoint].append(latency_ms)
        if len(self._endpoint_latencies[endpoint]) > 500:
            self._endpoint_latencies[endpoint] = self._endpoint_latencies[endpoint][-500:]

    @staticmethod
    def _percentile(data: list, p: int) -> float:
        """计算第 p 百分位数"""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        idx = int(len(sorted_data) * p / 100)
        return round(sorted_data[min(idx, len(sorted_data) - 1)], 1)

    def get_metrics(self) -> dict:
        """Get collected metrics with percentile latency breakdown."""
        avg_duration = (
            self.request_duration_total / self.request_count
            if self.request_count > 0 else 0
        )

        # 全局延迟分位数
        global_latency = {
            "avg_ms": round(avg_duration * 1000, 1),
            "p50_ms": self._percentile(self._all_latencies, 50),
            "p95_ms": self._percentile(self._all_latencies, 95),
            "p99_ms": self._percentile(self._all_latencies, 99),
        }

        # 按端点的延迟分位数
        endpoint_latency = {}
        for ep, latencies in self._endpoint_latencies.items():
            endpoint_latency[ep] = {
                "count": self._endpoint_counts.get(ep, 0),
                "p50_ms": self._percentile(latencies, 50),
                "p95_ms": self._percentile(latencies, 95),
                "p99_ms": self._percentile(latencies, 99),
            }

        return {
            "total_requests": self.request_count,
            "total_errors": self.error_count,
            "error_rate": round(self.error_count / max(self.request_count, 1), 4),
            "global_latency": global_latency,
            "endpoint_latency": endpoint_latency,
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