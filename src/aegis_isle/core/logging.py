"""
Logging configuration for AegisIsle with structured audit logging support.
"""

import json
import sys
from datetime import datetime
from typing import Optional, Dict, Any, Literal
from pathlib import Path

from loguru import logger

from .config import settings


# Audit log levels
AuditLevel = Literal["info", "warning", "error", "critical"]

# Audit event types
AuditEventType = Literal[
    "authentication", "authorization", "data_access", "data_modification",
    "system_configuration", "user_management", "security_event", "api_access",
    "file_operation", "model_inference", "agent_execution", "workflow_execution"
]


class AuditLogger:
    """
    Structured audit logger for security and compliance.

    Generates JSON-formatted audit logs compatible with ELK stack.
    """

    def __init__(self):
        """Initialize audit logger with ELK-compatible JSON formatter."""
        self.audit_logger = logger.bind(audit=True)

        # Ensure audit log directory exists
        audit_log_dir = Path("logs/audit")
        audit_log_dir.mkdir(parents=True, exist_ok=True)

        # Configure audit-specific handler with JSON format
        logger.add(
            "logs/audit/audit_{time:YYYY-MM-DD}.jsonl",
            level="INFO",
            format=self._json_formatter,
            rotation="1 day",
            retention="365 days",  # Keep audit logs for 1 year
            compression="gzip",
            filter=lambda record: record["extra"].get("audit", False),
            serialize=False  # We handle JSON serialization ourselves
        )

    def _json_formatter(self, record: Dict[str, Any]) -> str:
        """
        Format log record as JSON for ELK stack compatibility.

        Args:
            record: Loguru record dictionary

        Returns:
            JSON-formatted log string
        """
        # Extract audit data from record extra
        audit_data = record["extra"].get("audit_data", {})

        # Create ELK-compatible log structure
        log_entry = {
            "@timestamp": record["time"].isoformat(),
            "@version": "1",
            "level": record["level"].name.lower(),
            "logger": "aegis-isle-audit",
            "message": record["message"],
            "service": "aegis-isle",
            "environment": settings.environment,
            **audit_data
        }

        return json.dumps(log_entry, ensure_ascii=False) + "\n"

    def log_event(
        self,
        event_type: AuditEventType,
        action: str,
        level: AuditLevel = "info",
        user_id: Optional[str] = None,
        username: Optional[str] = None,
        resource: Optional[str] = None,
        resource_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        request_id: Optional[str] = None,
        session_id: Optional[str] = None,
        outcome: Optional[Literal["success", "failure", "error"]] = "success",
        error_code: Optional[str] = None,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """
        Log a structured audit event.

        Args:
            event_type: Type of audit event
            action: Specific action performed
            level: Log level for the event
            user_id: ID of the user performing the action
            username: Username of the user
            resource: Resource being accessed/modified
            resource_id: ID of the specific resource
            ip_address: Client IP address
            user_agent: Client user agent string
            request_id: Unique request identifier
            session_id: User session identifier
            outcome: Result of the action
            error_code: Error code if action failed
            error_message: Error message if action failed
            metadata: Additional metadata
            **kwargs: Additional fields
        """
        # Build audit data structure
        audit_data = {
            "event_type": event_type,
            "action": action,
            "outcome": outcome,
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Add user information
        if user_id:
            audit_data["user_id"] = user_id
        if username:
            audit_data["username"] = username

        # Add resource information
        if resource:
            audit_data["resource"] = resource
        if resource_id:
            audit_data["resource_id"] = resource_id

        # Add request context
        if ip_address:
            audit_data["ip_address"] = ip_address
        if user_agent:
            audit_data["user_agent"] = user_agent
        if request_id:
            audit_data["request_id"] = request_id
        if session_id:
            audit_data["session_id"] = session_id

        # Add error information
        if error_code:
            audit_data["error_code"] = error_code
        if error_message:
            audit_data["error_message"] = error_message

        # Add metadata
        if metadata:
            audit_data["metadata"] = metadata

        # Add any additional fields
        audit_data.update(kwargs)

        # Create log message
        message = f"{event_type.upper()}: {action}"
        if outcome == "failure":
            message += f" - FAILED"
        if error_message:
            message += f" - {error_message}"

        # Log the audit event
        self.audit_logger.bind(audit_data=audit_data).log(level.upper(), message)

    def log_authentication(
        self,
        action: str,
        username: str,
        outcome: Literal["success", "failure"] = "success",
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        error_message: Optional[str] = None,
        **kwargs
    ) -> None:
        """Log authentication events."""
        self.log_event(
            event_type="authentication",
            action=action,
            username=username,
            outcome=outcome,
            ip_address=ip_address,
            user_agent=user_agent,
            error_message=error_message,
            level="warning" if outcome == "failure" else "info",
            **kwargs
        )

    def log_authorization(
        self,
        action: str,
        user_id: str,
        username: str,
        resource: str,
        outcome: Literal["success", "failure"] = "success",
        required_permissions: Optional[list] = None,
        **kwargs
    ) -> None:
        """Log authorization events."""
        metadata = {}
        if required_permissions:
            metadata["required_permissions"] = required_permissions

        self.log_event(
            event_type="authorization",
            action=action,
            user_id=user_id,
            username=username,
            resource=resource,
            outcome=outcome,
            metadata=metadata,
            level="warning" if outcome == "failure" else "info",
            **kwargs
        )

    def log_data_access(
        self,
        action: str,
        user_id: str,
        username: str,
        resource: str,
        resource_id: Optional[str] = None,
        query: Optional[str] = None,
        **kwargs
    ) -> None:
        """Log data access events."""
        metadata = {}
        if query:
            metadata["query"] = query

        self.log_event(
            event_type="data_access",
            action=action,
            user_id=user_id,
            username=username,
            resource=resource,
            resource_id=resource_id,
            metadata=metadata,
            **kwargs
        )

    def log_security_event(
        self,
        action: str,
        level: AuditLevel = "warning",
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        threat_type: Optional[str] = None,
        **kwargs
    ) -> None:
        """Log security events."""
        metadata = {}
        if threat_type:
            metadata["threat_type"] = threat_type

        self.log_event(
            event_type="security_event",
            action=action,
            level=level,
            ip_address=ip_address,
            user_agent=user_agent,
            metadata=metadata,
            **kwargs
        )

    def log_api_access(
        self,
        method: str,
        endpoint: str,
        user_id: Optional[str] = None,
        username: Optional[str] = None,
        ip_address: Optional[str] = None,
        status_code: Optional[int] = None,
        response_time_ms: Optional[float] = None,
        request_id: Optional[str] = None,
        **kwargs
    ) -> None:
        """Log API access events."""
        metadata = {
            "http_method": method,
            "endpoint": endpoint,
        }
        if status_code:
            metadata["status_code"] = status_code
        if response_time_ms:
            metadata["response_time_ms"] = response_time_ms

        outcome = "success" if status_code and 200 <= status_code < 400 else "failure"

        self.log_event(
            event_type="api_access",
            action=f"{method} {endpoint}",
            user_id=user_id,
            username=username,
            ip_address=ip_address,
            request_id=request_id,
            outcome=outcome,
            metadata=metadata,
            **kwargs
        )


def configure_logging():
    """Configure logging with Loguru."""
    # Remove default handler
    logger.remove()

    # Add console handler
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
        colorize=True,
        filter=lambda record: not record["extra"].get("audit", False)  # Exclude audit logs
    )

    # Ensure log directory exists
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Add file handler for application logs
    logger.add(
        "logs/application_{time:YYYY-MM-DD}.log",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        rotation="1 day",
        retention="30 days",
        compression="gzip",
        filter=lambda record: not record["extra"].get("audit", False)  # Exclude audit logs
    )

    # Add error-only handler for critical issues
    logger.add(
        "logs/errors_{time:YYYY-MM-DD}.log",
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}\n{exception}",
        rotation="1 day",
        retention="90 days",
        compression="gzip",
        filter=lambda record: not record["extra"].get("audit", False)  # Exclude audit logs
    )


# Configure logging on import
configure_logging()

# Create global audit logger instance
audit_logger = AuditLogger()

# Export configured loggers
__all__ = ["logger", "audit_logger", "AuditLogger"]