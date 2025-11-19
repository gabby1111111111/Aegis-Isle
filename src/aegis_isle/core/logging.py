"""
Logging configuration for AegisIsle
"""

import sys
from loguru import logger

from .config import settings


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
    )

    # Add file handler for production
    if settings.environment == "production":
        logger.add(
            "logs/aegis_isle.log",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            rotation="1 day",
            retention="30 days",
            compression="zip",
        )


# Configure logging on import
configure_logging()

# Export configured logger
__all__ = ["logger"]