#!/usr/bin/env python3
"""
Main entry point for AegisIsle application.
"""

import uvicorn
from src.aegis_isle.core.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "src.aegis_isle.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload and settings.debug,
        log_level=settings.log_level.lower(),
        workers=1 if settings.debug else 4
    )