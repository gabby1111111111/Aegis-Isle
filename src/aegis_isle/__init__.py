"""
AegisIsle - Multi-Agent Collaborative RAG System

A comprehensive enterprise-grade RAG system with multi-agent orchestration,
supporting various LLM providers, vector databases, and multimodal processing.
"""

__version__ = "0.1.0"
__author__ = "AegisIsle Team"
__email__ = "team@aegisisle.com"

from .core.config import settings
from .core.logging import logger

__all__ = ["settings", "logger"]