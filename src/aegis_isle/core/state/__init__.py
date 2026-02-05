"""
State Management Module for Aegis-Isle Stateful Agent.

This module implements structured memory management inspired by Shujuku (神·数据库).
It converts unstructured chat logs into a structured JSON state tree.
"""

from .models import (
    SheetMetadata,
    Sheet,
    UserState,
    TableEditCommand,
    EditType,
)
from .manager import StateManager

__all__ = [
    "SheetMetadata",
    "Sheet",
    "UserState",
    "TableEditCommand",
    "EditType",
    "StateManager",
]
