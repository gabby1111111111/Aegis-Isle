"""
Pydantic models for structured state management.

Based on Shujuku's DEFAULT_TABLE_TEMPLATE_ACU schema.
Each UserState contains multiple Sheets (tables), each with metadata and row data.
"""

from typing import Dict, List, Any, Optional, Literal
from pydantic import BaseModel, Field
from enum import Enum


class EditType(str, Enum):
    """Types of table edit operations."""
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"


class SheetMetadata(BaseModel):
    """
    Metadata for a sheet, defining its purpose and update rules.
    
    Corresponds to Shujuku's 'sourceData' field.
    """
    note: str = Field(
        description="Description of the sheet's purpose and column definitions"
    )
    init_node: str = Field(
        alias="initNode",
        description="Instructions for initializing this sheet"
    )
    delete_node: str = Field(
        alias="deleteNode",
        description="Rules for deleting rows from this sheet"
    )
    update_node: str = Field(
        alias="updateNode",
        description="Conditions for updating existing rows"
    )
    insert_node: str = Field(
        alias="insertNode",
        default="",
        description="Conditions for inserting new rows"
    )

    class Config:
        populate_by_name = True


class Sheet(BaseModel):
    """
    A single data sheet (table) in the state system.
    
    Corresponds to one entry in Shujuku's DEFAULT_TABLE_TEMPLATE_ACU.
    """
    uid: str = Field(description="Unique identifier for this sheet")
    name: str = Field(description="Human-readable name of the sheet")
    source_data: SheetMetadata = Field(
        alias="sourceData",
        description="Metadata defining sheet behavior"
    )
    content: List[List[Any]] = Field(
        default_factory=lambda: [[]],
        description="Table data as 2D array. First row is header."
    )
    order_no: int = Field(
        alias="orderNo",
        default=0,
        description="Display order of this sheet"
    )

    class Config:
        populate_by_name = True

    def get_header(self) -> List[str]:
        """Get the header row (first row)."""
        if self.content and len(self.content) > 0:
            return [str(cell) if cell is not None else "" for cell in self.content[0]]
        return []

    def get_rows(self) -> List[List[Any]]:
        """Get data rows (excluding header)."""
        if len(self.content) > 1:
            return self.content[1:]
        return []

    def add_row(self, row: List[Any]) -> None:
        """Add a new data row."""
        self.content.append(row)

    def to_markdown(self) -> str:
        """Convert sheet to markdown table format for LLM context injection."""
        if not self.content or len(self.content) == 0:
            return f"## {self.name}\n(Empty)\n"
        
        lines = [f"## {self.name}\n"]
        header = self.get_header()
        rows = self.get_rows()
        
        if not header:
            return f"## {self.name}\n(No header defined)\n"
        
        # Header
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        
        # Rows
        for row in rows:
            row_str = [str(cell) if cell is not None else "" for cell in row]
            # Pad row to match header length
            while len(row_str) < len(header):
                row_str.append("")
            lines.append("| " + " | ".join(row_str[:len(header)]) + " |")
        
        return "\n".join(lines) + "\n"


class UserState(BaseModel):
    """
    Complete state for a single user/session.
    
    Contains all sheets (Global, Hero, NPC, Inventory, etc.)
    """
    sheets: Dict[str, Sheet] = Field(
        default_factory=dict,
        description="Dictionary of sheets keyed by UID"
    )
    version: int = Field(
        default=1,
        description="State schema version for migration support"
    )
    user_id: str = Field(
        default="default",
        description="User/session identifier"
    )

    def get_sheet_by_name(self, name: str) -> Optional[Sheet]:
        """Get a sheet by its human-readable name."""
        for sheet in self.sheets.values():
            if sheet.name == name:
                return sheet
        return None

    def to_context_string(self) -> str:
        """
        Convert entire state to markdown string for LLM context injection.
        
        Returns:
            Markdown-formatted string containing all sheets
        """
        lines = ["# 当前用户状态 (Current User State)\n"]
        
        # Sort sheets by order_no
        sorted_sheets = sorted(
            self.sheets.values(),
            key=lambda s: s.order_no
        )
        
        for sheet in sorted_sheets:
            lines.append(sheet.to_markdown())
        
        return "\n".join(lines)


class TableEditCommand(BaseModel):
    """
    A single table edit command extracted from LLM output.
    
    Example XML from Shujuku:
    <tableEdit type="insert" sheet="背包物品表" row='["生锈的铁剑", "1", "攻击力+3", "武器"]' />
    """
    edit_type: EditType = Field(
        alias="type",
        description="Type of edit operation"
    )
    sheet_name: str = Field(
        alias="sheet",
        description="Target sheet name"
    )
    row_data: Optional[List[Any]] = Field(
        default=None,
        alias="row",
        description="Row data for insert/update operations"
    )
    condition: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Condition for update/delete (e.g., {'column': 0, 'value': 'Key'})"
    )

    class Config:
        populate_by_name = True
