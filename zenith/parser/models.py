"""
Data models for parsed code output.
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class SymbolKind(str, Enum):
    """Kind of code symbol extracted from AST."""
    FUNCTION = "function"
    METHOD = "method"
    CLASS = "class"
    IMPORT = "import"


class CodeSymbol(BaseModel):
    """
    Represents a single extracted code symbol (function, class, method, import).
    """
    name: str = Field(description="Name of the symbol")
    kind: SymbolKind = Field(description="Type of symbol")
    start_line: int = Field(description="Starting line number (1-indexed)")
    end_line: int = Field(description="Ending line number (1-indexed)")
    signature: Optional[str] = Field(default=None, description="Function/method signature")
    docstring: Optional[str] = Field(default=None, description="Docstring if present")
    source_code: str = Field(description="Full source code of the symbol")
    parent_class: Optional[str] = Field(default=None, description="Parent class name if this is a method")


class CodeChunk(BaseModel):
    """
    A structural chunk of code ready for embedding / LLM context.
    Chunks are split along function/class boundaries, not arbitrary line counts.
    """
    chunk_id: str = Field(description="Unique identifier for this chunk")
    file_path: str = Field(description="Path to the source file")
    language: str = Field(description="Programming language")
    symbol_name: Optional[str] = Field(default=None, description="Primary symbol in this chunk")
    symbol_kind: Optional[SymbolKind] = Field(default=None, description="Kind of the primary symbol")
    content: str = Field(description="The actual code content")
    start_line: int = Field(description="Starting line number (1-indexed)")
    end_line: int = Field(description="Ending line number (1-indexed)")
    calls: List[str] = Field(default_factory=list, description="Names of functions called within this chunk")
    token_count: int = Field(default=0, description="Token count for LLM context budgeting")


class ParsedFile(BaseModel):
    """
    Complete parse result for a single source file.
    """
    file_path: str = Field(description="Path to the source file")
    language: str = Field(description="Detected programming language")
    symbols: List[CodeSymbol] = Field(default_factory=list, description="All extracted symbols")
    chunks: List[CodeChunk] = Field(default_factory=list, description="Structural code chunks")
    imports: List[str] = Field(default_factory=list, description="Import statements found")
    total_lines: int = Field(default=0, description="Total number of lines in the file")
