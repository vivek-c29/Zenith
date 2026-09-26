"""
Data models for Git operations.
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class ChangeType(str, Enum):
    """Type of file change in a git diff."""
    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"
    RENAMED = "renamed"


class FileChange(BaseModel):
    """Represents a single changed file in a git diff."""
    file_path: str = Field(description="Path to the changed file (relative to repo root)")
    change_type: ChangeType = Field(description="Type of change")
    diff: str = Field(default="", description="Unified diff text for this file")
    old_path: Optional[str] = Field(default=None, description="Original path if renamed")


class RepoInfo(BaseModel):
    """Summary information about a git repository."""
    path: str = Field(description="Absolute path to the repository root")
    active_branch: str = Field(description="Currently checked-out branch name")
    remote_url: Optional[str] = Field(default=None, description="URL of the 'origin' remote")
    is_dirty: bool = Field(description="Whether there are uncommitted changes")
    untracked_files: List[str] = Field(default_factory=list, description="List of untracked files")
    recent_commits: List[str] = Field(default_factory=list, description="Recent commit messages (last 5)")
