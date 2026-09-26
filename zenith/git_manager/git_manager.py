"""
Git repository manager using GitPython.
Handles repo inspection, file listing, diffs, branching, and commits.
"""

import os
from typing import List, Optional, Set

from git import Repo, InvalidGitRepositoryError, GitCommandError
from loguru import logger

from .models import ChangeType, FileChange, RepoInfo


class GitManager:
    """
    Manages git operations on a repository for the bug-fixing agent.
    Provides methods to inspect, branch, diff, commit, and push.
    """

    def __init__(self, repo_path: str):
        """
        Initialize GitManager with a path to a git repository.

        Args:
            repo_path: Absolute path to the git repository root.

        Raises:
            InvalidGitRepositoryError: If the path is not a valid git repo.
        """
        self.repo_path = os.path.abspath(repo_path)
        try:
            self.repo = Repo(self.repo_path)
            logger.info(f"Opened git repository: {self.repo_path}")
        except InvalidGitRepositoryError:
            logger.error(f"Not a valid git repository: {self.repo_path}")
            raise

    def get_repo_info(self) -> RepoInfo:
        """Get summary information about the repository."""
        # Active branch
        try:
            active_branch = self.repo.active_branch.name
        except TypeError:
            # Detached HEAD state
            active_branch = f"detached@{self.repo.head.commit.hexsha[:8]}"

        # Remote URL
        remote_url = None
        try:
            if self.repo.remotes:
                remote_url = self.repo.remotes.origin.url
        except (AttributeError, ValueError):
            pass

        # Recent commits
        recent_commits = []
        try:
            for commit in self.repo.iter_commits(max_count=5):
                recent_commits.append(
                    f"{commit.hexsha[:8]} - {commit.message.strip().split(chr(10))[0]}"
                )
        except Exception:
            pass

        return RepoInfo(
            path=self.repo_path,
            active_branch=active_branch,
            remote_url=remote_url,
            is_dirty=self.repo.is_dirty(untracked_files=True),
            untracked_files=self.repo.untracked_files,
            recent_commits=recent_commits,
        )

    def list_files(self, extensions_filter: Optional[Set[str]] = None) -> List[str]:
        """
        List all tracked files in the repository.

        Args:
            extensions_filter: Optional set of extensions to include (e.g. {'.py', '.js'}).
                               If None, returns all tracked files.

        Returns:
            List of file paths relative to the repo root.
        """
        try:
            tracked_files = [item.path for item in self.repo.tree().traverse() if item.type == "blob"]
        except Exception as e:
            logger.error(f"Failed to list files: {e}")
            return []

        if extensions_filter:
            tracked_files = [
                f for f in tracked_files
                if os.path.splitext(f)[1].lower() in extensions_filter
            ]

        logger.debug(f"Listed {len(tracked_files)} files (filter: {extensions_filter})")
        return sorted(tracked_files)

    def get_diff(self, staged_only: bool = False) -> List[FileChange]:
        """
        Get current uncommitted changes as a list of FileChange objects.

        Args:
            staged_only: If True, only return staged (indexed) changes.

        Returns:
            List of FileChange objects representing each changed file.
        """
        changes: List[FileChange] = []

        try:
            if staged_only:
                # Staged changes: diff between HEAD and index
                diffs = self.repo.index.diff("HEAD")
            else:
                # All unstaged changes: diff between index and working tree
                diffs = self.repo.index.diff(None)

            for diff_item in diffs:
                # Determine change type
                if diff_item.new_file:
                    change_type = ChangeType.ADDED
                elif diff_item.deleted_file:
                    change_type = ChangeType.DELETED
                elif diff_item.renamed_file:
                    change_type = ChangeType.RENAMED
                else:
                    change_type = ChangeType.MODIFIED

                # Get diff text
                try:
                    diff_text = diff_item.diff.decode("utf-8", errors="replace") if diff_item.diff else ""
                except Exception:
                    diff_text = ""

                file_path = diff_item.b_path or diff_item.a_path
                old_path = diff_item.a_path if diff_item.renamed_file else None

                changes.append(
                    FileChange(
                        file_path=file_path,
                        change_type=change_type,
                        diff=diff_text,
                        old_path=old_path,
                    )
                )

            # Also include untracked files as ADDED
            if not staged_only:
                for untracked in self.repo.untracked_files:
                    changes.append(
                        FileChange(
                            file_path=untracked,
                            change_type=ChangeType.ADDED,
                            diff="<new untracked file>",
                        )
                    )

        except Exception as e:
            logger.error(f"Failed to get diff: {e}")

        logger.info(f"Found {len(changes)} changes (staged_only={staged_only})")
        return changes

    def get_branch_diff(self, target_branch: str = "main", max_diff_length: int = 5000) -> Optional[str]:
        """
        Get the raw diff between the current branch and a target branch.
        Useful for CI/CD context where we want to know what a PR introduces.

        Args:
            target_branch: The base branch (e.g., 'main' or 'master').
            max_diff_length: Truncate diff if it's too large.

        Returns:
            The raw unified diff string, or None if it fails.
        """
        try:
            # e.g., git diff main...HEAD
            # This shows changes on the current branch since it diverged from main
            diff_text = self.repo.git.diff(f"{target_branch}...HEAD")
            if len(diff_text) > max_diff_length:
                diff_text = diff_text[:max_diff_length] + f"\n... [diff truncated to {max_diff_length} chars]"
            return diff_text
        except Exception as e:
            logger.error(f"Failed to get branch diff against {target_branch}: {e}")
            return None

    def get_file_content(self, file_path: str, ref: str = "HEAD") -> Optional[str]:
        """
        Read file content at a specific git ref (commit, branch, tag).

        Args:
            file_path: Path relative to the repo root.
            ref: Git ref to read from (default: HEAD).

        Returns:
            File content as string, or None if not found.
        """
        try:
            commit = self.repo.commit(ref)
            blob = commit.tree / file_path
            return blob.data_stream.read().decode("utf-8", errors="replace")
        except (KeyError, GitCommandError) as e:
            logger.warning(f"File {file_path} not found at ref {ref}: {e}")
            return None

    def create_branch(self, branch_name: str, checkout: bool = True) -> str:
        """
        Create a new branch and optionally check it out.

        Args:
            branch_name: Name of the new branch.
            checkout: Whether to switch to the new branch.

        Returns:
            The name of the created branch.
        """
        try:
            new_branch = self.repo.create_head(branch_name)
            if checkout:
                new_branch.checkout()
            logger.info(f"Created branch '{branch_name}' (checkout={checkout})")
            return branch_name
        except Exception as e:
            logger.error(f"Failed to create branch '{branch_name}': {e}")
            raise

    def stage_and_commit(self, files: List[str], message: str) -> str:
        """
        Stage specific files and create a commit.

        Args:
            files: List of file paths (relative to repo root) to stage.
            message: Commit message.

        Returns:
            The hexsha of the new commit.
        """
        try:
            self.repo.index.add(files)
            commit = self.repo.index.commit(message)
            logger.info(f"Committed {len(files)} files: {commit.hexsha[:8]} - {message}")
            return commit.hexsha
        except Exception as e:
            logger.error(f"Failed to commit: {e}")
            raise

    def push(self, remote_name: str = "origin", branch: Optional[str] = None) -> bool:
        """
        Push the current branch to a remote.

        Args:
            remote_name: Name of the remote (default: 'origin').
            branch: Branch to push. Defaults to the current active branch.

        Returns:
            True if push succeeded, False otherwise.
        """
        try:
            remote = self.repo.remote(remote_name)
            branch = branch or self.repo.active_branch.name
            remote.push(branch)
            logger.info(f"Pushed branch '{branch}' to '{remote_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to push: {e}")
            return False

    def get_commit_history(self, max_count: int = 20, file_path: Optional[str] = None) -> List[dict]:
        """
        Get commit history, optionally filtered to a specific file.

        Args:
            max_count: Maximum number of commits to return.
            file_path: If provided, only return commits that touched this file.

        Returns:
            List of dicts with commit metadata (sha, message, author, date).
        """
        commits = []
        try:
            kwargs = {"max_count": max_count}
            if file_path:
                kwargs["paths"] = file_path

            for commit in self.repo.iter_commits(**kwargs):
                commits.append({
                    "sha": commit.hexsha,
                    "short_sha": commit.hexsha[:8],
                    "message": commit.message.strip(),
                    "author": str(commit.author),
                    "date": commit.committed_datetime.isoformat(),
                    "files_changed": len(commit.stats.files),
                })
        except Exception as e:
            logger.error(f"Failed to get commit history: {e}")

        return commits

    def get_commit_diff(self, commit_sha: str, max_diff_length: int = 2000) -> Optional[str]:
        """
        Get the unified diff text for a specific commit.

        Args:
            commit_sha: The SHA of the commit.
            max_diff_length: Maximum character length for the diff (truncated if longer).

        Returns:
            Unified diff text as a string, or None if unavailable.
        """
        try:
            commit = self.repo.commit(commit_sha)

            if commit.parents:
                # Diff against parent
                diff_index = commit.parents[0].diff(commit, create_patch=True)
            else:
                # Initial commit — diff against empty tree
                diff_index = commit.diff(None, create_patch=True)

            diff_parts = []
            for diff_item in diff_index:
                file_header = f"--- {diff_item.a_path or '/dev/null'}\n+++ {diff_item.b_path or '/dev/null'}"
                diff_text = diff_item.diff.decode("utf-8", errors="replace") if diff_item.diff else ""
                diff_parts.append(f"{file_header}\n{diff_text}")

            full_diff = "\n".join(diff_parts)

            # Truncate if too long
            if len(full_diff) > max_diff_length:
                full_diff = full_diff[:max_diff_length] + "\n... [truncated]"

            return full_diff

        except Exception as e:
            logger.warning(f"Failed to get diff for commit {commit_sha}: {e}")
            return None
