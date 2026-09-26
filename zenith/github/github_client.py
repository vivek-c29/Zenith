"""
GitHub client for Zenith to autonomously open Pull Requests.
"""

from typing import Optional, Any
from loguru import logger
from github import Github
from github.GithubException import GithubException

from zenith.config import settings


class GitHubClient:
    """
    Wrapper around PyGithub to interact with repositories and pull requests.
    """

    def __init__(self, token: Optional[str] = None):
        """
        Initialize the client.
        Args:
            token: GitHub PAT. If None, tries to load from settings.GITHUB_TOKEN.
        """
        self.token = token or settings.GITHUB_TOKEN
        if not self.token:
            raise ValueError("GITHUB_TOKEN is not set.")
        self.client = Github(self.token)

    def create_pull_request(
        self,
        repo_full_name: str,
        head_branch: str,
        base_branch: str,
        title: str,
        body: str,
    ) -> tuple[Optional[str], Any]:
        """
        Create a Pull Request.

        Args:
            repo_full_name: e.g. 'octocat/Hello-World'
            head_branch: The branch containing the fix (e.g. 'zenith/fix-bug')
            base_branch: The target branch (e.g. 'main')
            title: Title of the PR
            body: Markdown body of the PR

        Returns:
            Tuple of (PR URL, PR Object) or (None, None) if it fails.
        """
        try:
            repo = self.client.get_repo(repo_full_name)
            logger.info(f"Creating PR on {repo_full_name}: {head_branch} -> {base_branch}")
            
            pr = repo.create_pull(
                title=title,
                body=body,
                head=head_branch,
                base=base_branch,
            )
            
            logger.success(f"PR created successfully: {pr.html_url}")
            return pr.html_url, pr

        except GithubException as e:
            logger.error(f"GitHub API Error creating PR: {e.data}")
            return None, None
        except Exception as e:
            logger.error(f"Unexpected error creating PR: {e}")
            return None, None
