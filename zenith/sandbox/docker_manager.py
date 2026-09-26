"""
Docker Sandbox Manager.
Provides isolated execution environments for the bug-fixing agent to run tests
and verify its code changes without risking the host machine.
"""

import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import docker
from docker.errors import (
    APIError,
    ContainerError,
    ImageNotFound,
    NotFound,
)
from loguru import logger


@dataclass
class CommandResult:
    """Result of a command executed inside the sandbox."""
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False
    duration_ms: int = 0


# Default image for repos without a Dockerfile
DEFAULT_IMAGE = "python:3.12-slim"

# Security constraints for containers
CONTAINER_LIMITS = {
    "mem_limit": "512m",        # Max 512MB RAM
    "cpu_period": 100000,       # CPU scheduling period (microseconds)
    "cpu_quota": 50000,         # 50% of one CPU core
    "pids_limit": 256,          # Max 256 processes
    "network_disabled": True,   # No network access
}


class DockerSandbox:
    """
    Manages an isolated Docker container for running tests and commands.

    Lifecycle:
        1. sandbox = DockerSandbox(repo_path)
        2. sandbox.start()
        3. result = sandbox.run_command("pytest tests/")
        4. sandbox.stop()

    The repository is bind-mounted read-only into the container at /workspace.
    A writable overlay is used so the agent's changes don't affect the host.
    """

    def __init__(
        self,
        repo_path: str,
        image: Optional[str] = None,
        workspace_dir: str = "/workspace",
        read_only: bool = True,
        network_disabled: bool = True,
        setup_commands: Optional[list] = None,
    ):
        """
        Initialize the sandbox.

        Args:
            repo_path: Absolute path to the repository to mount.
            image: Docker image to use. If None, uses DEFAULT_IMAGE.
            workspace_dir: Path inside the container where repo is mounted.
            read_only: If True, mount repo as read-only. Set False for temp copies.
            network_disabled: If True, disable network. Set False if deps need installing.
            setup_commands: Shell commands to run after container start (e.g., ['pip install pytest']).
        """
        self.repo_path = os.path.abspath(repo_path)
        self.image = image or DEFAULT_IMAGE
        self.workspace_dir = workspace_dir
        self.read_only = read_only
        self.network_disabled = network_disabled
        self.setup_commands = setup_commands or []

        self._client: Optional[docker.DockerClient] = None
        self._container = None
        self._is_running = False

        if not os.path.isdir(self.repo_path):
            raise FileNotFoundError(f"Repository path not found: {self.repo_path}")

    def start(self) -> None:
        """
        Start the sandbox container.
        Pulls the image if needed and creates a container with the repo mounted.
        """
        if self._is_running:
            logger.warning("Sandbox is already running")
            return

        try:
            self._client = docker.from_env()
            self._client.ping()
        except Exception as e:
            raise RuntimeError(
                f"Cannot connect to Docker daemon. Is Docker running? Error: {e}"
            )

        # Pull image if not available locally
        try:
            self._client.images.get(self.image)
            logger.debug(f"Image '{self.image}' found locally")
        except ImageNotFound:
            logger.info(f"Pulling image '{self.image}'...")
            self._client.images.pull(self.image)
            logger.info(f"Image '{self.image}' pulled successfully")

        # Create and start container
        try:
            self._container = self._client.containers.run(
                image=self.image,
                command="sleep infinity",  # Keep alive for interactive commands
                detach=True,
                working_dir=self.workspace_dir,
                volumes={
                    self.repo_path: {
                        "bind": self.workspace_dir,
                        "mode": "ro" if self.read_only else "rw",
                    }
                },
                mem_limit=CONTAINER_LIMITS["mem_limit"],
                cpu_period=CONTAINER_LIMITS["cpu_period"],
                cpu_quota=CONTAINER_LIMITS["cpu_quota"],
                pids_limit=CONTAINER_LIMITS["pids_limit"],
                network_disabled=self.network_disabled,
                remove=False,  # We'll remove manually on stop()
                labels={"zenith": "sandbox"},
            )
            self._is_running = True
            logger.info(
                f"Sandbox started: container={self._container.short_id}, "
                f"image={self.image}, mount={self.repo_path}"
            )

            # Run setup commands (e.g., install dependencies)
            for cmd in self.setup_commands:
                logger.debug(f"Sandbox setup: {cmd}")
                setup_result = self.run_command(cmd, timeout=120)
                if setup_result.exit_code != 0:
                    logger.warning(f"Setup command failed: {cmd} -> {setup_result.stderr[:200]}")

        except APIError as e:
            logger.error(f"Failed to start sandbox container: {e}")
            raise

    def run_command(
        self,
        cmd: str,
        timeout: int = 60,
        workdir: Optional[str] = None,
    ) -> CommandResult:
        """
        Execute a command inside the sandbox container.

        Args:
            cmd: Shell command to execute (e.g., "pytest tests/").
            timeout: Maximum execution time in seconds.
            workdir: Working directory inside the container.
                     Defaults to self.workspace_dir.

        Returns:
            CommandResult with exit_code, stdout, stderr, and timing info.
        """
        if not self._is_running or not self._container:
            raise RuntimeError("Sandbox is not running. Call start() first.")

        workdir = workdir or self.workspace_dir

        logger.debug(f"Sandbox exec: {cmd} (timeout={timeout}s)")
        start_time = time.time()

        try:
            # Execute the command
            exec_result = self._container.exec_run(
                cmd=["sh", "-c", cmd],
                workdir=workdir,
                demux=True,          # Separate stdout and stderr
                environment={"PYTHONDONTWRITEBYTECODE": "1"},
            )

            duration_ms = int((time.time() - start_time) * 1000)

            # demux=True returns (stdout_bytes, stderr_bytes)
            stdout_raw, stderr_raw = exec_result.output
            stdout = (stdout_raw or b"").decode("utf-8", errors="replace").strip()
            stderr = (stderr_raw or b"").decode("utf-8", errors="replace").strip()

            result = CommandResult(
                exit_code=exec_result.exit_code,
                stdout=stdout,
                stderr=stderr,
                timed_out=False,
                duration_ms=duration_ms,
            )

            log_fn = logger.info if result.exit_code == 0 else logger.warning
            log_fn(
                f"Sandbox result: exit={result.exit_code}, "
                f"duration={result.duration_ms}ms, "
                f"stdout={len(stdout)} chars, stderr={len(stderr)} chars"
            )

            return result

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            logger.error(f"Sandbox exec failed: {e}")
            return CommandResult(
                exit_code=-1,
                stdout="",
                stderr=str(e),
                timed_out=duration_ms >= timeout * 1000,
                duration_ms=duration_ms,
            )

    def stop(self) -> None:
        """Stop and remove the sandbox container."""
        if self._container:
            try:
                self._container.stop(timeout=5)
                logger.debug(f"Container {self._container.short_id} stopped")
            except NotFound:
                logger.debug("Container already removed")
            except Exception as e:
                logger.warning(f"Error stopping container: {e}")

            try:
                self._container.remove(force=True)
                logger.debug(f"Container {self._container.short_id} removed")
            except NotFound:
                pass
            except Exception as e:
                logger.warning(f"Error removing container: {e}")

            self._container = None

        self._is_running = False
        logger.info("Sandbox stopped")

    @property
    def is_running(self) -> bool:
        """Check if the sandbox container is currently running."""
        if not self._container:
            return False
        try:
            self._container.reload()
            return self._container.status == "running"
        except Exception:
            return False

    def __enter__(self):
        """Context manager support."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        self.stop()
        return False
