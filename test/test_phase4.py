"""
Phase 4 Verification: Docker Sandbox
Tests container lifecycle, command execution, exit codes, and output capture.
"""

import os
import tempfile
import shutil

from zenith.config import setup_logger
from zenith.sandbox import DockerSandbox


def test_sandbox_lifecycle():
    """Test starting and stopping a sandbox container."""
    logger.info("--- Test: Sandbox Lifecycle ---")

    # Create a temp repo directory
    tmp_dir = tempfile.mkdtemp(prefix="zenith_sandbox_test_")

    try:
        # Write a simple test file
        with open(os.path.join(tmp_dir, "hello.py"), "w") as f:
            f.write('print("Hello from Zenith sandbox!")\n')

        with open(os.path.join(tmp_dir, "test_math.py"), "w") as f:
            f.write(
                "def test_add():\n"
                "    assert 1 + 1 == 2\n\n"
                "def test_fail():\n"
                "    assert 1 + 1 == 3, 'Intentional failure'\n"
            )

        sandbox = DockerSandbox(repo_path=tmp_dir)

        # Test start
        sandbox.start()
        assert sandbox.is_running, "Sandbox should be running after start()"
        logger.info(f"Sandbox is running: {sandbox.is_running}")

        # Test: successful command
        result = sandbox.run_command("python hello.py")
        logger.info(f"Success cmd: exit={result.exit_code}, stdout='{result.stdout}'")
        assert result.exit_code == 0, f"Expected exit code 0, got {result.exit_code}"
        assert "Hello from Zenith sandbox!" in result.stdout, "Output should contain hello message"

        # Test: failing command
        result_fail = sandbox.run_command('python -c "1/0"')
        logger.info(f"Fail cmd: exit={result_fail.exit_code}, stderr='{result_fail.stderr[:100]}'")
        assert result_fail.exit_code != 0, "Division by zero should return non-zero exit code"
        assert "ZeroDivisionError" in result_fail.stderr, "Should capture traceback in stderr"

        # Test: command output with multiple lines
        result_multi = sandbox.run_command("python -c \"for i in range(5): print(f'Line {i}')\"")
        lines = result_multi.stdout.strip().split("\n")
        assert len(lines) == 5, f"Should have 5 lines, got {len(lines)}"
        logger.info(f"Multi-line output: {lines}")

        # Test: ls workspace to verify repo is mounted
        result_ls = sandbox.run_command("ls")
        logger.info(f"Workspace contents: {result_ls.stdout}")
        assert "hello.py" in result_ls.stdout, "hello.py should be visible in /workspace"

        # Test: verify read-only mount (write should fail)
        result_ro = sandbox.run_command("touch /workspace/should_fail.txt")
        logger.info(f"Read-only test: exit={result_ro.exit_code}")
        assert result_ro.exit_code != 0, "Writing to read-only mount should fail"

        # Test: check timing
        result_slow = sandbox.run_command("python -c \"import time; time.sleep(0.5); print('done')\"")
        logger.info(f"Timing test: duration={result_slow.duration_ms}ms")
        assert result_slow.duration_ms >= 400, "Should take at least 400ms"
        assert result_slow.stdout == "done"

        # Test stop
        sandbox.stop()
        assert not sandbox.is_running, "Sandbox should not be running after stop()"

        logger.success("Sandbox Lifecycle: All assertions passed!")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_context_manager():
    """Test the context manager (with statement) usage."""
    logger.info("--- Test: Context Manager ---")

    tmp_dir = tempfile.mkdtemp(prefix="zenith_ctx_test_")

    try:
        with open(os.path.join(tmp_dir, "app.py"), "w") as f:
            f.write("print('context manager works')\n")

        with DockerSandbox(repo_path=tmp_dir) as sandbox:
            assert sandbox.is_running, "Should be running inside context"
            result = sandbox.run_command("python app.py")
            assert result.exit_code == 0
            assert "context manager works" in result.stdout

        # After exiting context, container should be stopped
        assert not sandbox.is_running, "Should be stopped after context exit"

        logger.success("Context Manager: All assertions passed!")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    logger = setup_logger()
    logger.info("=" * 60)
    logger.info("Phase 4 Verification: Docker Sandbox")
    logger.info("=" * 60)

    test_sandbox_lifecycle()
    test_context_manager()

    logger.success("=" * 60)
    logger.success("ALL PHASE 4 TESTS PASSED!")
    logger.success("=" * 60)
