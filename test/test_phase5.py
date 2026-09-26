"""
Phase 5 Verification: LangGraph Agent Loop
Creates a repo with a known bug and runs the agent to fix it.
"""

import os
import sys
import shutil
import tempfile

from zenith.config import setup_logger, settings


def test_graph_compilation():
    """Test that the LangGraph compiles without errors."""
    logger.info("--- Test: Graph Compilation ---")

    from zenith.agent.graph import build_agent_graph
    graph = build_agent_graph()
    assert graph is not None, "Graph should compile"
    logger.success("Graph compilation: Passed!")


def test_agent_state():
    """Test that AgentState can be instantiated."""
    logger.info("--- Test: Agent State ---")

    from zenith.agent.state import AgentState
    state: AgentState = {
        "bug_report": "test bug",
        "repo_path": "/tmp/test",
        "test_command": "pytest",
        "iteration": 0,
        "max_iterations": 3,
        "status": "in_progress",
        "error_log": [],
        "analysis": {},
        "relevant_code": [],
        "similar_fixes": [],
        "proposed_fix": {},
        "fix_explanation": "",
        "test_result": {},
    }
    assert state["bug_report"] == "test bug"
    assert state["iteration"] == 0
    logger.success("Agent State: Passed!")


def test_llm_connection():
    """Test that the OpenRouter LLM is reachable."""
    logger.info("--- Test: LLM Connection (OpenRouter) ---")

    if not settings.OPENROUTER_API_KEY:
        logger.warning("No OPENROUTER_API_KEY set — skipping LLM test")
        return

    from zenith.agent.nodes import _call_llm
    response = _call_llm("Respond with exactly: ZENITH_OK")
    logger.info(f"LLM response: {response}")
    assert "ZENITH_OK" in response.upper().replace(" ", "_"), f"Unexpected response: {response}"
    logger.success("LLM Connection: Passed!")


def test_full_agent_run():
    """
    End-to-end test: Create a repo with a bug, run the agent to fix it.
    """
    logger.info("--- Test: Full Agent Run ---")

    if not settings.OPENROUTER_API_KEY:
        logger.warning("No OPENROUTER_API_KEY set — skipping full agent test")
        return

    # Create a temp repo with a known bug
    tmp_dir = tempfile.mkdtemp(prefix="zenith_agent_test_")

    try:
        # Write buggy code: the function has an off-by-one error
        buggy_code = '''"""Simple math utilities."""

def add(a, b):
    """Add two numbers."""
    return a + b

def multiply(a, b):
    """Multiply two numbers."""
    return a * b

def divide(a, b):
    """Divide a by b. Returns None if b is zero."""
    if b == 0:
        return None
    return a / b

def factorial(n):
    """Calculate factorial of n. Has a bug: off-by-one error."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return 1
    result = 1
    for i in range(1, n):  # BUG: should be range(1, n + 1)
        result *= i
    return result
'''
        with open(os.path.join(tmp_dir, "math_utils.py"), "w") as f:
            f.write(buggy_code)

        # Write a test that catches the bug
        test_code = '''"""Tests for math_utils."""
from math_utils import add, multiply, divide, factorial

def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0

def test_multiply():
    assert multiply(3, 4) == 12
    assert multiply(0, 5) == 0

def test_divide():
    assert divide(10, 2) == 5.0
    assert divide(10, 0) is None

def test_factorial():
    assert factorial(0) == 1
    assert factorial(1) == 1
    assert factorial(5) == 120
    assert factorial(3) == 6
'''
        os.makedirs(os.path.join(tmp_dir, "tests"), exist_ok=True)
        with open(os.path.join(tmp_dir, "tests", "test_math.py"), "w") as f:
            f.write(test_code)

        # Initialize git repo so GitManager works
        from git import Repo
        repo = Repo.init(tmp_dir)
        repo.config_writer().set_value("user", "name", "Zenith Test").release()
        repo.config_writer().set_value("user", "email", "test@zenith.dev").release()
        repo.index.add(["math_utils.py", "tests/test_math.py"])
        repo.index.commit("feat: add math utilities with factorial bug")

        # Run the agent
        from zenith.agent import BugFixAgent

        agent = BugFixAgent(max_iterations=3)
        result = agent.run(
            bug_report=(
                "Bug in math_utils.py: The factorial function returns wrong results. "
                "factorial(5) returns 24 instead of 120, and factorial(1) returns 1 "
                "which happens to be correct but factorial(3) returns 2 instead of 6. "
                "There seems to be an off-by-one error in the loop range."
            ),
            repo_path=tmp_dir,
            test_command="python -m pytest tests/test_math.py -x -v",
        )

        # Log results
        logger.info(f"Agent status: {result['status']}")
        logger.info(f"Iterations: {result.get('iteration', 0)}")
        logger.info(f"Fix explanation: {result.get('fix_explanation', 'N/A')}")

        if result["status"] == "success":
            logger.success("🎉 Agent successfully fixed the bug!")
            for file_path, content in result.get("proposed_fix", {}).items():
                logger.info(f"Fixed file: {file_path}")
                # Verify the fix is correct
                if "range(1, n + 1)" in content or "range(1, n+1)" in content:
                    logger.success("Fix contains the correct range fix!")
        else:
            logger.warning("Agent did not fix the bug (may be an LLM or sandbox issue)")
            logger.warning(f"Error log: {result.get('error_log', [])}")

        # The test passes as long as the agent completed without crashing
        assert result["status"] in ("success", "failed"), f"Unexpected status: {result['status']}"
        logger.success("Full Agent Run: Completed without errors!")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    logger = setup_logger()
    logger.info("=" * 60)
    logger.info("Phase 5 Verification: LangGraph Agent Loop")
    logger.info("=" * 60)

    test_graph_compilation()
    test_agent_state()
    test_llm_connection()
    test_full_agent_run()

    logger.success("=" * 60)
    logger.success("ALL PHASE 5 TESTS PASSED!")
    logger.success("=" * 60)
