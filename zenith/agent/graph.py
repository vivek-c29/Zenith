"""
LangGraph state machine for the Multi-Agent bug-fixing architecture.
"""

import shutil
from typing import Any, Dict, Optional

from langgraph.graph import StateGraph, END
from loguru import logger

from .state import AgentState
from .nodes import (
    manager_node,
    researcher_agent,
    coder_agent,
    reviewer_agent,
    check_review,
    sandbox_node,
    check_tests,
    error_analyzer_agent,
    raise_pr_node,
    mark_success,
    mark_failed,
)


def build_agent_graph() -> StateGraph:
    """
    Build and compile the LangGraph Multi-Agent state machine.

    Graph structure:
        manager_node → researcher_agent → coder_agent → reviewer_agent
        
        reviewer_agent
            ├── rejected → coder_agent
            └── approved → sandbox_node

        sandbox_node
            ├── success → mark_success → END
            ├── retry   → error_analyzer_agent → coder_agent
            └── failed  → mark_failed → END
    """
    graph = StateGraph(AgentState)

    # ── Add nodes ──────────────────────────────────────────────────────
    graph.add_node("manager_node", manager_node)
    graph.add_node("researcher_agent", researcher_agent)
    graph.add_node("coder_agent", coder_agent)
    graph.add_node("reviewer_agent", reviewer_agent)
    graph.add_node("sandbox_node", sandbox_node)
    graph.add_node("error_analyzer_agent", error_analyzer_agent)
    graph.add_node("raise_pr_node", raise_pr_node)
    graph.add_node("mark_success", mark_success)
    graph.add_node("mark_failed", mark_failed)

    # ── Define edges ───────────────────────────────────────────────────
    # Initial sequence
    graph.set_entry_point("manager_node")
    graph.add_edge("manager_node", "researcher_agent")
    graph.add_edge("researcher_agent", "coder_agent")
    graph.add_edge("coder_agent", "reviewer_agent")

    # Reviewer Routing
    graph.add_conditional_edges(
        "reviewer_agent",
        check_review,
        {
            "approved": "sandbox_node",
            "rejected": "coder_agent",
        },
    )

    # Sandbox Routing
    graph.add_conditional_edges(
        "sandbox_node",
        check_tests,
        {
            "success": "raise_pr_node",
            "retry": "error_analyzer_agent",
            "failed": "mark_failed",
        },
    )

    # Error Analyzer Loop
    graph.add_edge("error_analyzer_agent", "coder_agent")

    # Terminal states
    graph.add_edge("raise_pr_node", "mark_success")
    graph.add_edge("mark_success", END)
    graph.add_edge("mark_failed", END)

    return graph.compile()


class AuditAgent:
    """
    Standalone agent that runs the auditor_node to detect silent logic bugs.
    """
    def __init__(self):
        from .nodes import auditor_node
        self.auditor_node = auditor_node
        
    def run(self, repo_path: str, git_diff: str) -> Dict[str, Any]:
        logger.info("=" * 60)
        logger.info("🕵️‍♂️ Starting Proactive Audit (Tests Passed)")
        logger.info(f"Repo: {repo_path}")
        logger.info("=" * 60)
        
        state: AgentState = {
            "repo_path": repo_path,
            "git_diff": git_diff,
        }
        
        return self.auditor_node(state)


class BugFixAgent:
    """
    High-level interface for the multi-agent bug-fixing system.
    """

    def __init__(self, max_iterations: int = 3):
        self.max_iterations = max_iterations
        self._graph = build_agent_graph()
        logger.info(f"Multi-Agent BugFixAgent initialized (max_iter={max_iterations})")

    def run(
        self,
        repo_path: str,
        bug_report: Optional[str] = None,
        ci_failure_log: Optional[str] = None,
        git_diff: Optional[str] = None,
        test_command: str = "python -m pytest -x -v",
    ) -> Dict[str, Any]:
        """
        Run the full multi-agent pipeline for either a bug report or a CI/CD failure.
        """
        logger.info("=" * 60)
        logger.info("🚀 Starting Multi-Agent Team")
        if bug_report:
            logger.info(f"Mode: Reactive Bug Fix")
            logger.info(f"Bug: {bug_report[:100]}...")
        else:
            logger.info(f"Mode: Proactive CI/CD Regression Check")
            
        logger.info(f"Repo: {repo_path}")
        logger.info(f"Test: {test_command}")
        logger.info("=" * 60)

        initial_state: AgentState = {
            "bug_report": bug_report,
            "ci_failure_log": ci_failure_log,
            "git_diff": git_diff,
            "repo_path": repo_path,
            "test_command": test_command,
            "iteration": 0,
            "max_iterations": self.max_iterations,
            "status": "in_progress",
            "error_log": [],
            "analysis": {},
            "relevant_code": [],
            "similar_fixes": [],
            "strategy_document": "",
            "proposed_fix": {},
            "proposed_patches": [],
            "fix_explanation": "",
            "reviewer_approved": False,
            "reviewer_feedback": "",
            "risk_level": "HIGH",
            "test_result": {},
            "error_analysis": "",
            "_sandbox_path": None,
            "_indexer": None,
        }

        # Run the graph
        final_state = self._graph.invoke(initial_state)

        # Cleanup sandbox temp directory
        sandbox_path = final_state.get("_sandbox_path")
        if sandbox_path:
            try:
                shutil.rmtree(sandbox_path, ignore_errors=True)
            except Exception:
                pass

        # Log summary
        status = final_state.get("status", "unknown")
        iterations = final_state.get("iteration", 0)
        if status == "success":
            logger.success(f"🎉 Multi-Agent Team fixed the bug in {iterations + 1} iteration(s)!")
        else:
            logger.error(f"💀 Team failed after {iterations} iteration(s)")
            for err in final_state.get("error_log", []):
                logger.error(f"  {err}")

        return final_state
