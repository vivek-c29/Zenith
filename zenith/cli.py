"""
Zenith CLI — Command-line entry point for CI/CD integration.

Usage (inside a GitHub Action):
    python -m zenith.cli --repo . --test-command "python -m pytest -x -v"
"""

import argparse
import subprocess
import sys
import os

from loguru import logger


def main():
    parser = argparse.ArgumentParser(
        description="Zenith: Autonomous AI Bug-Fixing Agent"
    )
    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="Path to the git repository to analyze.",
    )
    parser.add_argument(
        "--test-command",
        type=str,
        default="python -m pytest -x -v",
        help="The test command to run (default: python -m pytest -x -v).",
    )
    parser.add_argument(
        "--base-branch",
        type=str,
        default="main",
        help="The base branch to diff against (default: main).",
    )
    parser.add_argument(
        "--bug-report",
        type=str,
        default=None,
        help="A text description of a bug to fix (Reactive Mode).",
    )
    args = parser.parse_args()

    repo_path = os.path.abspath(args.repo)
    test_command = args.test_command
    base_branch = args.base_branch
    bug_report = args.bug_report

    logger.info("=" * 60)
    if bug_report:
        logger.info("🤖 Zenith Support Mode — Reactive Bug Fix")
        logger.info(f"   Bug: {bug_report}")
    else:
        logger.info("🤖 Zenith CI/CD Mode — Autonomous Safety Net")
    logger.info(f"   Repo: {repo_path}")
    logger.info(f"   Test: {test_command}")
    logger.info(f"   Base: {base_branch}")
    logger.info("=" * 60)

    # ── Reactive Mode: Bypass CI/CD and go straight to fixing ──
    if bug_report:
        logger.info("📋 Step 1: Activating Zenith Multi-Agent Team...")
        from zenith.agent.graph import BugFixAgent
        agent = BugFixAgent(max_iterations=3)
        final_state = agent.run(
            repo_path=repo_path,
            bug_report=bug_report,
            test_command=test_command,
        )
        status = final_state.get("status", "unknown")
        if status == "success":
            logger.success("🎉 Zenith successfully fixed the reported bug!")
            sys.exit(0)
        else:
            logger.error("💀 Zenith could not fix the bug automatically.")
            sys.exit(1)

    # ── Step 1: Run the test suite ────────────────────────────────
    logger.info("📋 Step 1: Running test suite...")
    result = subprocess.run(
        test_command,
        shell=True,
        cwd=repo_path,
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        logger.success("✅ All tests passed! Checking for silent logic bugs...")
        from zenith.git_manager import GitManager
        gm = GitManager(repo_path)
        git_diff = gm.get_branch_diff(target_branch=base_branch)
        if not git_diff:
            try:
                git_diff = gm.repo.git.diff("HEAD~1")
            except Exception:
                git_diff = ""
                
        if git_diff:
            from zenith.agent.graph import AuditAgent
            auditor = AuditAgent()
            audit_result = auditor.run(repo_path, git_diff)
            
            if audit_result.get("auditor_found_bug"):
                ci_failure_log = f"Silent Logic Bug Detected:\n{audit_result.get('auditor_report')}"
                logger.warning(ci_failure_log)
                logger.warning("Activating BugFixAgent to fix the newly generated test case...")
            else:
                logger.success("✅ Audit passed! No silent bugs found.")
                sys.exit(0)
        else:
            logger.info("No git diff found to audit.")
            sys.exit(0)
    else:
        # Tests failed — capture the failure log
        ci_failure_log = result.stdout + "\n" + result.stderr
        logger.warning(f"❌ Tests failed (exit code {result.returncode}). Activating agents...")
        
        # ── Step 2: Extract the git diff ──────────────────────────────
        logger.info("📋 Step 2: Extracting git diff...")
        from zenith.git_manager import GitManager
        gm = GitManager(repo_path)
        git_diff = gm.get_branch_diff(target_branch=base_branch)
        if not git_diff:
            try:
                git_diff = gm.repo.git.diff("HEAD~1")
            except Exception:
                git_diff = "Could not extract diff."
                
        logger.info(f"   Diff length: {len(git_diff)} chars")

    # ── Step 3: Activate the Multi-Agent Team ─────────────────────
    logger.info("📋 Step 3: Activating Zenith Multi-Agent Team...")
    from zenith.agent.graph import BugFixAgent

    agent = BugFixAgent(max_iterations=3)
    final_state = agent.run(
        repo_path=repo_path,
        ci_failure_log=ci_failure_log,
        git_diff=git_diff,
        test_command=test_command,
    )

    # ── Step 4: Report result ─────────────────────────────────────
    status = final_state.get("status", "unknown")
    if status == "success":
        logger.success("🎉 Zenith successfully fixed the failing tests!")
        sys.exit(0)
    else:
        logger.error("💀 Zenith could not fix the issue automatically.")
        sys.exit(1)


if __name__ == "__main__":
    main()
