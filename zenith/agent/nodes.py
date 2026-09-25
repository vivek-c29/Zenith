"""
Agent node functions for the LangGraph multi-agent state machine.
"""

import os
import json
import shutil
import tempfile
from typing import Any, Dict, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from zenith.config import settings
from zenith.utils.json_utils import safe_parse_json
from zenith.vector_store import CodeIndexer, EmbeddingEngine, LocalVectorStore
from zenith.vector_store.vector_store import create_vector_store
from zenith.git_manager import GitManager
from zenith.sandbox import DockerSandbox

from .state import AgentState
from .prompts import (
    AUDITOR_PROMPT,
    MANAGER_PROMPT,
    CI_MANAGER_PROMPT,
    RESEARCHER_PROMPT,
    CODER_PROMPT,
    REVIEWER_PROMPT,
    ERROR_ANALYZER_PROMPT,
)


def _get_llm(model_override: Optional[str] = None) -> ChatOpenAI:
    model_name = model_override or settings.LLM_MODEL
    logger.debug(f"Instantiating LLM: {model_name}")
    return ChatOpenAI(
        model=model_name,
        api_key=settings.OPENROUTER_API_KEY,
        base_url=settings.OPENROUTER_BASE_URL,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS,
    )


def _call_llm(prompt: str, expect_json: bool = True, model_override: Optional[str] = None) -> str:
    llm = _get_llm(model_override)
    sys_msg = "You are Zenith. " + ("Respond ONLY with valid JSON." if expect_json else "Respond in plain text.")
    messages = [
        SystemMessage(content=sys_msg),
        HumanMessage(content=prompt),
    ]
    return llm.invoke(messages).content


def _parse_llm_json(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    result = safe_parse_json(text)
    if result is None:
        raise ValueError(f"Failed to parse LLM JSON: {text[:200]}")
    return result


# ============================================================
# Agent Nodes
# ============================================================

def auditor_node(state: AgentState) -> Dict[str, Any]:
    """Auditor: Scans git diff for silent logic bugs when tests pass."""
    logger.info("🕵️‍♂️ Auditor Agent: Inspecting git diff for silent logic bugs...")
    
    if not state.get("git_diff"):
        logger.info("  -> No git diff provided. Skipping audit.")
        return {"auditor_found_bug": False}

    prompt = AUDITOR_PROMPT.format(git_diff=state["git_diff"])
    raw_response = _call_llm(prompt)
    analysis = _parse_llm_json(raw_response)
    
    found_bug = analysis.get("found_bug", False)
    if found_bug:
        logger.warning(f"  -> 🚨 Silent Logic Bug Detected! {analysis.get('bug_report')}")
        
        # Save the generated test code to a temp file in the repo so the test suite fails
        test_code = analysis.get("generated_test_code", "")
        if test_code:
            repo_path = state["repo_path"]
            test_file_path = os.path.join(repo_path, "tests", "test_auditor_generated.py")
            os.makedirs(os.path.dirname(test_file_path), exist_ok=True)
            with open(test_file_path, "w", encoding="utf-8") as f:
                f.write(test_code)
            logger.info(f"  -> Wrote generated failing test to {test_file_path}")
            
    else:
        logger.info("  -> ✅ No silent logic bugs detected.")
        
    return {
        "auditor_found_bug": found_bug,
        "auditor_report": analysis.get("bug_report", ""),
        "generated_test_code": analysis.get("generated_test_code", ""),
    }

def manager_node(state: AgentState) -> Dict[str, Any]:
    """Manager: Parses bug report or CI failure log and plans initial steps."""
    logger.info("👨‍💼 Manager Agent: Analyzing intake payload...")
    
    if state.get("ci_failure_log") and state.get("git_diff"):
        logger.info("  -> Mode: Proactive CI/CD Regression Check")
        prompt = CI_MANAGER_PROMPT.format(
            ci_failure_log=state["ci_failure_log"],
            git_diff=state["git_diff"]
        )
    else:
        logger.info("  -> Mode: Reactive Bug Report")
        prompt = MANAGER_PROMPT.format(bug_report=state.get("bug_report", "No report provided."))
        
    raw_response = _call_llm(prompt)
    analysis = _parse_llm_json(raw_response)
    
    logger.info(f"  -> Error type: {analysis.get('error_type')}")
    return {"analysis": analysis}


def researcher_agent(state: AgentState) -> Dict[str, Any]:
    """Researcher: Runs RAG searches and writes a Strategy Document."""
    logger.info("🕵️‍♀️ Researcher Agent: Gathering context...")
    analysis = state.get("analysis", {})
    repo_path = state["repo_path"]

    # 1. Search Codebase
    query = f"{analysis.get('error_message', '')} {analysis.get('root_cause_hypothesis', '')}"
    
    engine = EmbeddingEngine()
    store = create_vector_store(dimension=engine.dimension)
    indexer = CodeIndexer(embedding_engine=engine, vector_store=store)

    # Use Incremental Sync if Pinecone or Neo4j, else Full Index
    backend = store.get_stats().get("backend")
    if backend in ("pinecone", "neo4j"):
        indexer.sync_repository(repo_path, base_branch="main")
    else:
        indexer.index_repository(repo_path, max_files=200)
    relevant_code = indexer.search_code(query, top_k=5, strategy="hybrid")

    # 2. Search Fix History
    try:
        gm = GitManager(repo_path)
        indexer.index_fix_history(gm, max_commits=30)
        similar_fixes = indexer.search_similar_fixes(query, top_k=3, strategy="hybrid")
    except Exception:
        similar_fixes = []

    # Format for LLM
    code_context = ""
    for c in relevant_code:
        meta = c.get("metadata", {})
        code_context += f"### {meta.get('file_path')}\n```\n{meta.get('content')}\n```\n"
        
    fix_context = ""
    for f in similar_fixes:
        meta = f.get("metadata", {})
        fix_context += f"### Fix: {meta.get('message')}\n```diff\n{meta.get('diff')}\n```\n"

    # 3. Write Strategy
    prompt = RESEARCHER_PROMPT.format(
        analysis=json.dumps(analysis, indent=2),
        relevant_code=code_context or "None found.",
        similar_fixes=fix_context or "None found."
    )
    strategy = _call_llm(prompt, expect_json=False)
    
    logger.info("  -> Strategy Document created.")
    return {
        "relevant_code": relevant_code,
        "similar_fixes": similar_fixes,
        "strategy_document": strategy,
        "_indexer": indexer,
    }


def coder_agent(state: AgentState) -> Dict[str, Any]:
    """Coder: Writes the code patch based on strategy & feedback."""
    logger.info(f"👨‍💻 Coder Agent (Iter {state.get('iteration', 0)}): Writing code...")
    
    code_context = ""
    for c in state.get("relevant_code", []):
        meta = c.get("metadata", {})
        code_context += f"### {meta.get('file_path')}\n```\n{meta.get('content')}\n```\n"

    prompt = CODER_PROMPT.format(
        analysis=json.dumps(state.get("analysis", {})),
        strategy_document=state.get("strategy_document", ""),
        relevant_code=code_context,
        reviewer_feedback=state.get("reviewer_feedback", "None"),
        error_analysis=state.get("error_analysis", "None"),
    )

    raw_response = _call_llm(prompt)
    fix_data = _parse_llm_json(raw_response)

    proposed_patches = fix_data.get("patches", [])
    explanation = fix_data.get("explanation", "")
    
    files_modified = list(set(p.get("file_path") for p in proposed_patches))
    logger.info(f"  -> Generated {len(proposed_patches)} patches for {files_modified}")
    return {
        "proposed_patches": proposed_patches,
        "fix_explanation": explanation,
        # Clear old feedback since we've now written new code
        "reviewer_feedback": "",
        "error_analysis": ""
    }


def reviewer_agent(state: AgentState) -> Dict[str, Any]:
    """Reviewer: Critiques the Coder's patch."""
    logger.info("🧐 Reviewer Agent: Inspecting code...")
    
    fix_str = ""
    for patch in state.get("proposed_patches", []):
        fix_str += f"### {patch.get('file_path')}\nSearch:\n```\n{patch.get('search_text')}\n```\nReplace:\n```\n{patch.get('replace_text')}\n```\n"
        
    prompt = REVIEWER_PROMPT.format(
        strategy_document=state.get("strategy_document", ""),
        proposed_patches=fix_str
    )
    
    raw_response = _call_llm(prompt)
    review = _parse_llm_json(raw_response)
    
    approved = review.get("approved", False)
    feedback = review.get("feedback", "")
    risk_level = review.get("risk_level", "HIGH").upper()
    
    if approved:
        logger.info(f"  -> ✅ Code Approved! (Risk: {risk_level}) Sending to Sandbox.")
    else:
        logger.warning(f"  -> ❌ Code Rejected: {feedback[:100]}...")
        
    return {
        "reviewer_approved": approved,
        "reviewer_feedback": feedback,
        "risk_level": risk_level
    }


def check_review(state: AgentState) -> str:
    """Conditional edge router from Reviewer."""
    if state.get("reviewer_approved"):
        return "approved"
    return "rejected"


def sandbox_node(state: AgentState) -> Dict[str, Any]:
    """Sandbox: Applies code and runs tests in Docker."""
    logger.info("🧪 Sandbox Node: Running tests...")
    repo_path = state["repo_path"]
    proposed_fix = state.get("proposed_fix", {})

    sandbox_path = state.get("_sandbox_path")
    if not sandbox_path:
        sandbox_path = tempfile.mkdtemp(prefix="zenith_fix_")
        shutil.copytree(repo_path, sandbox_path, dirs_exist_ok=True)

    for patch in state.get("proposed_patches", []):
        file_path = patch.get("file_path")
        search_text = patch.get("search_text")
        replace_text = patch.get("replace_text")
        
        full_path = os.path.join(sandbox_path, file_path)
        if os.path.exists(full_path):
            with open(full_path, "r") as f:
                content = f.read()
            content = content.replace(search_text, replace_text)
            with open(full_path, "w") as f:
                f.write(content)

    test_command = state.get("test_command", "python -m pytest")
    
    try:
        with DockerSandbox(
            repo_path=sandbox_path,
            read_only=False,
            network_disabled=False,
            setup_commands=[
            "apt-get update && apt-get install -y git",
            "pip install pytest"
        ]
        ) as sandbox:
            result = sandbox.run_command(test_command, timeout=120)

        test_result = {
            "exit_code": result.exit_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

        if result.exit_code == 0:
            logger.success("  -> ✅ Tests PASSED!")
        else:
            logger.error(f"  -> ❌ Tests FAILED (exit={result.exit_code})")
            
        return {"test_result": test_result, "_sandbox_path": sandbox_path}

    except Exception as e:
        logger.error(f"Sandbox error: {e}")
        return {"test_result": {"exit_code": -1, "stdout": "", "stderr": str(e)}}


def check_tests(state: AgentState) -> str:
    """Conditional edge router from Sandbox."""
    exit_code = state.get("test_result", {}).get("exit_code", -1)
    if exit_code == 0:
        return "success"
    
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 3)
    if iteration + 1 >= max_iter:
        return "failed"
        
    return "retry"


def error_analyzer_agent(state: AgentState) -> Dict[str, Any]:
    """Error Analyzer: Reads stack traces and tells Coder what went wrong."""
    logger.info("🚨 Error Analyzer Agent: Diagnosing test failure...")
    
    test_result = state.get("test_result", {})
    fix_str = ""
    for patch in state.get("proposed_patches", []):
        fix_str += f"### {patch.get('file_path')}\nSearch:\n```\n{patch.get('search_text')}\n```\nReplace:\n```\n{patch.get('replace_text')}\n```\n"

    prompt = ERROR_ANALYZER_PROMPT.format(
        proposed_fix=fix_str,
        test_stdout=test_result.get("stdout", "")[-2000:],
        test_stderr=test_result.get("stderr", "")[-2000:]
    )
    
    analysis = _call_llm(prompt, expect_json=False)
    logger.info(f"  -> Diagnosis: {analysis[:100]}...")
    
    return {
        "error_analysis": analysis,
        "iteration": state.get("iteration", 0) + 1
    }


import uuid

def raise_pr_node(state: AgentState) -> Dict[str, Any]:
    """GitHub Node: Pushes the code and opens a Pull Request."""
    logger.info("🐙 GitHub Node: Raising Pull Request...")
    
    repo_path = state["repo_path"]
    proposed_patches = state.get("proposed_patches", [])
    
    if not proposed_patches:
        logger.warning("No fix to PR.")
        return {}

    # We need to apply the fix to the original repo (not just the sandbox)
    # Then branch, commit, push, and PR.
    try:
        from zenith.github import GitHubClient
        gh = GitHubClient()
    except Exception as e:
        logger.error(f"GitHub setup failed (missing token?): {e}")
        return {}
        
    gm = GitManager(repo_path)
    
    # 1. Apply fix to the real repo working directory
    files_to_commit = set()
    for patch in proposed_patches:
        file_path = patch.get("file_path")
        search_text = patch.get("search_text")
        replace_text = patch.get("replace_text")
        
        full_path = os.path.join(repo_path, file_path)
        if os.path.exists(full_path):
            with open(full_path, "r") as f:
                content = f.read()
            content = content.replace(search_text, replace_text)
            with open(full_path, "w") as f:
                f.write(content)
            files_to_commit.add(file_path)
            
    # Generate contextual PR and Commit messages via LLM
    logger.info("  -> Generating commit and PR messages...")
    msg_prompt = (
        "You are a Senior Engineer. Write a conventional commit message and a Pull Request title/body for the following fix.\n"
        f"Bug Analysis: {json.dumps(state.get('analysis', {}))}\n"
        f"Fix Explanation: {state.get('fix_explanation', '')}\n"
        "Respond ONLY with valid JSON: { \"commit_message\": \"...\", \"pr_title\": \"...\", \"pr_body\": \"...\" }"
    )
    msg_response = _call_llm(msg_prompt, model_override="stealth/space-bunny-alpha")
    msg_data = _parse_llm_json(msg_response)
    
    commit_msg = msg_data.get("commit_message", "fix: autonomous repair by Zenith")
    title = msg_data.get("pr_title", f"Fix: Autonomous resolution for {state.get('analysis', {}).get('error_type', 'bug')}")
    body = msg_data.get("pr_body", f"## Zenith Autonomous Fix\nThis PR was generated by the Zenith AI agent.\n\n**Explanation:**\n{state.get('fix_explanation', '')}")
    
    # 2. Branch and Commit
    short_id = uuid.uuid4().hex[:6]
    branch_name = f"zenith/fix-{short_id}"
    logger.info(f"  -> Creating branch {branch_name}")
    gm.create_branch(branch_name, checkout=True)
    
    files_to_commit = list(files_to_commit)
    gm.stage_and_commit(files_to_commit, commit_msg)
    
    # 3. Push
    logger.info("  -> Pushing branch to origin...")
    success = gm.push("origin", branch_name)
    if not success:
        logger.error("Failed to push branch.")
        return {}
        
    # 4. Open PR
    # For a real PR, we need the repo full name (e.g. "user/repo").
    remote_url = gm.repo.remotes.origin.url
    repo_full_name = ""
    if "github.com" in remote_url:
        parts = remote_url.split("github.com")[-1].split(".git")[0]
        repo_full_name = parts.strip(":/")
    else:
        logger.warning(f"Could not parse repo name from remote url: {remote_url}")
        repo_full_name = "test/test" # fallback
        
    logger.info(f"  -> Opening PR on {repo_full_name}...")
    
    pr_url, pr_obj = gh.create_pull_request(repo_full_name, branch_name, "main", title, body)
    if pr_url:
        logger.success(f"  -> 🎉 PR raised successfully: {pr_url}")
        
        # Risk-Based Auto-Merge
        risk_level = state.get("risk_level", "HIGH")
        if risk_level == "LOW" and pr_obj:
            logger.info("  -> Risk level is LOW. Auto-merging PR...")
            try:
                pr_obj.merge(merge_method="squash")
                logger.success(f"  -> 🚀 PR auto-merged successfully!")
            except Exception as e:
                logger.error(f"  -> Failed to auto-merge PR: {e}")
        else:
            logger.info(f"  -> Risk level is {risk_level}. Waiting for human review.")
            
    return {}


def mark_success(state: AgentState) -> Dict[str, Any]:
    return {"status": "success"}

def mark_failed(state: AgentState) -> Dict[str, Any]:
    return {"status": "failed"}
