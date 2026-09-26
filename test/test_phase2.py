"""
Phase 2 Verification: Code Parser & Git Manager
"""
import os
import tempfile
import shutil

from zenith.config import setup_logger
from zenith.parser import CodeParser, detect_language
from zenith.git_manager import GitManager


def test_language_detection():
    """Test file extension to language mapping."""
    logger.info("--- Test: Language Detection ---")
    assert detect_language("app.py") == "python"
    assert detect_language("index.js") == "javascript"
    assert detect_language("main.go") == "go"
    assert detect_language("server.ts") == "typescript"
    assert detect_language("Dockerfile") == "dockerfile"
    assert detect_language("Makefile") == "make"
    assert detect_language("data.csv") is None  # unsupported
    logger.success("Language detection: All assertions passed!")


def test_code_parser():
    """Test tree-sitter parsing and symbol extraction."""
    logger.info("--- Test: Code Parser ---")

    parser = CodeParser()

    sample_python = '''
import os
from typing import List, Optional

class AuthService:
    """Handles user authentication."""

    def __init__(self, db):
        self.db = db

    def validate_token(self, token: str) -> bool:
        """Validates a JWT token."""
        if not token:
            return False
        return self.db.check_token(token)

    def refresh_token(self, user_id: int) -> str:
        """Issues a new token for a user."""
        return self.db.create_token(user_id)

def helper_function(x: int) -> int:
    """A standalone helper."""
    return x * 2
'''

    result = parser.parse_source(sample_python, "python", file_path="auth_service.py")

    # Check we found the right symbols
    symbol_names = [s.name for s in result.symbols]
    logger.info(f"Extracted symbols: {symbol_names}")

    assert "AuthService" in symbol_names, "Should find AuthService class"
    assert "validate_token" in symbol_names, "Should find validate_token method"
    assert "refresh_token" in symbol_names, "Should find refresh_token method"
    assert "helper_function" in symbol_names, "Should find helper_function"

    # Check symbol kinds
    auth_class = [s for s in result.symbols if s.name == "AuthService"][0]
    assert auth_class.kind.value == "class", f"AuthService should be a class, got {auth_class.kind}"

    validate_method = [s for s in result.symbols if s.name == "validate_token"][0]
    assert validate_method.kind.value == "method", f"validate_token should be a method, got {validate_method.kind}"
    assert validate_method.parent_class == "AuthService", "validate_token should belong to AuthService"

    helper = [s for s in result.symbols if s.name == "helper_function"][0]
    assert helper.kind.value == "function", f"helper_function should be a function, got {helper.kind}"

    # Check docstring extraction
    assert auth_class.docstring is not None, "AuthService should have a docstring"
    logger.info(f"AuthService docstring: {auth_class.docstring}")

    # Check imports
    assert len(result.imports) == 2, f"Should have 2 imports, got {len(result.imports)}"
    logger.info(f"Imports: {result.imports}")

    # Check chunks
    assert len(result.chunks) > 0, "Should have at least one chunk"
    logger.info(f"Generated {len(result.chunks)} chunks:")
    for chunk in result.chunks:
        logger.info(f"  [{chunk.symbol_kind}] {chunk.symbol_name} "
                     f"(lines {chunk.start_line}-{chunk.end_line}, {chunk.token_count} tokens)")

    logger.success("Code Parser: All assertions passed!")


def test_git_manager():
    """Test GitManager with a temporary repository."""
    logger.info("--- Test: Git Manager ---")

    # Create a temporary directory for our test repo
    tmp_dir = tempfile.mkdtemp(prefix="zenith_test_")

    try:
        # Initialize a new git repo
        from git import Repo
        repo = Repo.init(tmp_dir)

        # Configure git user for commits
        repo.config_writer().set_value("user", "name", "Zenith Test").release()
        repo.config_writer().set_value("user", "email", "test@zenith.dev").release()

        # Create some test files
        test_file = os.path.join(tmp_dir, "app.py")
        with open(test_file, "w") as f:
            f.write('def main():\n    print("Hello from Zenith")\n')

        readme = os.path.join(tmp_dir, "README.md")
        with open(readme, "w") as f:
            f.write("# Test Repository\n")

        # Stage and commit
        repo.index.add(["app.py", "README.md"])
        repo.index.commit("Initial commit")

        # Now test our GitManager
        gm = GitManager(tmp_dir)

        # Test get_repo_info
        info = gm.get_repo_info()
        logger.info(f"Repo info: branch={info.active_branch}, dirty={info.is_dirty}")
        assert info.active_branch in ("main", "master"), f"Branch should be main/master, got {info.active_branch}"
        assert not info.is_dirty, "Repo should be clean after commit"
        assert len(info.recent_commits) > 0, "Should have at least 1 commit"
        logger.info(f"Recent commits: {info.recent_commits}")

        # Test list_files
        files = gm.list_files()
        logger.info(f"Tracked files: {files}")
        assert "app.py" in files, "Should list app.py"
        assert "README.md" in files, "Should list README.md"

        # Test list_files with extension filter
        py_files = gm.list_files(extensions_filter={".py"})
        assert len(py_files) == 1 and py_files[0] == "app.py"

        # Test get_file_content
        content = gm.get_file_content("app.py")
        assert "Hello from Zenith" in content, "Should read app.py content"

        # Test create_branch
        gm.create_branch("fix/test-bug-42")
        info2 = gm.get_repo_info()
        assert info2.active_branch == "fix/test-bug-42", f"Should be on fix branch, got {info2.active_branch}"

        # Make a change and commit on the new branch
        with open(test_file, "w") as f:
            f.write('def main():\n    print("Bug #42 fixed!")\n')

        commit_sha = gm.stage_and_commit(["app.py"], "fix: resolve null email in auth (#42)")
        logger.info(f"New commit: {commit_sha[:8]}")
        assert len(commit_sha) == 40, "Commit SHA should be 40 chars"

        # Test get_commit_history
        history = gm.get_commit_history(max_count=5)
        logger.info(f"Commit history ({len(history)} commits):")
        for c in history:
            logger.info(f"  {c['short_sha']} - {c['message']}")
        assert len(history) == 2, f"Should have 2 commits, got {len(history)}"

        logger.success("Git Manager: All assertions passed!")

    finally:
        # Cleanup
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    logger = setup_logger()
    logger.info("=" * 60)
    logger.info("Phase 2 Verification: Code Parser & Git Manager")
    logger.info("=" * 60)

    test_language_detection()
    test_code_parser()
    test_git_manager()

    logger.success("=" * 60)
    logger.success("ALL PHASE 2 TESTS PASSED!")
    logger.success("=" * 60)
