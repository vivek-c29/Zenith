"""
Maps file extensions to tree-sitter language identifiers.
"""

import os
from typing import Optional
from loguru import logger

# Comprehensive mapping of file extensions to tree-sitter language names
EXTENSION_TO_LANGUAGE: dict[str, str] = {
    # Python
    ".py": "python",
    ".pyi": "python",
    # JavaScript / TypeScript
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".mjs": "javascript",
    ".cjs": "javascript",
    # Web
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".scss": "scss",
    # JVM
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".scala": "scala",
    ".groovy": "groovy",
    # Systems
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".rs": "rust",
    ".go": "go",
    ".zig": "zig",
    # Scripting
    ".rb": "ruby",
    ".php": "php",
    ".lua": "lua",
    ".pl": "perl",
    ".pm": "perl",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".fish": "fish",
    # Data & Config
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".xml": "xml",
    ".sql": "sql",
    # Functional
    ".ex": "elixir",
    ".exs": "elixir",
    ".erl": "erlang",
    ".hs": "haskell",
    ".ml": "ocaml",
    ".clj": "clojure",
    # Mobile
    ".swift": "swift",
    ".dart": "dart",
    # Other
    ".r": "r",
    ".R": "r",
    ".cs": "csharp",
    ".fs": "fsharp",
    ".vim": "vim",
    ".dockerfile": "dockerfile",
    ".proto": "proto",
    ".graphql": "graphql",
    ".gql": "graphql",
    ".md": "markdown",
    ".rst": "rst",
    ".tex": "latex",
    ".makefile": "make",
}

# Also match some filenames without extensions
FILENAME_TO_LANGUAGE: dict[str, str] = {
    "Dockerfile": "dockerfile",
    "Makefile": "make",
    "CMakeLists.txt": "cmake",
    "Gemfile": "ruby",
    "Rakefile": "ruby",
    "Jenkinsfile": "groovy",
    "BUILD": "starlark",
    "WORKSPACE": "starlark",
}


def detect_language(file_path: str) -> Optional[str]:
    """
    Detects the programming language of a file based on its extension or filename.
    
    Returns the tree-sitter language name, or None if the language is not supported.
    """
    basename = os.path.basename(file_path)

    # Check exact filename matches first (Dockerfile, Makefile, etc.)
    if basename in FILENAME_TO_LANGUAGE:
        return FILENAME_TO_LANGUAGE[basename]

    # Check file extension
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext in EXTENSION_TO_LANGUAGE:
        return EXTENSION_TO_LANGUAGE[ext]

    logger.debug(f"Could not detect language for: {file_path}")
    return None
