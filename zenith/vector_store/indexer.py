"""
Code indexing pipeline: parses code, generates embeddings, and stores them
in the vector store for semantic retrieval by the bug-fixing agent.

Enhanced with:
- Hybrid search (dense + keyword)
- MMR (Maximal Marginal Relevance) for diverse results
- Cross-encoder reranking for precision
- Actual commit diffs stored in fix history
"""

import os
from typing import Any, Dict, List, Optional, Set

from loguru import logger

from zenith.parser import CodeParser
from zenith.parser.models import CodeChunk
from zenith.git_manager import GitManager
from .embedding_engine import EmbeddingEngine, Reranker
from .vector_store import BaseVectorStore


# Namespaces to separate different types of indexed content
NS_CODE = "code"
NS_FIX_HISTORY = "fix_history"


class CodeIndexer:
    """
    Orchestrates the full indexing pipeline:
    1. Parse code files into structural chunks.
    2. Generate embeddings for each chunk.
    3. Store embeddings + metadata in the vector store.

    Also indexes historical bug fixes from git commit diffs.

    Search methods support three retrieval strategies:
    - Standard: Pure dense (cosine) similarity
    - Hybrid: Dense + keyword scoring
    - MMR: Maximal Marginal Relevance for diversity

    All strategies optionally support cross-encoder reranking.
    """

    def __init__(
        self,
        embedding_engine: EmbeddingEngine,
        vector_store: BaseVectorStore,
        parser: Optional[CodeParser] = None,
        reranker: Optional[Reranker] = None,
    ):
        """
        Initialize the indexer.

        Args:
            embedding_engine: Engine for generating embeddings.
            vector_store: Store for persisting and searching vectors.
            parser: Code parser instance. Created automatically if not provided.
            reranker: Optional cross-encoder reranker for precision search.
        """
        self.embedding = embedding_engine
        self.store = vector_store
        self.parser = parser or CodeParser()
        self.reranker = reranker

    def index_repository(
        self,
        repo_path: str,
        extensions: Optional[Set[str]] = None,
        max_files: int = 500,
    ) -> Dict[str, int]:
        """
        Index all supported source files in a repository.

        Args:
            repo_path: Absolute path to the repository root.
            extensions: File extensions to index (e.g., {'.py', '.js'}).
                        Defaults to common code extensions.
            max_files: Maximum number of files to index.

        Returns:
            Dict with indexing statistics.
        """
        if extensions is None:
            extensions = {
                ".py",
                ".js",
                ".ts",
                ".jsx",
                ".tsx",
                ".java",
                ".go",
                ".rs",
                ".cpp",
                ".c",
                ".rb",
            }

        logger.info(
            f"Indexing repository: {repo_path} "
            f"(extensions={extensions})"
        )

        # Collect files
        files_to_index = []

        for root, dirs, files in os.walk(repo_path):
            # Skip hidden dirs, node_modules, venvs, etc.
            dirs[:] = [
                d
                for d in dirs
                if not d.startswith(".")
                and d not in (
                    "node_modules",
                    "__pycache__",
                    ".venv",
                    "venv",
                    "dist",
                    "build",
                )
            ]

            for file in files:
                _, ext = os.path.splitext(file)

                if ext.lower() in extensions:
                    files_to_index.append(os.path.join(root, file))

                if len(files_to_index) >= max_files:
                    break

            if len(files_to_index) >= max_files:
                break

        logger.info(f"Found {len(files_to_index)} files to index")

        # Parse and chunk
        all_chunks: List[CodeChunk] = []
        parsed_files = []
        files_parsed = 0

        for file_path in files_to_index:
            parsed = self.parser.parse_file(file_path)

            if parsed and parsed.chunks:
                # Use relative paths in metadata for portability
                rel_path = os.path.relpath(file_path, repo_path)
                parsed.file_path = rel_path

                for chunk in parsed.chunks:
                    chunk.file_path = rel_path

                # Keep the complete parsed file for Neo4j graph indexing
                parsed_files.append(parsed)

                all_chunks.extend(parsed.chunks)
                files_parsed += 1

        logger.info(
            f"Parsed {files_parsed} files -> "
            f"{len(all_chunks)} chunks"
        )

        if not all_chunks:
            logger.warning("No chunks to index")
            return {
                "files_parsed": 0,
                "chunks_indexed": 0,
                "total_files": len(files_to_index),
            }

        # Generate embeddings
        texts = [
            self._chunk_to_text(chunk)
            for chunk in all_chunks
        ]

        embeddings = self.embedding.embed_batch(texts)

        # Prepare vectors for non-graph vector stores
        vectors = []

        for chunk, embedding in zip(all_chunks, embeddings):
            vectors.append(
                {
                    "id": chunk.chunk_id,
                    "values": embedding,
                    "metadata": {
                        "file_path": chunk.file_path,
                        "language": chunk.language,
                        "symbol_name": chunk.symbol_name or "",
                        "symbol_kind": (
                            chunk.symbol_kind.value
                            if chunk.symbol_kind
                            else "module"
                        ),
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                        "token_count": chunk.token_count,
                        "content": chunk.content[:1000],
                        "type": "code",
                    },
                }
            )

        # Upsert to vector store
        #
        # Neo4j:
        #   Use graph-aware upsert_file_graph() so the index
        #   contains File -> CONTAINS -> CodeChunk relationships.
        #
        # Pinecone / Local:
        #   Keep the existing generic vector upsert() behavior.
        backend = self.store.get_stats().get("backend")

        if backend == "neo4j":
            upserted = 0
            offset = 0

            for parsed_file in parsed_files:
                file_chunks = parsed_file.chunks

                file_embeddings = embeddings[
                    offset : offset + len(file_chunks)
                ]

                self.store.upsert_file_graph(
                    parsed_file,
                    file_chunks,
                    file_embeddings,
                )

                upserted += len(file_chunks)
                offset += len(file_chunks)
            self.store.rebuild_all_call_relationships()

        else:
            upserted = self.store.upsert(
                vectors,
                namespace=NS_CODE,
            )

        stats = {
            "files_parsed": files_parsed,
            "chunks_indexed": upserted,
            "total_files": len(files_to_index),
        }

        logger.info(f"Indexing complete: {stats}")
        return stats

    def sync_repository(
        self,
        repo_path: str,
        base_branch: str = "main",
        extensions: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        """
        Incrementally sync repository to vector store.
        Deletes vectors for modified/deleted files, then parses and upserts modified/added files.
        """
        import subprocess

        if extensions is None:
            extensions = {
                ".py",
                ".js",
                ".ts",
                ".jsx",
                ".tsx",
                ".java",
                ".go",
                ".rs",
                ".cpp",
                ".c",
                ".rb",
            }

        logger.info(
            f"Syncing repository : "
            f"{repo_path} against {base_branch}"
        )

        try:
            # Get files changed since the target branch.
            result = subprocess.run(
                f"git diff --name-status {base_branch}...HEAD",
                cwd=repo_path,
                shell=True,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                logger.warning(
                    "Failed to get git diff for sync. "
                    f"Falling back to full index. {result.stderr}"
                )
                return self.index_repository(
                    repo_path,
                    extensions=extensions,
                )

            diff_text = result.stdout.strip()

            # CI/CD may run Zenith on the target branch itself, where
            # base_branch == HEAD. In that case branch diff is empty even
            # though the latest commit introduced changes.
            if not diff_text:
                try:
                    base_sha = subprocess.run(
                        ["git", "rev-parse", base_branch],
                        cwd=repo_path,
                        capture_output=True,
                        text=True,
                        check=True,
                    ).stdout.strip()

                    head_sha = subprocess.run(
                        ["git", "rev-parse", "HEAD"],
                        cwd=repo_path,
                        capture_output=True,
                        text=True,
                        check=True,
                    ).stdout.strip()

                    if base_sha == head_sha:
                        fallback = subprocess.run(
                            ["git", "diff", "--name-status", "HEAD~1"],
                            cwd=repo_path,
                            capture_output=True,
                            text=True,
                        )

                        if fallback.returncode == 0:
                            diff_text = fallback.stdout.strip()
                            logger.info(
                                "Base branch points to HEAD; using HEAD~1 "
                                "to detect the latest commit changes."
                            )

                except Exception as e:
                    logger.warning(f"Failed latest-commit diff fallback: {e}")

            diff_output = diff_text.splitlines() if diff_text else []
        except Exception as e:
            logger.warning(f"Error getting git diff for sync: {e}")

            return self.index_repository(
                repo_path,
                extensions=extensions,
            )

        if not diff_output or diff_output == [""]:
            logger.info("No changed files found for sync.")

            return {
                "files_parsed": 0,
                "chunks_indexed": 0,
                "total_files": 0,
            }

        files_to_delete = []
        files_to_upsert = []

        for line in diff_output:
            parts = line.split("\t")

            if len(parts) >= 2:
                status, file_rel_path = parts[0][0], parts[-1]

                _, ext = os.path.splitext(file_rel_path)

                if ext.lower() in extensions:
                    if status in ("M", "D", "R", "C"):
                        files_to_delete.append(file_rel_path)

                    if status in ("M", "A", "R", "C"):
                        files_to_upsert.append(file_rel_path)

        logger.info(
            f"Sync plan: {len(files_to_delete)} files to delete, "
            f"{len(files_to_upsert)} files to upsert."
        )

        # 1. Prune Phase: Delete old chunks
        deleted_count = 0

        for rel_path in files_to_delete:
            self.store.delete(
                filter={"file_path": rel_path},
                namespace=NS_CODE,
            )
            deleted_count += 1

        # 2. Patch Phase: Parse and Upsert new chunks
        all_chunks = []
        files_parsed = 0

        is_neo4j = self.store.get_stats().get("backend") == "neo4j"

        upserted = 0

        for rel_path in files_to_upsert:
            abs_path = os.path.join(repo_path, rel_path)

            if os.path.exists(abs_path):
                parsed = self.parser.parse_file(abs_path)

                if parsed and parsed.chunks:
                    parsed.file_path = rel_path

                    for chunk in parsed.chunks:
                        chunk.file_path = rel_path

                    if is_neo4j:
                        texts = [
                            self._chunk_to_text(chunk)
                            for chunk in parsed.chunks
                        ]

                        embeddings = self.embedding.embed_batch(texts)

                        # We know self.store is Neo4jGraphStore
                        self.store.upsert_file_graph(
                            parsed,
                            parsed.chunks,
                            embeddings,
                        )

                        upserted += len(parsed.chunks)

                    else:
                        all_chunks.extend(parsed.chunks)

                    files_parsed += 1

        if not is_neo4j and all_chunks:
            texts = [
                self._chunk_to_text(chunk)
                for chunk in all_chunks
            ]

            embeddings = self.embedding.embed_batch(texts)

            vectors = []

            for chunk, embedding in zip(all_chunks, embeddings):
                vectors.append(
                    {
                        "id": chunk.chunk_id,
                        "values": embedding,
                        "metadata": {
                            "file_path": chunk.file_path,
                            "language": chunk.language,
                            "symbol_name": chunk.symbol_name or "",
                            "type": "code",
                            "content": chunk.content[:1000],
                        },
                    }
                )

            upserted = self.store.upsert(
                vectors,
                namespace=NS_CODE,
            )

        stats = {
            "files_parsed": files_parsed,
            "chunks_indexed": upserted,
            "files_pruned": deleted_count,
            "total_files": len(files_to_upsert),
        }

        logger.info(f"Sync complete: {stats}")

        return stats

    def index_fix_history(
        self,
        git_manager: GitManager,
        max_commits: int = 50,
    ) -> Dict[str, int]:
        """
        Index historical bug fix commits as 'fix memory' for the agent.
        Stores commit messages AND actual diffs so the agent can see
        HOW past bugs were fixed, not just that they were fixed.

        Args:
            git_manager: GitManager instance for the repository.
            max_commits: Maximum number of commits to index.

        Returns:
            Dict with indexing statistics.
        """
        logger.info(
            f"Indexing fix history (max_commits={max_commits})"
        )

        commits = git_manager.get_commit_history(
            max_count=max_commits
        )

        if not commits:
            logger.warning("No commit history found")

            return {
                "commits_indexed": 0
            }

        # Prepare text representations of each commit with actual diffs
        texts = []
        commit_data = []

        for commit_info in commits:
            # Get the actual diff for this commit
            diff_text = git_manager.get_commit_diff(
                commit_info["sha"],
                max_diff_length=2000,
            )

            # Create a rich text representation for embedding
            text = (
                f"Commit: {commit_info['message']}\n"
                f"Author: {commit_info['author']}\n"
                f"Files changed: {commit_info['files_changed']}\n"
                f"Date: {commit_info['date']}"
            )

            if diff_text:
                text += f"\n\nDiff:\n{diff_text}"

            texts.append(text)

            commit_data.append(
                {
                    **commit_info,
                    "diff": diff_text or "",
                }
            )

        # Generate embeddings
        embeddings = self.embedding.embed_batch(texts)

        # Prepare vectors
        vectors = []

        for commit_info, embedding in zip(commit_data, embeddings):
            vectors.append(
                {
                    "id": f"fix_{commit_info['sha'][:16]}",
                    "values": embedding,
                    "metadata": {
                        "sha": commit_info["sha"],
                        "short_sha": commit_info["short_sha"],
                        "message": commit_info["message"][:500],
                        "author": commit_info["author"],
                        "date": commit_info["date"],
                        "files_changed": commit_info["files_changed"],
                        "diff": commit_info["diff"][:1500],
                        "content": commit_info["message"],
                        "type": "fix_history",
                    },
                }
            )

        # Upsert to vector store
        upserted = self.store.upsert(
            vectors,
            namespace=NS_FIX_HISTORY,
        )

        stats = {
            "commits_indexed": upserted
        }

        logger.info(
            f"Fix history indexing complete: {stats}"
        )

        return stats

    def search_code(
        self,
        query: str,
        top_k: int = 5,
        language_filter: Optional[str] = None,
        strategy: str = "hybrid",
        use_reranker: bool = False,
    ) -> List[Dict]:
        """
        Search the codebase using a natural language query.

        Args:
            query: Natural language search query
                   (e.g., "authentication token validation").
            top_k: Number of results to return.
            language_filter: Optional language to filter results.
            strategy: Search strategy - "standard", "hybrid", or "mmr".
            use_reranker: Whether to apply cross-encoder reranking.

        Returns:
            List of search results with score and metadata.
        """
        query_embedding = self.embedding.embed_text(query)

        filter_dict = None

        if language_filter:
            filter_dict = {
                "language": language_filter
            }

        # Execute search based on strategy
        if strategy == "hybrid":
            results = self.store.hybrid_search(
                query_embedding=query_embedding,
                query_text=query,
                top_k=(
                    top_k
                    if not use_reranker
                    else top_k * 2
                ),
                alpha=0.7,
                filter=filter_dict,
                namespace=NS_CODE,
            )

        elif strategy == "mmr":
            results = self.store.mmr_search(
                query_embedding=query_embedding,
                top_k=(
                    top_k
                    if not use_reranker
                    else top_k * 2
                ),
                fetch_k=top_k * 4,
                lambda_mult=0.7,
                filter=filter_dict,
                namespace=NS_CODE,
            )

        else:
            results = self.store.search(
                query_embedding=query_embedding,
                top_k=(
                    top_k
                    if not use_reranker
                    else top_k * 2
                ),
                filter=filter_dict,
                namespace=NS_CODE,
            )

        # Apply reranking if requested and reranker is available
        if use_reranker and self.reranker and results:
            results = self.reranker.rerank(
                query,
                results,
                top_k=top_k,
            )

        logger.info(
            f"Code search ({strategy}) for "
            f"'{query[:50]}...' returned "
            f"{len(results)} results"
        )

        return results[:top_k]

    def search_similar_fixes(
        self,
        bug_description: str,
        top_k: int = 5,
        strategy: str = "hybrid",
        use_reranker: bool = False,
    ) -> List[Dict]:
        """
        Find historical bug fixes similar to the given bug description.

        Args:
            bug_description: Description of the current bug.
            top_k: Number of similar fixes to return.
            strategy: Search strategy - "standard", "hybrid", or "mmr".
            use_reranker: Whether to apply cross-encoder reranking.

        Returns:
            List of similar fix results with score, metadata, and diff.
        """
        query_embedding = self.embedding.embed_text(
            bug_description
        )

        if strategy == "hybrid":
            results = self.store.hybrid_search(
                query_embedding=query_embedding,
                query_text=bug_description,
                top_k=(
                    top_k
                    if not use_reranker
                    else top_k * 2
                ),
                alpha=0.7,
                namespace=NS_FIX_HISTORY,
            )

        elif strategy == "mmr":
            results = self.store.mmr_search(
                query_embedding=query_embedding,
                top_k=(
                    top_k
                    if not use_reranker
                    else top_k * 2
                ),
                fetch_k=top_k * 4,
                lambda_mult=0.7,
                namespace=NS_FIX_HISTORY,
            )

        else:
            results = self.store.search(
                query_embedding=query_embedding,
                top_k=(
                    top_k
                    if not use_reranker
                    else top_k * 2
                ),
                namespace=NS_FIX_HISTORY,
            )

        if use_reranker and self.reranker and results:
            # For fix history, rerank against commit message + diff
            results = self.reranker.rerank(
                bug_description,
                results,
                top_k=top_k,
                content_key="message",
            )

        logger.info(
            f"Fix search ({strategy}) for "
            f"'{bug_description[:50]}...' returned "
            f"{len(results)} results"
        )

        return results[:top_k]

    @staticmethod
    def _chunk_to_text(chunk: CodeChunk) -> str:
        """
        Convert a code chunk to a text representation suitable for embedding.
        Prepends file path and symbol info for better retrieval.
        """
        parts = [
            f"File: {chunk.file_path}"
        ]

        if chunk.symbol_name:
            parts.append(
                f"Symbol: {chunk.symbol_name}"
            )

        if chunk.symbol_kind:
            parts.append(
                f"Kind: {chunk.symbol_kind.value}"
            )

        parts.append(
            f"\n{chunk.content}"
        )

        return "\n".join(parts)