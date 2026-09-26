"""
Phase 3 Verification: Embeddings & Vector Store
Tests embedding generation, local vector store, code indexing, and semantic search.
Now includes Hybrid Search, MMR, and Reranking tests.
"""

import os
import shutil
import tempfile

from zenith.config import setup_logger
from zenith.vector_store import EmbeddingEngine, LocalVectorStore, CodeIndexer
from zenith.vector_store.embedding_engine import Reranker


def test_embedding_engine():
    """Test embedding model loading and vector generation."""
    logger.info("--- Test: Embedding Engine ---")

    engine = EmbeddingEngine()

    # Test single text embedding
    text = "def validate_token(self, token: str) -> bool:"
    embedding = engine.embed_text(text)

    logger.info(f"Embedding dimension: {len(embedding)}")
    assert len(embedding) == engine.dimension, f"Expected dim {engine.dimension}, got {len(embedding)}"

    # Test batch embedding
    texts = [
        "class AuthService handles user login",
        "function to calculate tax on purchases",
    ]
    batch_embeddings = engine.embed_batch(texts)
    assert len(batch_embeddings) == 2, "Should return 2 embeddings"

    logger.success("Embedding Engine: All assertions passed!")
    return engine


def test_reranker():
    """Test CrossEncoder reranker."""
    logger.info("--- Test: Reranker ---")
    reranker = Reranker()
    
    query = "How to authenticate users"
    results = [
        {"score": 0.8, "metadata": {"content": "Random unrelated text about database connections."}},
        {"score": 0.5, "metadata": {"content": "This module handles user authentication and JWT tokens."}}
    ]
    
    reranked = reranker.rerank(query, results, top_k=2)
    # The second document should now have a higher score and be first
    assert "authentication" in reranked[0]["metadata"]["content"], "Reranker failed to prioritize the correct result"
    logger.success("Reranker: All assertions passed!")
    return reranker


def test_code_indexer(engine: EmbeddingEngine, reranker: Reranker):
    """Test the full indexing pipeline on a temporary repo with hybrid/MMR/reranking."""
    logger.info("--- Test: Code Indexer ---")

    tmp_dir = tempfile.mkdtemp(prefix="zenith_index_test_")

    try:
        auth_code = '''
import jwt
class AuthService:
    def validate_token(self, token: str) -> bool:
        return True
'''
        with open(os.path.join(tmp_dir, "auth_service.py"), "w") as f:
            f.write(auth_code)

        db_code = '''
class DatabasePool:
    def get_connection(self):
        return True
'''
        with open(os.path.join(tmp_dir, "database.py"), "w") as f:
            f.write(db_code)

        store = LocalVectorStore()
        indexer = CodeIndexer(
            embedding_engine=engine,
            vector_store=store,
            reranker=reranker
        )

        stats = indexer.index_repository(tmp_dir, extensions={".py"})
        assert stats["files_parsed"] == 2
        assert stats["chunks_indexed"] > 0

        # Test standard search
        res_std = indexer.search_code("token authentication", strategy="standard")
        assert len(res_std) > 0

        # Test hybrid search
        res_hybrid = indexer.search_code("token authentication", strategy="hybrid")
        assert len(res_hybrid) > 0

        # Test MMR search
        res_mmr = indexer.search_code("connection", strategy="mmr")
        assert len(res_mmr) > 0

        # Test with reranker
        res_rerank = indexer.search_code("token", strategy="standard", use_reranker=True)
        assert len(res_rerank) > 0

        logger.success("Code Indexer: All assertions passed!")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_fix_history_indexer(engine: EmbeddingEngine):
    """Test indexing git commit history with actual diffs."""
    logger.info("--- Test: Fix History Indexer ---")

    tmp_dir = tempfile.mkdtemp(prefix="zenith_fix_test_")

    try:
        from git import Repo
        repo = Repo.init(tmp_dir)
        repo.config_writer().set_value("user", "name", "Zenith Test").release()
        repo.config_writer().set_value("user", "email", "test@zenith.dev").release()

        app_file = os.path.join(tmp_dir, "app.py")
        with open(app_file, "w") as f:
            f.write("def login(email):\n    return email.lower()\n")
        repo.index.add(["app.py"])
        repo.index.commit("feat: add login function")

        with open(app_file, "w") as f:
            f.write("def login(email):\n    if not email:\n        return None\n    return email.lower()\n")
        repo.index.add(["app.py"])
        repo.index.commit("fix: handle null email")

        from zenith.git_manager import GitManager
        gm = GitManager(tmp_dir)

        store = LocalVectorStore()
        indexer = CodeIndexer(embedding_engine=engine, vector_store=store)

        stats = indexer.index_fix_history(gm, max_commits=10)
        assert stats["commits_indexed"] == 2

        # Check if diff is stored in metadata
        res = indexer.search_similar_fixes("handle null email", strategy="hybrid")
        assert len(res) > 0
        assert "diff" in res[0]["metadata"]
        assert "return None" in res[0]["metadata"]["diff"]

        logger.success("Fix History Indexer: All assertions passed!")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    logger = setup_logger()
    logger.info("=" * 60)
    logger.info("Phase 3 Verification: Advanced RAG")
    logger.info("=" * 60)

    engine = test_embedding_engine()
    reranker = test_reranker()
    test_code_indexer(engine, reranker)
    test_fix_history_indexer(engine)

    logger.success("=" * 60)
    logger.success("ALL ADVANCED RAG TESTS PASSED!")
    logger.success("=" * 60)
