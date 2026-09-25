"""
Local embedding generation using sentence-transformers.
Supports BGE-M3 (production) and bge-small-en-v1.5 (dev/testing).
"""

from typing import List

from loguru import logger
from sentence_transformers import SentenceTransformer

from zenith.config import settings


class EmbeddingEngine:
    """
    Generates dense vector embeddings for text using sentence-transformers.
    Used to embed code chunks and search queries for semantic retrieval.
    """

    def __init__(self, model_name: str = None, device: str = None):
        """
        Initialize the embedding engine.

        Args:
            model_name: HuggingFace model identifier.
                        Defaults to settings.EMBEDDING_MODEL.
            device: Device to run on ('cpu', 'cuda', 'mps').
                    Defaults to settings.EMBEDDING_DEVICE.
        """
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.device = device or settings.EMBEDDING_DEVICE
        self._model = None
        self._dimension = None

    def _load_model(self):
        """Lazily load the model on first use."""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name} (device={self.device})")
            self._model = SentenceTransformer(self.model_name, device=self.device)
            # Determine dimension from a test embedding
            test_embedding = self._model.encode("test", show_progress_bar=False)
            self._dimension = len(test_embedding)
            logger.info(f"Embedding model loaded. Dimension: {self._dimension}")

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        self._load_model()
        return self._dimension

    def embed_text(self, text: str) -> List[float]:
        """
        Generate an embedding for a single text string.

        Args:
            text: The text to embed.

        Returns:
            A list of floats representing the dense embedding vector.
        """
        self._load_model()
        embedding = self._model.encode(text, show_progress_bar=False, normalize_embeddings=True)
        return embedding.tolist()

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of text strings to embed.
            batch_size: Number of texts to process per batch.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        self._load_model()
        logger.debug(f"Embedding batch of {len(texts)} texts (batch_size={batch_size})")
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embeddings.tolist()


class Reranker:
    """
    Cross-encoder reranker for precision re-scoring of search results.
    Unlike bi-encoders (used for initial retrieval), cross-encoders process
    query-document pairs jointly for higher accuracy, but are slower.

    Used as a final step: retrieve top-K with embeddings → rerank top-K with cross-encoder.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize the reranker.

        Args:
            model_name: HuggingFace cross-encoder model.
                        Default is a lightweight 22MB model suitable for CPU.
        """
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        """Lazily load the cross-encoder on first use."""
        if self._model is None:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading reranker model: {self.model_name}")
            self._model = CrossEncoder(self.model_name)
            logger.info("Reranker model loaded.")

    def rerank(
        self,
        query: str,
        results: List[dict],
        top_k: int = 5,
        content_key: str = "content",
    ) -> List[dict]:
        """
        Rerank search results using the cross-encoder.

        Args:
            query: The original search query.
            results: List of search result dicts (must have 'metadata' with content).
            top_k: Number of results to return after reranking.
            content_key: Key in metadata containing the text to rerank against.

        Returns:
            Reranked list of results with updated scores.
        """
        if not results:
            return []

        self._load_model()

        # Build query-document pairs
        pairs = []
        for result in results:
            doc_text = result.get("metadata", {}).get(content_key, "")
            # Also include symbol name for better matching
            symbol = result.get("metadata", {}).get("symbol_name", "")
            combined = f"{symbol}: {doc_text}" if symbol else doc_text
            pairs.append((query, combined))

        # Score all pairs
        scores = self._model.predict(pairs)

        # Attach rerank scores
        for result, rerank_score in zip(results, scores):
            result["original_score"] = result["score"]
            result["rerank_score"] = float(rerank_score)
            result["score"] = float(rerank_score)

        # Sort by rerank score descending
        results.sort(key=lambda x: x["score"], reverse=True)

        logger.debug(f"Reranked {len(results)} results, returning top {top_k}")
        return results[:top_k]
