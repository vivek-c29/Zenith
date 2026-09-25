"""
Vector store abstraction with Pinecone (production) and local in-memory (dev) backends.
Supports hybrid search (dense + keyword), MMR (Maximal Marginal Relevance) for diversity.
"""

import re
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from loguru import logger

from zenith.config import settings


class BaseVectorStore(ABC):
    """Abstract base for vector store backends."""

    @abstractmethod
    def upsert(self, vectors: List[Dict[str, Any]], namespace: str = "") -> int:
        """
        Upsert vectors into the store.

        Args:
            vectors: List of dicts with keys: 'id', 'values' (embedding), 'metadata'.

        Returns:
            Number of vectors upserted.
        """
        ...

    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter: Optional[Dict] = None,
        namespace: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.

        Args:
            query_embedding: The query vector.
            top_k: Number of results to return.
            filter: Optional metadata filter.
            namespace: Optional namespace to search within.

        Returns:
            List of results with keys: 'id', 'score', 'metadata'.
        """
        ...

    @abstractmethod
    def delete(self, ids: List[str], namespace: str = "") -> None:
        """Delete vectors by their IDs."""
        ...

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Return store statistics."""
        ...

    def hybrid_search(
        self,
        query_embedding: List[float],
        query_text: str,
        top_k: int = 5,
        alpha: float = 0.7,
        filter: Optional[Dict] = None,
        namespace: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining dense (semantic) and sparse (keyword) scores.

        Args:
            query_embedding: Dense query vector.
            query_text: Original query text for keyword matching.
            top_k: Number of results to return.
            alpha: Weight for dense score (0.0 = pure keyword, 1.0 = pure semantic).
            filter: Optional metadata filter.
            namespace: Namespace to search.

        Returns:
            List of results sorted by hybrid score.
        """
        # Get more candidates than needed for re-scoring
        candidates = self.search(
            query_embedding=query_embedding,
            top_k=top_k * 3,
            filter=filter,
            namespace=namespace,
        )

        if not candidates:
            return []

        # Extract keywords from query
        query_keywords = _extract_keywords(query_text)

        # Re-score with keyword boost
        for result in candidates:
            dense_score = result["score"]
            content = result.get("metadata", {}).get("content", "")
            symbol = result.get("metadata", {}).get("symbol_name", "")
            keyword_score = _keyword_score(query_keywords, content + " " + symbol)

            result["dense_score"] = dense_score
            result["keyword_score"] = keyword_score
            result["score"] = alpha * dense_score + (1 - alpha) * keyword_score

        # Re-sort by hybrid score
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_k]

    def mmr_search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        fetch_k: int = 20,
        lambda_mult: float = 0.7,
        filter: Optional[Dict] = None,
        namespace: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Maximal Marginal Relevance search for diverse results.
        Balances relevance to query with diversity among selected results.

        MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, already_selected))

        Args:
            query_embedding: The query vector.
            top_k: Number of diverse results to return.
            fetch_k: Number of initial candidates to fetch (should be > top_k).
            lambda_mult: Diversity tradeoff (1.0 = pure relevance, 0.0 = pure diversity).
            filter: Optional metadata filter.
            namespace: Namespace to search.

        Returns:
            List of diverse, relevant results.
        """
        # Fetch initial candidates
        candidates = self.search(
            query_embedding=query_embedding,
            top_k=fetch_k,
            filter=filter,
            namespace=namespace,
        )

        if not candidates or len(candidates) <= top_k:
            return candidates[:top_k]

        query_vec = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)
        if query_norm > 0:
            query_vec = query_vec / query_norm

        # Build embedding matrix for candidates (we need their vectors)
        # Since we only have scores from initial search, we use scores as proxy
        # for query-doc similarity and compute doc-doc similarity from metadata content
        selected: List[Dict] = []
        remaining = list(candidates)

        for _ in range(min(top_k, len(candidates))):
            if not remaining:
                break

            best_idx = -1
            best_mmr = -float("inf")

            for i, candidate in enumerate(remaining):
                # Relevance to query (from original search score)
                relevance = candidate["score"]

                # Max similarity to already selected (using score as proxy)
                max_sim_to_selected = 0.0
                if selected:
                    candidate_content = candidate.get("metadata", {}).get("content", "")
                    for sel in selected:
                        sel_content = sel.get("metadata", {}).get("content", "")
                        # Content overlap as diversity proxy
                        sim = _content_similarity(candidate_content, sel_content)
                        max_sim_to_selected = max(max_sim_to_selected, sim)

                mmr_score = lambda_mult * relevance - (1 - lambda_mult) * max_sim_to_selected

                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = i

            if best_idx >= 0:
                selected_item = remaining.pop(best_idx)
                selected_item["mmr_score"] = best_mmr
                selected.append(selected_item)

        return selected


class LocalVectorStore(BaseVectorStore):
    """
    In-memory vector store using numpy cosine similarity.
    Used for development and testing without a Pinecone API key.
    """

    def __init__(self):
        # namespace -> { id: { 'values': list, 'metadata': dict } }
        self._store: Dict[str, Dict[str, Dict[str, Any]]] = {}
        logger.info("Initialized LocalVectorStore (in-memory)")

    def upsert(self, vectors: List[Dict[str, Any]], namespace: str = "") -> int:
        """Upsert vectors into the local store."""
        if namespace not in self._store:
            self._store[namespace] = {}

        count = 0
        for vec in vectors:
            vec_id = vec["id"]
            self._store[namespace][vec_id] = {
                "values": vec["values"],
                "metadata": vec.get("metadata", {}),
            }
            count += 1

        logger.debug(f"Upserted {count} vectors to namespace '{namespace}' (total: {len(self._store[namespace])})")
        return count

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter: Optional[Dict] = None,
        namespace: str = "",
    ) -> List[Dict[str, Any]]:
        """Search using cosine similarity."""
        if namespace not in self._store or not self._store[namespace]:
            return []

        query_vec = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []
        query_vec = query_vec / query_norm

        results = []
        for vec_id, data in self._store[namespace].items():
            # Apply metadata filter if provided
            if filter:
                metadata = data.get("metadata", {})
                if not self._matches_filter(metadata, filter):
                    continue

            stored_vec = np.array(data["values"], dtype=np.float32)
            stored_norm = np.linalg.norm(stored_vec)
            if stored_norm == 0:
                continue
            stored_vec = stored_vec / stored_norm

            # Cosine similarity (vectors are normalized, so dot product = cosine sim)
            score = float(np.dot(query_vec, stored_vec))
            results.append({
                "id": vec_id,
                "score": score,
                "metadata": data.get("metadata", {}),
            })

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def _matches_filter(self, metadata: Dict, filter: Dict) -> bool:
        """Check if metadata matches a simple equality filter."""
        for key, value in filter.items():
            if key not in metadata:
                return False
            if isinstance(value, dict):
                # Handle $eq, $in operators
                if "$eq" in value and metadata[key] != value["$eq"]:
                    return False
                if "$in" in value and metadata[key] not in value["$in"]:
                    return False
            elif metadata[key] != value:
                return False
        return True

    def delete(self, ids: Optional[List[str]] = None, filter: Optional[Dict] = None, namespace: str = "") -> None:
        """Delete vectors by ID or filter."""
        if namespace in self._store:
            if ids:
                for vec_id in ids:
                    self._store[namespace].pop(vec_id, None)
                logger.debug(f"Deleted {len(ids)} vectors from namespace '{namespace}'")
            elif filter:
                to_delete = []
                for vec_id, data in self._store[namespace].items():
                    if self._matches_filter(data.get("metadata", {}), filter):
                        to_delete.append(vec_id)
                for vec_id in to_delete:
                    self._store[namespace].pop(vec_id, None)
                logger.debug(f"Deleted {len(to_delete)} vectors using filter from namespace '{namespace}'")

    def get_stats(self) -> Dict[str, Any]:
        """Return store statistics."""
        total = sum(len(ns) for ns in self._store.values())
        return {
            "total_vectors": total,
            "namespaces": {ns: len(vecs) for ns, vecs in self._store.items()},
            "backend": "local_memory",
        }


class VectorStore(BaseVectorStore):
    """
    Pinecone-backed vector store for production use.
    Requires PINECONE_API_KEY in environment.
    """

    def __init__(
        self,
        api_key: str = None,
        index_name: str = None,
        dimension: int = 384,
    ):
        from pinecone import Pinecone, ServerlessSpec

        self.api_key = api_key or settings.PINECONE_API_KEY
        self.index_name = index_name or settings.PINECONE_INDEX_NAME
        self.dimension = dimension

        if not self.api_key:
            raise ValueError(
                "PINECONE_API_KEY is required. Set it in .env or pass directly."
            )

        self._client = Pinecone(api_key=self.api_key)

        existing_indexes = [idx.name for idx in self._client.list_indexes()]
        if self.index_name not in existing_indexes:
            logger.info(f"Creating Pinecone index '{self.index_name}' (dim={self.dimension})")
            self._client.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

        self._index = self._client.Index(self.index_name)
        logger.info(f"Connected to Pinecone index: {self.index_name}")

    def upsert(self, vectors: List[Dict[str, Any]], namespace: str = "") -> int:
        batch_size = 100
        total = 0
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            pinecone_vectors = [
                {"id": v["id"], "values": v["values"], "metadata": v.get("metadata", {})}
                for v in batch
            ]
            self._index.upsert(vectors=pinecone_vectors, namespace=namespace)
            total += len(batch)
        logger.debug(f"Upserted {total} vectors to Pinecone namespace '{namespace}'")
        return total

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter: Optional[Dict] = None,
        namespace: str = "",
    ) -> List[Dict[str, Any]]:
        kwargs = {
            "vector": query_embedding,
            "top_k": top_k,
            "include_metadata": True,
            "namespace": namespace,
        }
        if filter:
            kwargs["filter"] = filter

        response = self._index.query(**kwargs)
        results = []
        for match in response.matches:
            results.append({
                "id": match.id,
                "score": match.score,
                "metadata": match.metadata or {},
            })
        return results

    def delete(self, ids: Optional[List[str]] = None, filter: Optional[Dict] = None, namespace: str = "") -> None:
        if ids:
            self._index.delete(ids=ids, namespace=namespace)
            logger.debug(f"Deleted {len(ids)} vectors from Pinecone namespace '{namespace}'")
        elif filter:
            self._index.delete(filter=filter, namespace=namespace)
            logger.debug(f"Deleted vectors using filter {filter} from Pinecone namespace '{namespace}'")

    def get_stats(self) -> Dict[str, Any]:
        stats = self._index.describe_index_stats()
        return {
            "total_vectors": stats.total_vector_count,
            "namespaces": {
                ns: data.vector_count for ns, data in stats.namespaces.items()
            },
            "backend": "pinecone",
        }


# ============================================================
# Utility functions for hybrid search and MMR
# ============================================================

def _extract_keywords(text: str) -> set:
    """
    Extract meaningful keywords from text for keyword matching.
    Handles camelCase, snake_case, and regular words.
    """
    # Split camelCase and PascalCase
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    # Split snake_case
    text = text.replace("_", " ")
    # Tokenize and normalize
    tokens = re.findall(r'[a-zA-Z]{2,}', text.lower())
    # Remove very common stop words
    stop_words = {"the", "is", "in", "at", "to", "for", "of", "on", "and", "or", "it", "we", "do", "how", "does"}
    return set(tokens) - stop_words


def _keyword_score(query_keywords: set, document_text: str) -> float:
    """
    Simple keyword overlap score between query keywords and document text.
    Returns a score between 0.0 and 1.0.
    """
    if not query_keywords or not document_text:
        return 0.0

    doc_keywords = _extract_keywords(document_text)
    if not doc_keywords:
        return 0.0

    overlap = query_keywords & doc_keywords
    # Jaccard-like coefficient, weighted toward query coverage
    query_coverage = len(overlap) / len(query_keywords) if query_keywords else 0.0
    return query_coverage


def _content_similarity(content_a: str, content_b: str) -> float:
    """
    Simple content overlap similarity for MMR diversity calculation.
    Uses keyword Jaccard similarity as a fast proxy.
    """
    if not content_a or not content_b:
        return 0.0

    keywords_a = _extract_keywords(content_a)
    keywords_b = _extract_keywords(content_b)

    if not keywords_a or not keywords_b:
        return 0.0

    intersection = keywords_a & keywords_b
    union = keywords_a | keywords_b
    return len(intersection) / len(union) if union else 0.0


def create_vector_store(dimension: int = 384):
    """
    Factory function: returns Neo4j if configured, else Pinecone, else Local.
    """
    if settings.NEO4J_URI:
        logger.info("Using Neo4j GraphRAG store (URI found)")
        from zenith.vector_store.neo4j_store import Neo4jGraphStore
        return Neo4jGraphStore(dimension=dimension)
    elif settings.PINECONE_API_KEY:
        logger.info("Using Pinecone vector store (API key found)")
        return VectorStore(dimension=dimension)
    else:
        logger.warning("No PINECONE_API_KEY or NEO4J_URI found. Using local in-memory vector store.")
        return LocalVectorStore()
