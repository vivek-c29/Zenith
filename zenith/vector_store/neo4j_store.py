from typing import Any, Dict, List, Optional

from loguru import logger
from neo4j import GraphDatabase

from zenith.config.settings import settings
from zenith.parser.models import ParsedFile, CodeChunk
from zenith.vector_store.vector_store import BaseVectorStore


CODE_NAMESPACE = "code"
FIX_HISTORY_NAMESPACE = "fix_history"
VECTOR_INDEX_NAME = "chunk_embeddings_v2"


class Neo4jGraphStore(BaseVectorStore):
    """
    Neo4j-backed vector and code-relationship store.

    Code chunks are stored as:
        (File)-[:CONTAINS]->(CodeChunk)

    Function-call relationships are stored as:
        (CodeChunk)-[:CALLS]->(CodeChunk)

    Vector search is shared across code and fix-history nodes,
    but namespace filtering keeps the two datasets separate.
    """

    def __init__(
        self,
        uri: str = None,
        username: str = None,
        password: str = None,
        dimension: int = 384,
    ):
        self.uri = uri or settings.NEO4J_URI
        self.username = username or settings.NEO4J_USERNAME
        self.password = password or settings.NEO4J_PASSWORD
        self.dimension = dimension

        if not self.uri or not self.password:
            raise ValueError(
                "NEO4J_URI and NEO4J_PASSWORD are required for GraphRAG."
            )

        self.driver = GraphDatabase.driver(
            self.uri,
            auth=(self.username, self.password),
        )

        self._initialize_db()

    def _initialize_db(self):
        """Create constraints and the Neo4j vector index."""
        logger.info(
            "Initializing Neo4j Graph DB constraints and vector index..."
        )

        with self.driver.session() as session:
            session.run(
                """
                CREATE CONSTRAINT IF NOT EXISTS
                FOR (f:File)
                REQUIRE f.path IS UNIQUE
                """
            )

            session.run(
                """
                CREATE CONSTRAINT IF NOT EXISTS
                FOR (c:CodeChunk)
                REQUIRE c.id IS UNIQUE
                """
            )

            try:
                session.run(
                    f"""
                    CREATE VECTOR INDEX {VECTOR_INDEX_NAME} IF NOT EXISTS
                    FOR (c:CodeChunk)
                    ON c.embedding
                    WITH [c.namespace]
                    OPTIONS {{
                        indexConfig: {{
                            `vector.dimensions`: {self.dimension},
                            `vector.similarity_function`: 'cosine'
                        }}
                    }}
                    """
                )
            except Exception as e:
                logger.warning(
                    f"Vector index creation warning: {e}"
                )

    def close(self):
        self.driver.close()

    def delete(
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[Dict] = None,
        namespace: str = "",
    ) -> None:
        """Delete vectors/nodes by ID or file path."""
        if filter and "file_path" in filter:
            self.delete_file(
                filter["file_path"],
                namespace=namespace or CODE_NAMESPACE,
            )
            return

        if ids:
            with self.driver.session() as session:
                if namespace:
                    session.run(
                        """
                        MATCH (c:CodeChunk)
                        WHERE c.id IN $ids
                          AND c.namespace = $namespace
                        DETACH DELETE c
                        """,
                        ids=ids,
                        namespace=namespace,
                    )
                else:
                    session.run(
                        """
                        MATCH (c:CodeChunk)
                        WHERE c.id IN $ids
                        DETACH DELETE c
                        """,
                        ids=ids,
                    )

    def delete_file(
        self,
        file_path: str,
        namespace: str = CODE_NAMESPACE,
    ):
        """Delete a source file and all code chunks contained by it."""
        if namespace != CODE_NAMESPACE:
            return

        with self.driver.session() as session:
            session.run(
                """
                MATCH (f:File {path: $path})
                OPTIONAL MATCH (f)-[:CONTAINS]->(c:CodeChunk)
                WHERE c IS NULL OR c.namespace = $namespace
                DETACH DELETE f, c
                """,
                path=file_path,
                namespace=namespace,
            )

    def upsert_file_graph(
        self,
        parsed_file: ParsedFile,
        chunks: List[CodeChunk],
        embeddings: List[List[float]],
    ):
        """
        Upsert a source file, its chunks and metadata.

        CALLS relationships are refreshed after all chunks for this
        file are present.
        """
        file_path = parsed_file.file_path

        with self.driver.session() as session:
            # Create/update File node.
            session.run(
                """
                MERGE (f:File {path: $path})
                SET f.language = $language,
                    f.total_lines = $total_lines
                """,
                path=file_path,
                language=parsed_file.language,
                total_lines=parsed_file.total_lines,
            )

            # Create/update CodeChunk nodes.
            for chunk, embedding in zip(chunks, embeddings):
                symbol_kind = (
                    chunk.symbol_kind.value
                    if hasattr(chunk.symbol_kind, "value")
                    else (chunk.symbol_kind or "module")
                )

                session.run(
                    """
                    MATCH (f:File {path: $path})
                    MERGE (c:CodeChunk {id: $chunk_id})
                    SET c.namespace = $namespace,
                        c.file_path = $file_path,
                        c.content = $content,
                        c.symbol_name = $symbol_name,
                        c.symbol_kind = $symbol_kind,
                        c.start_line = $start_line,
                        c.end_line = $end_line,
                        c.language = $language,
                        c.token_count = $token_count,
                        c.calls = $calls,
                        c.embedding = $embedding
                    MERGE (f)-[:CONTAINS]->(c)
                    """,
                    path=file_path,
                    chunk_id=chunk.chunk_id,
                    namespace=CODE_NAMESPACE,
                    file_path=chunk.file_path,
                    content=chunk.content[:1000],
                    symbol_name=chunk.symbol_name or "",
                    symbol_kind=symbol_kind,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    language=chunk.language,
                    token_count=chunk.token_count,
                    calls=chunk.calls or [],
                    embedding=embedding,
                )

        # Rebuild relationships for this file after all nodes exist.
        self._refresh_calls_for_file(file_path)

    def _refresh_calls_for_file(self, file_path: str):
        """
        Refresh outgoing CALLS relationships for chunks belonging
        to one file.
        """
        with self.driver.session() as session:
            session.run(
                """
                MATCH (f:File {path: $file_path})
                      -[:CONTAINS]->(caller:CodeChunk)
                OPTIONAL MATCH (caller)-[r:CALLS]->()
                DELETE r
                """,
                file_path=file_path,
            )

            session.run(
                """
                MATCH (f:File {path: $file_path})
                      -[:CONTAINS]->(caller:CodeChunk)
                UNWIND coalesce(caller.calls, []) AS callee_name
                MATCH (callee:CodeChunk {
                    namespace: $namespace,
                    symbol_name: callee_name
                })
                MERGE (caller)-[:CALLS]->(callee)
                """,
                file_path=file_path,
                namespace=CODE_NAMESPACE,
            )

    def rebuild_all_call_relationships(self):
        """
        Rebuild all CALLS relationships from the stored `calls`
        metadata.

        Used after a full repository indexing operation.
        """
        with self.driver.session() as session:
            session.run(
                """
                MATCH ()-[r:CALLS]->()
                DELETE r
                """
            )

            session.run(
                """
                MATCH (caller:CodeChunk)
                WHERE caller.namespace = $namespace
                UNWIND coalesce(caller.calls, []) AS callee_name
                MATCH (callee:CodeChunk {
                    namespace: $namespace,
                    symbol_name: callee_name
                })
                MERGE (caller)-[:CALLS]->(callee)
                """,
                namespace=CODE_NAMESPACE,
            )

    def upsert(
        self,
        vectors: List[Dict[str, Any]],
        namespace: str = "",
    ) -> int:
        """
        Standard vector-store upsert.

        Used for fix-history vectors and as a fallback.
        Code indexing should normally use upsert_file_graph().
        """
        namespace = namespace or CODE_NAMESPACE
        count = 0

        with self.driver.session() as session:
            for vec in vectors:
                meta = vec.get("metadata", {})

                session.run(
                    """
                    MERGE (c:CodeChunk {id: $chunk_id})
                    SET c.namespace = $namespace,
                        c.file_path = $file_path,
                        c.content = $content,
                        c.symbol_name = $symbol_name,
                        c.symbol_kind = $symbol_kind,
                        c.language = $language,
                        c.start_line = $start_line,
                        c.end_line = $end_line,
                        c.token_count = $token_count,
                        c.embedding = $embedding
                    """,
                    chunk_id=vec["id"],
                    namespace=namespace,
                    file_path=meta.get("file_path"),
                    content=meta.get("content", "")[:1000],
                    symbol_name=meta.get("symbol_name", ""),
                    symbol_kind=meta.get("symbol_kind", "module"),
                    language=meta.get("language"),
                    start_line=meta.get("start_line"),
                    end_line=meta.get("end_line"),
                    token_count=meta.get("token_count"),
                    embedding=vec["values"],
                )

                count += 1

        return count

    def get_stats(self) -> Dict[str, Any]:
        """Return Neo4j vector statistics grouped by namespace."""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (c:CodeChunk)
                RETURN
                    count(c) AS total,
                    count(CASE WHEN c.namespace = $code THEN 1 END)
                        AS code_count,
                    count(CASE WHEN c.namespace = $fix THEN 1 END)
                        AS fix_count
                """,
                code=CODE_NAMESPACE,
                fix=FIX_HISTORY_NAMESPACE,
            )

            record = result.single()

            return {
                "backend": "neo4j",
                "total_vectors": record["total"],
                "namespaces": {
                    CODE_NAMESPACE: record["code_count"],
                    FIX_HISTORY_NAMESPACE: record["fix_count"],
                },
            }

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter: Optional[Dict] = None,
        namespace: str = CODE_NAMESPACE,
    ) -> List[Dict[str, Any]]:
        """
        Vector search using Neo4j SEARCH.

        Namespace filtering prevents code vectors and fix-history
        vectors from being mixed together.
        """
        namespace = namespace or CODE_NAMESPACE

        candidate_limit = max(top_k * 5, 50)

        with self.driver.session() as session:
            result = session.run(
                f"""
                MATCH (c:CodeChunk)
                SEARCH c IN (
                    VECTOR INDEX {VECTOR_INDEX_NAME}
                    FOR $query_vector
                    WHERE c.namespace = $namespace
                    LIMIT $candidate_limit
                )
                SCORE AS score

                OPTIONAL MATCH (f:File)-[:CONTAINS]->(c)

                WITH c, f, score
                WHERE
                    $namespace <> $code_namespace
                    OR f IS NOT NULL

                RETURN
                    c.id AS id,
                    score,
                    c.content AS content,
                    coalesce(f.path, c.file_path) AS file_path,
                    c.symbol_name AS symbol_name,
                    c.symbol_kind AS symbol_kind,
                    c.language AS language,
                    c.start_line AS start_line,
                    c.end_line AS end_line,
                    c.token_count AS token_count,
                    c.namespace AS namespace

                ORDER BY score DESC
                LIMIT $top_k
                """,
                query_vector=query_embedding,
                namespace=namespace,
                candidate_limit=candidate_limit,
                top_k=top_k,
                code_namespace=CODE_NAMESPACE,
            )

            records = []

            for record in result:
                metadata = {
                    "content": record["content"],
                    "file_path": record["file_path"],
                    "symbol_name": record["symbol_name"],
                    "symbol_kind": record["symbol_kind"],
                    "language": record["language"],
                    "start_line": record["start_line"],
                    "end_line": record["end_line"],
                    "token_count": record["token_count"],
                    "namespace": record["namespace"],
                }

                if filter:
                    # Neo4j SEARCH namespace filtering is handled above.
                    # Apply remaining simple metadata filters here.
                    matches = True

                    for key, value in filter.items():
                        if metadata.get(key) != value:
                            matches = False
                            break

                    if not matches:
                        continue

                records.append(
                    {
                        "id": record["id"],
                        "score": record["score"],
                        "metadata": metadata,
                    }
                )

            return records[:top_k]