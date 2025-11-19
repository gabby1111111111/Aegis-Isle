"""
Document retrieval components for RAG pipeline.
"""

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel

from ..core.config import settings
from ..core.logging import logger
from .document_processor import DocumentChunk


class RetrievalResult(BaseModel):
    """Result of a retrieval operation."""

    chunk: DocumentChunk
    score: float
    metadata: Dict[str, Any] = {}


class QueryResult(BaseModel):
    """Complete result of a query operation."""

    query: str
    results: List[RetrievalResult]
    total_time: float
    metadata: Dict[str, Any] = {}


class BaseRetriever(ABC):
    """Base class for document retrievers."""

    def __init__(self, **kwargs):
        self.config = kwargs

    @abstractmethod
    async def add_chunks(self, chunks: List[DocumentChunk]) -> bool:
        """Add document chunks to the retriever."""
        pass

    @abstractmethod
    async def search(
        self,
        query: str,
        limit: int = 5,
        **kwargs
    ) -> QueryResult:
        """Search for relevant chunks."""
        pass

    @abstractmethod
    async def delete_document(self, document_id: str) -> bool:
        """Delete all chunks for a document."""
        pass

    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        pass


class VectorRetriever(BaseRetriever):
    """Vector-based retrieval using embedding similarity."""

    def __init__(
        self,
        embedding_model: str = "text-embedding-ada-002",
        vector_db_type: str = "qdrant",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.embedding_model = embedding_model
        self.vector_db_type = vector_db_type
        self._embedder = None
        self._vector_db = None
        self._initialize_components()

    def _initialize_components(self):
        """Initialize embedding and vector database components."""
        self._initialize_embedder()
        self._initialize_vector_db()

    def _initialize_embedder(self):
        """Initialize the embedding model."""
        try:
            if "openai" in self.embedding_model.lower():
                from openai import OpenAI
                self._embedder = OpenAI(api_key=settings.openai_api_key)
                self._embed_method = self._openai_embed
            else:
                # Use sentence transformers for other models
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(self.embedding_model)
                self._embed_method = self._sentence_transformer_embed

            logger.info(f"Initialized embedding model: {self.embedding_model}")

        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise

    def _initialize_vector_db(self):
        """Initialize the vector database."""
        if self.vector_db_type == "qdrant":
            self._initialize_qdrant()
        elif self.vector_db_type == "chromadb":
            self._initialize_chromadb()
        elif self.vector_db_type == "faiss":
            self._initialize_faiss()
        else:
            raise ValueError(f"Unsupported vector database: {self.vector_db_type}")

    def _initialize_qdrant(self):
        """Initialize Qdrant vector database."""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams, PointStruct

            self._vector_db = QdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port
            )

            # Ensure collection exists
            collections = self._vector_db.get_collections().collections
            collection_names = [c.name for c in collections]

            if settings.qdrant_collection not in collection_names:
                self._vector_db.create_collection(
                    collection_name=settings.qdrant_collection,
                    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
                )
                logger.info(f"Created Qdrant collection: {settings.qdrant_collection}")

            logger.info("Initialized Qdrant vector database")

        except Exception as e:
            logger.error(f"Failed to initialize Qdrant: {e}")
            raise

    def _initialize_chromadb(self):
        """Initialize ChromaDB vector database."""
        try:
            import chromadb

            self._vector_db = chromadb.Client()
            self._collection = self._vector_db.get_or_create_collection(
                name=settings.qdrant_collection  # Reuse collection name setting
            )

            logger.info("Initialized ChromaDB vector database")

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise

    def _initialize_faiss(self):
        """Initialize FAISS vector database."""
        try:
            import faiss
            import numpy as np

            # Initialize with a basic index (can be improved)
            self._dimension = 1536  # OpenAI embedding dimension
            self._vector_db = faiss.IndexFlatIP(self._dimension)  # Inner Product
            self._id_to_chunk = {}  # Map FAISS IDs to chunks

            logger.info("Initialized FAISS vector database")

        except Exception as e:
            logger.error(f"Failed to initialize FAISS: {e}")
            raise

    async def _openai_embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI."""
        try:
            response = self._embedder.embeddings.create(
                model=self.embedding_model,
                input=texts
            )
            return [item.embedding for item in response.data]

        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            raise

    async def _sentence_transformer_embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using SentenceTransformers."""
        try:
            embeddings = self._embedder.encode(texts, convert_to_tensor=False)
            return embeddings.tolist()

        except Exception as e:
            logger.error(f"SentenceTransformer embedding failed: {e}")
            raise

    async def add_chunks(self, chunks: List[DocumentChunk]) -> bool:
        """Add document chunks to the vector database."""
        if not chunks:
            return True

        try:
            # Generate embeddings for chunks
            texts = [chunk.content for chunk in chunks]
            embeddings = await self._embed_method(texts)

            # Add to vector database
            if self.vector_db_type == "qdrant":
                await self._add_to_qdrant(chunks, embeddings)
            elif self.vector_db_type == "chromadb":
                await self._add_to_chromadb(chunks, embeddings)
            elif self.vector_db_type == "faiss":
                await self._add_to_faiss(chunks, embeddings)

            logger.info(f"Added {len(chunks)} chunks to vector database")
            return True

        except Exception as e:
            logger.error(f"Failed to add chunks to vector database: {e}")
            return False

    async def _add_to_qdrant(self, chunks: List[DocumentChunk], embeddings: List[List[float]]):
        """Add chunks to Qdrant."""
        from qdrant_client.models import PointStruct

        points = []
        for chunk, embedding in zip(chunks, embeddings):
            point = PointStruct(
                id=chunk.id,
                vector=embedding,
                payload={
                    "document_id": chunk.document_id,
                    "content": chunk.content,
                    "chunk_index": chunk.chunk_index,
                    "metadata": chunk.metadata
                }
            )
            points.append(point)

        self._vector_db.upsert(
            collection_name=settings.qdrant_collection,
            points=points
        )

    async def _add_to_chromadb(self, chunks: List[DocumentChunk], embeddings: List[List[float]]):
        """Add chunks to ChromaDB."""
        self._collection.upsert(
            ids=[chunk.id for chunk in chunks],
            embeddings=embeddings,
            documents=[chunk.content for chunk in chunks],
            metadatas=[{
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                **chunk.metadata
            } for chunk in chunks]
        )

    async def _add_to_faiss(self, chunks: List[DocumentChunk], embeddings: List[List[float]]):
        """Add chunks to FAISS."""
        import numpy as np

        # Convert embeddings to numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)

        # Add to FAISS index
        start_id = self._vector_db.ntotal
        self._vector_db.add(embeddings_array)

        # Store chunk mappings
        for i, chunk in enumerate(chunks):
            self._id_to_chunk[start_id + i] = chunk

    async def search(
        self,
        query: str,
        limit: int = 5,
        **kwargs
    ) -> QueryResult:
        """Search for relevant chunks."""
        start_time = time.time()

        try:
            # Generate query embedding
            query_embedding = (await self._embed_method([query]))[0]

            # Search in vector database
            if self.vector_db_type == "qdrant":
                results = await self._search_qdrant(query_embedding, limit, **kwargs)
            elif self.vector_db_type == "chromadb":
                results = await self._search_chromadb(query_embedding, limit, **kwargs)
            elif self.vector_db_type == "faiss":
                results = await self._search_faiss(query_embedding, limit, **kwargs)
            else:
                results = []

            total_time = time.time() - start_time

            return QueryResult(
                query=query,
                results=results,
                total_time=total_time,
                metadata={
                    "vector_db_type": self.vector_db_type,
                    "embedding_model": self.embedding_model,
                    "limit": limit
                }
            )

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return QueryResult(
                query=query,
                results=[],
                total_time=time.time() - start_time,
                metadata={"error": str(e)}
            )

    async def _search_qdrant(
        self,
        query_embedding: List[float],
        limit: int,
        **kwargs
    ) -> List[RetrievalResult]:
        """Search in Qdrant."""
        search_result = self._vector_db.search(
            collection_name=settings.qdrant_collection,
            query_vector=query_embedding,
            limit=limit,
            score_threshold=kwargs.get("score_threshold", 0.0)
        )

        results = []
        for hit in search_result:
            chunk = DocumentChunk(
                id=hit.id,
                document_id=hit.payload["document_id"],
                content=hit.payload["content"],
                chunk_index=hit.payload["chunk_index"],
                metadata=hit.payload.get("metadata", {})
            )

            result = RetrievalResult(
                chunk=chunk,
                score=hit.score,
                metadata={"source": "qdrant"}
            )
            results.append(result)

        return results

    async def _search_chromadb(
        self,
        query_embedding: List[float],
        limit: int,
        **kwargs
    ) -> List[RetrievalResult]:
        """Search in ChromaDB."""
        search_result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=limit
        )

        results = []
        for i in range(len(search_result["ids"][0])):
            chunk = DocumentChunk(
                id=search_result["ids"][0][i],
                document_id=search_result["metadatas"][0][i]["document_id"],
                content=search_result["documents"][0][i],
                chunk_index=search_result["metadatas"][0][i]["chunk_index"],
                metadata=search_result["metadatas"][0][i]
            )

            # ChromaDB returns distances, convert to similarity scores
            distance = search_result["distances"][0][i]
            score = 1.0 / (1.0 + distance)  # Simple conversion

            result = RetrievalResult(
                chunk=chunk,
                score=score,
                metadata={"source": "chromadb", "distance": distance}
            )
            results.append(result)

        return results

    async def _search_faiss(
        self,
        query_embedding: List[float],
        limit: int,
        **kwargs
    ) -> List[RetrievalResult]:
        """Search in FAISS."""
        import numpy as np

        query_array = np.array([query_embedding], dtype=np.float32)
        scores, indices = self._vector_db.search(query_array, limit)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx in self._id_to_chunk:
                chunk = self._id_to_chunk[idx]
                result = RetrievalResult(
                    chunk=chunk,
                    score=float(score),
                    metadata={"source": "faiss", "index": int(idx)}
                )
                results.append(result)

        return results

    async def delete_document(self, document_id: str) -> bool:
        """Delete all chunks for a document."""
        try:
            if self.vector_db_type == "qdrant":
                from qdrant_client.models import Filter, FieldCondition, MatchValue

                self._vector_db.delete(
                    collection_name=settings.qdrant_collection,
                    points_selector=Filter(
                        must=[
                            FieldCondition(
                                key="document_id",
                                match=MatchValue(value=document_id)
                            )
                        ]
                    )
                )

            elif self.vector_db_type == "chromadb":
                # ChromaDB doesn't have a direct way to delete by metadata
                # Would need to query first, then delete by IDs
                logger.warning("ChromaDB document deletion not implemented")
                return False

            elif self.vector_db_type == "faiss":
                # FAISS doesn't support deletion - would need rebuild
                logger.warning("FAISS document deletion not implemented")
                return False

            logger.info(f"Deleted document {document_id} from vector database")
            return True

        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        try:
            if self.vector_db_type == "qdrant":
                info = self._vector_db.get_collection(settings.qdrant_collection)
                return {
                    "total_chunks": info.vectors_count,
                    "vector_dimension": info.config.params.vectors.size,
                    "distance_metric": info.config.params.vectors.distance,
                }

            elif self.vector_db_type == "chromadb":
                return {
                    "total_chunks": self._collection.count(),
                }

            elif self.vector_db_type == "faiss":
                return {
                    "total_chunks": self._vector_db.ntotal,
                    "vector_dimension": self._dimension,
                }

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")

        return {}


class HybridRetriever(BaseRetriever):
    """Hybrid retrieval combining multiple retrieval strategies."""

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        keyword_weight: float = 0.3,
        vector_weight: float = 0.7,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vector_retriever = vector_retriever
        self.keyword_weight = keyword_weight
        self.vector_weight = vector_weight
        self._keyword_index = {}  # Simple keyword index

    async def add_chunks(self, chunks: List[DocumentChunk]) -> bool:
        """Add chunks to both vector and keyword indices."""
        # Add to vector retriever
        vector_success = await self.vector_retriever.add_chunks(chunks)

        # Add to keyword index
        for chunk in chunks:
            words = set(chunk.content.lower().split())
            for word in words:
                if word not in self._keyword_index:
                    self._keyword_index[word] = []
                self._keyword_index[word].append(chunk)

        return vector_success

    async def search(
        self,
        query: str,
        limit: int = 5,
        **kwargs
    ) -> QueryResult:
        """Hybrid search combining vector and keyword results."""
        start_time = time.time()

        try:
            # Get vector search results
            vector_results = await self.vector_retriever.search(
                query, limit * 2, **kwargs  # Get more results for reranking
            )

            # Get keyword search results
            keyword_results = self._keyword_search(query, limit * 2)

            # Combine and rerank results
            combined_results = self._combine_results(
                vector_results.results,
                keyword_results,
                limit
            )

            total_time = time.time() - start_time

            return QueryResult(
                query=query,
                results=combined_results,
                total_time=total_time,
                metadata={
                    "retrieval_type": "hybrid",
                    "vector_weight": self.vector_weight,
                    "keyword_weight": self.keyword_weight
                }
            )

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return QueryResult(
                query=query,
                results=[],
                total_time=time.time() - start_time,
                metadata={"error": str(e)}
            )

    def _keyword_search(self, query: str, limit: int) -> List[RetrievalResult]:
        """Simple keyword-based search."""
        query_words = set(query.lower().split())
        chunk_scores = {}

        # Score chunks based on keyword matches
        for word in query_words:
            if word in self._keyword_index:
                for chunk in self._keyword_index[word]:
                    if chunk.id not in chunk_scores:
                        chunk_scores[chunk.id] = {"chunk": chunk, "score": 0}
                    chunk_scores[chunk.id]["score"] += 1

        # Sort by score and convert to RetrievalResult
        sorted_chunks = sorted(
            chunk_scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )

        results = []
        for item in sorted_chunks[:limit]:
            result = RetrievalResult(
                chunk=item["chunk"],
                score=item["score"] / len(query_words),  # Normalize
                metadata={"source": "keyword"}
            )
            results.append(result)

        return results

    def _combine_results(
        self,
        vector_results: List[RetrievalResult],
        keyword_results: List[RetrievalResult],
        limit: int
    ) -> List[RetrievalResult]:
        """Combine and rerank vector and keyword results."""
        # Create a map of all unique chunks
        all_chunks = {}

        # Add vector results
        for result in vector_results:
            chunk_id = result.chunk.id
            all_chunks[chunk_id] = {
                "chunk": result.chunk,
                "vector_score": result.score,
                "keyword_score": 0.0
            }

        # Add keyword results
        for result in keyword_results:
            chunk_id = result.chunk.id
            if chunk_id in all_chunks:
                all_chunks[chunk_id]["keyword_score"] = result.score
            else:
                all_chunks[chunk_id] = {
                    "chunk": result.chunk,
                    "vector_score": 0.0,
                    "keyword_score": result.score
                }

        # Calculate hybrid scores
        hybrid_results = []
        for chunk_data in all_chunks.values():
            hybrid_score = (
                self.vector_weight * chunk_data["vector_score"] +
                self.keyword_weight * chunk_data["keyword_score"]
            )

            result = RetrievalResult(
                chunk=chunk_data["chunk"],
                score=hybrid_score,
                metadata={
                    "source": "hybrid",
                    "vector_score": chunk_data["vector_score"],
                    "keyword_score": chunk_data["keyword_score"]
                }
            )
            hybrid_results.append(result)

        # Sort by hybrid score and return top results
        hybrid_results.sort(key=lambda x: x.score, reverse=True)
        return hybrid_results[:limit]

    async def delete_document(self, document_id: str) -> bool:
        """Delete document from both indices."""
        vector_success = await self.vector_retriever.delete_document(document_id)

        # Remove from keyword index
        chunks_to_remove = []
        for word_chunks in self._keyword_index.values():
            chunks_to_remove.extend([
                chunk for chunk in word_chunks
                if chunk.document_id == document_id
            ])

        for chunk in chunks_to_remove:
            for word, word_chunks in self._keyword_index.items():
                self._keyword_index[word] = [
                    c for c in word_chunks if c.document_id != document_id
                ]

        return vector_success

    async def get_stats(self) -> Dict[str, Any]:
        """Get hybrid retriever statistics."""
        vector_stats = await self.vector_retriever.get_stats()
        keyword_stats = {
            "keyword_vocabulary_size": len(self._keyword_index),
            "total_keyword_entries": sum(
                len(chunks) for chunks in self._keyword_index.values()
            )
        }

        return {
            **vector_stats,
            **keyword_stats,
            "retrieval_type": "hybrid"
        }