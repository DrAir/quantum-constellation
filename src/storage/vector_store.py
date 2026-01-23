# Vector Store using Qdrant for Contract RAG
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
import numpy as np

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from qdrant_client.http.models import Distance, VectorParams, PointStruct
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    logging.warning("qdrant-client not installed. Run: pip install qdrant-client")

try:
    from FlagEmbedding import BGEM3FlagModel
    BGE_AVAILABLE = True
except ImportError:
    BGE_AVAILABLE = False
    logging.warning("FlagEmbedding not installed. Run: pip install FlagEmbedding")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Search result with score and metadata"""
    chunk_id: str
    content: str
    score: float
    contract_id: str
    article_number: Optional[int] = None
    section_number: Optional[str] = None
    chunk_type: str = "child"
    parent_id: Optional[str] = None
    metadata: Dict[str, Any] = None


class VectorStore:
    """
    Vector database operations using Qdrant.
    Supports both dense (BGE-M3) and sparse (BM25) vectors for hybrid search.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "contracts",
        embedding_model: str = "BAAI/bge-m3",
        use_sparse: bool = True
    ):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.use_sparse = use_sparse
        
        # Initialize Qdrant client
        if QDRANT_AVAILABLE:
            self.client = QdrantClient(host=host, port=port)
        else:
            self.client = None
            logger.error("Qdrant client not available")
        
        # Initialize embedding model
        self.embedding_model = None
        if BGE_AVAILABLE:
            logger.info(f"Loading embedding model: {embedding_model}")
            self.embedding_model = BGEM3FlagModel(
                embedding_model,
                use_fp16=True  # Use FP16 for faster inference
            )
            self.vector_size = 1024  # BGE-M3 dense vector size
        else:
            self.vector_size = 1024
            logger.warning("BGE model not available, embeddings will not work")
    
    def create_collection(self, recreate: bool = False):
        """Create the Qdrant collection for contracts"""
        if not self.client:
            raise RuntimeError("Qdrant client not initialized")
        
        # Check if collection exists
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        
        if exists and not recreate:
            logger.info(f"Collection {self.collection_name} already exists")
            return
        
        if exists and recreate:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted existing collection {self.collection_name}")
        
        # Create collection with dense and sparse vectors
        vectors_config = {
            "dense": VectorParams(
                size=self.vector_size,
                distance=Distance.COSINE
            )
        }
        
        # Add sparse vector config if enabled
        sparse_vectors_config = None
        if self.use_sparse:
            sparse_vectors_config = {
                "sparse": models.SparseVectorParams(
                    index=models.SparseIndexParams()
                )
            }
        
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors_config
        )
        
        logger.info(f"Created collection {self.collection_name}")
    
    def index_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 32):
        """
        Index chunks into Qdrant.
        
        Args:
            chunks: List of chunk dictionaries with 'content' and metadata
            batch_size: Number of chunks to process at once
        """
        if not self.client:
            raise RuntimeError("Qdrant client not initialized")
        
        if not self.embedding_model:
            raise RuntimeError("Embedding model not initialized")
        
        total_chunks = len(chunks)
        logger.info(f"Indexing {total_chunks} chunks...")
        
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i + batch_size]
            
            # Extract texts for embedding
            texts = [chunk['content'] for chunk in batch]
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(
                texts,
                return_dense=True,
                return_sparse=self.use_sparse,
                return_colbert_vecs=False
            )
            
            # Prepare points for Qdrant
            points = []
            for j, chunk in enumerate(batch):
                point_id = chunk.get('chunk_id', f"{i+j}")
                
                # Build vectors dict
                vectors = {
                    "dense": embeddings['dense_vecs'][j].tolist()
                }
                
                if self.use_sparse and 'lexical_weights' in embeddings:
                    # Convert sparse embeddings
                    sparse_dict = embeddings['lexical_weights'][j]
                    vectors["sparse"] = models.SparseVector(
                        indices=list(sparse_dict.keys()),
                        values=list(sparse_dict.values())
                    )
                
                # Build payload (metadata)
                payload = {
                    'content': chunk['content'],
                    'contract_id': chunk.get('contract_id', ''),
                    'chunk_type': chunk.get('chunk_type', 'child'),
                    'article_number': chunk.get('article_number'),
                    'article_title': chunk.get('article_title', ''),
                    'section_number': chunk.get('section_number'),
                    'parent_id': chunk.get('parent_id'),
                }
                
                points.append(PointStruct(
                    id=hash(point_id) % (2**63),  # Convert to int64
                    vector=vectors,
                    payload=payload
                ))
            
            # Upsert to Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            logger.info(f"Indexed {min(i + batch_size, total_chunks)}/{total_chunks} chunks")
        
        logger.info(f"Successfully indexed {total_chunks} chunks")
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_contract_id: Optional[str] = None,
        filter_year: Optional[int] = None,
        filter_partner: Optional[str] = None,
        chunk_type: Optional[str] = None,
        use_hybrid: bool = True
    ) -> List[SearchResult]:
        """
        Search for relevant chunks.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            filter_contract_id: Optional filter by contract ID
            filter_year: Optional filter by year
            filter_partner: Optional filter by partner name
            chunk_type: Optional filter by chunk type ('parent' or 'child')
            use_hybrid: Whether to use hybrid search (dense + sparse)
            
        Returns:
            List of SearchResult objects
        """
        if not self.client:
            raise RuntimeError("Qdrant client not initialized")
        
        if not self.embedding_model:
            raise RuntimeError("Embedding model not initialized")
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(
            [query],
            return_dense=True,
            return_sparse=self.use_sparse,
            return_colbert_vecs=False
        )
        
        # Build filter conditions
        filter_conditions = []
        
        if filter_contract_id:
            filter_conditions.append(
                models.FieldCondition(
                    key="contract_id",
                    match=models.MatchValue(value=filter_contract_id)
                )
            )
        
        if chunk_type:
            filter_conditions.append(
                models.FieldCondition(
                    key="chunk_type",
                    match=models.MatchValue(value=chunk_type)
                )
            )
        
        query_filter = None
        if filter_conditions:
            query_filter = models.Filter(must=filter_conditions)
        
        # Perform search using query() method (Qdrant v2.x API)
        try:
            # Use query_points for Qdrant v2.x
            dense_results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding['dense_vecs'][0].tolist(),
                using="dense",
                query_filter=query_filter,
                limit=top_k * 2 if use_hybrid else top_k
            ).points
            
            sparse_results = []
            if use_hybrid and self.use_sparse and 'lexical_weights' in query_embedding:
                sparse_dict = query_embedding['lexical_weights'][0]
                sparse_results = self.client.query_points(
                    collection_name=self.collection_name,
                    query=models.SparseVector(
                        indices=list(sparse_dict.keys()),
                        values=list(sparse_dict.values())
                    ),
                    using="sparse",
                    query_filter=query_filter,
                    limit=top_k * 2
                ).points
            
            # Fuse results using RRF if hybrid
            if use_hybrid and sparse_results:
                results = self._rrf_fusion(dense_results, sparse_results, top_k)
            else:
                results = dense_results[:top_k]
                
        except Exception as e:
            logger.error(f"Search error: {e}")
            # Fallback: try legacy search API
            try:
                results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=("dense", query_embedding['dense_vecs'][0].tolist()),
                    query_filter=query_filter,
                    limit=top_k
                )
            except Exception as e2:
                logger.error(f"Legacy search also failed: {e2}")
                return []
        
        # Convert to SearchResult objects
        search_results = []
        for result in results:
            payload = result.payload
            search_results.append(SearchResult(
                chunk_id=str(result.id),
                content=payload.get('content', ''),
                score=result.score,
                contract_id=payload.get('contract_id', ''),
                article_number=payload.get('article_number'),
                section_number=payload.get('section_number'),
                chunk_type=payload.get('chunk_type', 'child'),
                parent_id=payload.get('parent_id'),
                metadata=payload
            ))
        
        return search_results
    
    def _rrf_fusion(self, dense_results, sparse_results, top_k: int, k: int = 60):
        """
        Reciprocal Rank Fusion for combining dense and sparse results.
        """
        scores = {}
        
        # Score dense results
        for rank, result in enumerate(dense_results):
            scores[result.id] = scores.get(result.id, 0) + 1 / (k + rank + 1)
        
        # Score sparse results
        for rank, result in enumerate(sparse_results):
            scores[result.id] = scores.get(result.id, 0) + 1 / (k + rank + 1)
        
        # Sort by combined score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:top_k]
        
        # Get results in order
        all_results = {r.id: r for r in dense_results}
        all_results.update({r.id: r for r in sparse_results})
        
        fused_results = []
        for id_ in sorted_ids:
            if id_ in all_results:
                result = all_results[id_]
                result.score = scores[id_]
                fused_results.append(result)
        
        return fused_results
    
    def get_parent_chunk(self, parent_id: str) -> Optional[SearchResult]:
        """Retrieve a parent chunk by its ID"""
        if not self.client:
            return None
        
        try:
            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="chunk_id",
                            match=models.MatchValue(value=parent_id)
                        )
                    ]
                ),
                limit=1
            )
            
            if results[0]:
                point = results[0][0]
                payload = point.payload
                return SearchResult(
                    chunk_id=str(point.id),
                    content=payload.get('content', ''),
                    score=1.0,
                    contract_id=payload.get('contract_id', ''),
                    article_number=payload.get('article_number'),
                    section_number=payload.get('section_number'),
                    chunk_type=payload.get('chunk_type', 'parent'),
                    metadata=payload
                )
        except Exception as e:
            logger.error(f"Error getting parent chunk: {e}")
        
        return None
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        if not self.client:
            return {}
        
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                'name': self.collection_name,
                'vectors_count': info.vectors_count,
                'points_count': info.points_count,
                'status': info.status.value
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}
