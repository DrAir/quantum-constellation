# Reranker using BGE-Reranker-v2-m3
from typing import List, Tuple
from dataclasses import dataclass
import logging

try:
    from FlagEmbedding import FlagReranker
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False
    logging.warning("FlagEmbedding reranker not installed. Run: pip install FlagEmbedding")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    """Result after reranking"""
    content: str
    original_score: float
    rerank_score: float
    original_rank: int
    new_rank: int
    metadata: dict = None


class Reranker:
    """
    Cross-encoder reranker using BGE-Reranker-v2-m3.
    Provides more accurate relevance scoring than bi-encoder retrieval.
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "cpu",
        use_fp16: bool = True
    ):
        self.model_name = model_name
        self.device = device
        self.reranker = None
        
        if RERANKER_AVAILABLE:
            logger.info(f"Loading reranker model: {model_name}")
            self.reranker = FlagReranker(
                model_name,
                use_fp16=use_fp16
            )
            logger.info("Reranker loaded successfully")
        else:
            logger.warning("Reranker not available, using original scores")
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 3,
        return_scores: bool = True
    ) -> List[Tuple[int, float]]:
        """
        Rerank documents based on relevance to query.
        
        Args:
            query: The search query
            documents: List of document texts to rerank
            top_k: Number of top results to return
            return_scores: Whether to return scores
            
        Returns:
            List of (original_index, score) tuples sorted by relevance
        """
        if not self.reranker:
            # Fallback: return original order
            return [(i, 1.0) for i in range(min(top_k, len(documents)))]
        
        if not documents:
            return []
        
        # Prepare pairs for reranking
        pairs = [[query, doc] for doc in documents]
        
        # Get scores from reranker
        scores = self.reranker.compute_score(pairs)
        
        # Handle single document case
        if isinstance(scores, float):
            scores = [scores]
        
        # Create (index, score) pairs and sort by score
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k results
        return indexed_scores[:top_k]
    
    def rerank_results(
        self,
        query: str,
        results: List[dict],
        content_key: str = "content",
        top_k: int = 3
    ) -> List[dict]:
        """
        Rerank search results and return reordered list.
        
        Args:
            query: The search query
            results: List of result dictionaries
            content_key: Key to access text content in results
            top_k: Number of results to return
            
        Returns:
            Reranked and filtered list of results
        """
        if not results:
            return []
        
        # Extract documents
        documents = [r.get(content_key, "") for r in results]
        
        # Rerank
        reranked = self.rerank(query, documents, top_k=top_k)
        
        # Build result list with rerank scores
        reranked_results = []
        for new_rank, (orig_idx, score) in enumerate(reranked):
            result = results[orig_idx].copy()
            result['rerank_score'] = score
            result['original_rank'] = orig_idx
            result['new_rank'] = new_rank
            reranked_results.append(result)
        
        return reranked_results
    
    def rerank_with_details(
        self,
        query: str,
        results: List[dict],
        content_key: str = "content",
        top_k: int = 3
    ) -> List[RerankResult]:
        """
        Rerank with detailed result objects.
        """
        if not results:
            return []
        
        documents = [r.get(content_key, "") for r in results]
        reranked = self.rerank(query, documents, top_k=top_k)
        
        detailed_results = []
        for new_rank, (orig_idx, rerank_score) in enumerate(reranked):
            orig_result = results[orig_idx]
            detailed_results.append(RerankResult(
                content=orig_result.get(content_key, ""),
                original_score=orig_result.get('score', 0.0),
                rerank_score=rerank_score,
                original_rank=orig_idx,
                new_rank=new_rank,
                metadata={k: v for k, v in orig_result.items() if k != content_key}
            ))
        
        return detailed_results


# CLI test
if __name__ == "__main__":
    reranker = Reranker()
    
    query = "Điều kiện phạt hợp đồng"
    documents = [
        "Điều 9. Bồi thường thiệt hại và các điều khoản phạt hợp đồng. Nếu một trong hai Bên không thực hiện đầy đủ...",
        "Điều 5. Thời gian thực hiện hợp đồng. Thời gian thực hiện là 15 ngày kể từ ngày ký...",
        "Điều 7. Bảo hành. Bên B có trách nhiệm bảo hành trong 12 tháng...",
        "Phạt chậm thực hiện hợp đồng: Bên B phải chịu mức phạt 0.5% giá trị cho mỗi ngày chậm..."
    ]
    
    results = reranker.rerank(query, documents, top_k=3)
    
    print(f"\nQuery: {query}")
    print("\nReranked results:")
    for idx, score in results:
        print(f"  Score: {score:.4f} - {documents[idx][:60]}...")
