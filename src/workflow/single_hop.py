# Single-hop Query Workflow
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from llama_index.llms.ollama import Ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

from src.retrieval.hybrid_search import HybridSearchEngine, EnrichedSearchResult
from src.retrieval.reranker import Reranker
from config.prompts import SINGLE_HOP_QUERY_PROMPT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SingleHopResponse:
    """Response from single-hop query"""
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    query: str


class SingleHopWorkflow:
    """
    Workflow for answering questions about specific contracts or clauses.
    
    Pipeline:
    1. Filter by metadata (contract_id, partner, year)
    2. Hybrid search (vector + keyword)
    3. Rerank top results
    4. Generate answer with LLM
    """
    
    def __init__(
        self,
        search_engine: HybridSearchEngine,
        reranker: Reranker,
        llm_model: str = "gpt-oss:20b",
        top_k_search: int = 10,
        top_k_rerank: int = 3
    ):
        self.search_engine = search_engine
        self.reranker = reranker
        self.top_k_search = top_k_search
        self.top_k_rerank = top_k_rerank
        
        # Initialize LLM
        self.llm = None
        if OLLAMA_AVAILABLE:
            try:
                self.llm = Ollama(
                    model=llm_model,
                    request_timeout=120.0,
                    context_window=32768
                )
                logger.info(f"Single-hop workflow using LLM: {llm_model}")
            except Exception as e:
                logger.error(f"Could not initialize LLM: {e}")
    
    async def run(
        self,
        query: str,
        contract_id: Optional[str] = None,
        partner_name: Optional[str] = None,
        year: Optional[int] = None,
        article_number: Optional[int] = None
    ) -> SingleHopResponse:
        """
        Execute single-hop query workflow.
        
        Args:
            query: User's question
            contract_id: Optional specific contract to search
            partner_name: Optional filter by partner
            year: Optional filter by year
            article_number: Optional filter by article
            
        Returns:
            SingleHopResponse with answer and sources
        """
        logger.info(f"Single-hop query: {query[:50]}...")
        
        # Step 1: Hybrid search with filters
        search_results = self.search_engine.search(
            query=query,
            top_k=self.top_k_search,
            contract_id=contract_id,
            partner_name=partner_name,
            year=year,
            include_parent_context=True,
            include_contract_metadata=True
        )
        
        if not search_results:
            return SingleHopResponse(
                answer="Không tìm thấy thông tin liên quan trong các hợp đồng.",
                sources=[],
                confidence=0.0,
                query=query
            )
        
        # Step 2: Rerank results
        results_for_rerank = [
            {
                'content': r.content,
                'score': r.score,
                'contract_id': r.contract_id,
                'article_number': r.article_number,
                'section_number': r.section_number,
                'parent_content': r.parent_content,
                'contract_metadata': r.contract_metadata
            }
            for r in search_results
        ]
        
        reranked = self.reranker.rerank_results(
            query=query,
            results=results_for_rerank,
            content_key='content',
            top_k=self.top_k_rerank
        )
        
        # Step 3: Build context from reranked results
        # Use parent content if available for better context
        context_parts = []
        sources = []
        
        for i, result in enumerate(reranked):
            # Use parent content if available, otherwise use chunk content
            content = result.get('parent_content') or result.get('content', '')
            
            # Build context entry
            contract_id = result.get('contract_id', 'Unknown')
            article_num = result.get('article_number', '')
            article_ref = f"Điều {article_num}" if article_num else ""
            
            context_parts.append(f"[Nguồn {i+1}: {contract_id} - {article_ref}]\n{content}\n")
            
            # Build source info
            sources.append({
                'contract_id': contract_id,
                'article_number': article_num,
                'section_number': result.get('section_number'),
                'score': result.get('rerank_score', result.get('score', 0)),
                'content_preview': result.get('content', '')[:200],
                'metadata': result.get('contract_metadata')
            })
        
        context = "\n---\n".join(context_parts)
        
        # Step 4: Generate answer with LLM
        answer = await self._generate_answer(query, context)
        
        # Calculate confidence based on rerank scores
        avg_score = sum(s.get('score', 0) for s in sources) / len(sources) if sources else 0
        confidence = min(avg_score, 1.0)
        
        return SingleHopResponse(
            answer=answer,
            sources=sources,
            confidence=confidence,
            query=query
        )
    
    def run_sync(
        self,
        query: str,
        contract_id: Optional[str] = None,
        partner_name: Optional[str] = None,
        year: Optional[int] = None,
        article_number: Optional[int] = None
    ) -> SingleHopResponse:
        """Synchronous version of run()"""
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            self.run(query, contract_id, partner_name, year, article_number)
        )
    
    async def _generate_answer(self, query: str, context: str) -> str:
        """Generate answer using LLM"""
        if not self.llm:
            return f"[LLM không khả dụng]\n\nNgữ cảnh tìm được:\n{context[:1000]}..."
        
        prompt = SINGLE_HOP_QUERY_PROMPT.format(
            context=context,
            question=query
        )
        
        try:
            response = self.llm.complete(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return f"Lỗi khi tạo câu trả lời: {e}\n\nNgữ cảnh tìm được:\n{context[:500]}..."
