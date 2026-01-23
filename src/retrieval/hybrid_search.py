# Hybrid Search Engine for Contract RAG
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.storage.vector_store import VectorStore, SearchResult
from src.storage.metadata_store import MetadataStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EnrichedSearchResult:
    """Search result enriched with parent context"""
    chunk_id: str
    content: str
    score: float
    contract_id: str
    article_number: Optional[int] = None
    section_number: Optional[str] = None
    chunk_type: str = "child"
    parent_content: Optional[str] = None  # Full article context
    contract_metadata: Optional[Dict[str, Any]] = None


class HybridSearchEngine:
    """
    Hybrid search engine combining:
    1. Metadata filtering (SQLite)
    2. Vector search (Qdrant with BGE-M3)
    3. Keyword search (BM25 sparse vectors)
    4. Parent chunk enrichment
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        metadata_store: MetadataStore,
        alpha: float = 0.5  # Balance between dense and sparse
    ):
        self.vector_store = vector_store
        self.metadata_store = metadata_store
        self.alpha = alpha
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        # Metadata filters
        contract_id: Optional[str] = None,
        partner_name: Optional[str] = None,
        year: Optional[int] = None,
        contract_type: Optional[str] = None,
        # Search options
        include_parent_context: bool = True,
        include_contract_metadata: bool = True,
        chunk_type: Optional[str] = None  # 'parent' or 'child'
    ) -> List[EnrichedSearchResult]:
        """
        Perform hybrid search with metadata filtering.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            contract_id: Filter by specific contract
            partner_name: Filter by partner name
            year: Filter by year
            contract_type: Filter by contract type
            include_parent_context: Whether to fetch parent chunks for context
            include_contract_metadata: Whether to include contract metadata
            chunk_type: Filter by chunk type
            
        Returns:
            List of enriched search results
        """
        logger.info(f"Searching for: {query[:50]}...")
        
        # Step 1: Resolve contract_id if it looks like a contract number (e.g., "112/2024")
        resolved_contract_id = None
        if contract_id:
            # Check if this looks like a contract number pattern (contains /)
            if '/' in contract_id:
                # Try to find the actual contract_id by matching contract_number
                contracts = self.metadata_store.search_contracts(limit=100)
                for c in contracts:
                    cn = c.get('contract_number', '')
                    # Match if contract_number starts with the query pattern
                    if cn and contract_id in cn:
                        resolved_contract_id = c['id']
                        logger.info(f"Resolved contract number '{contract_id}' to ID: {resolved_contract_id}")
                        break
            else:
                # Try partial match on contract_id or contract_number
                contracts = self.metadata_store.search_contracts(limit=100)
                for c in contracts:
                    cid = c.get('id', '')
                    cn = c.get('contract_number', '')
                    # Match if ID or number contains the query
                    if contract_id in cid or contract_id in cn:
                        resolved_contract_id = c['id']
                        logger.info(f"Resolved contract '{contract_id}' to ID: {resolved_contract_id}")
                        break
            
            if not resolved_contract_id:
                logger.warning(f"Could not resolve contract: {contract_id}")
                # Continue without filter - maybe vector search will find something
        
        # Step 2: Pre-filter using metadata if filters provided
        filtered_contract_ids = None
        if partner_name or year or contract_type:
            filtered_contract_ids = self.metadata_store.get_contract_ids(
                partner_name=partner_name,
                year=year,
                contract_type=contract_type
            )
            
            if not filtered_contract_ids:
                logger.warning("No contracts match the metadata filters")
                return []
            
            logger.info(f"Pre-filtered to {len(filtered_contract_ids)} contracts")
        
        # Step 3: Vector search (hybrid with dense + sparse)
        search_results = self.vector_store.search(
            query=query,
            top_k=top_k * 2,  # Get more for filtering
            filter_contract_id=resolved_contract_id,  # Use resolved ID
            chunk_type=chunk_type,
            use_hybrid=True
        )
        
        # Step 3: Filter by contract IDs if we have metadata filters
        if filtered_contract_ids:
            search_results = [
                r for r in search_results 
                if r.contract_id in filtered_contract_ids
            ]
        
        # Take top_k results
        search_results = search_results[:top_k]
        
        # Step 4: Enrich results
        enriched_results = []
        for result in search_results:
            enriched = EnrichedSearchResult(
                chunk_id=result.chunk_id,
                content=result.content,
                score=result.score,
                contract_id=result.contract_id,
                article_number=result.article_number,
                section_number=result.section_number,
                chunk_type=result.chunk_type
            )
            
            # Get parent context if this is a child chunk
            if include_parent_context and result.parent_id:
                parent = self.vector_store.get_parent_chunk(result.parent_id)
                if parent:
                    enriched.parent_content = parent.content
            
            # Get contract metadata
            if include_contract_metadata:
                contract_meta = self.metadata_store.get_contract(result.contract_id)
                if contract_meta:
                    enriched.contract_metadata = contract_meta
            
            enriched_results.append(enriched)
        
        logger.info(f"Found {len(enriched_results)} results")
        return enriched_results
    
    def search_by_article(
        self,
        query: str,
        article_number: int,
        contract_id: Optional[str] = None,
        top_k: int = 5
    ) -> List[EnrichedSearchResult]:
        """
        Search within a specific article (Điều) across contracts.
        """
        # Search with parent chunks only
        results = self.search(
            query=query,
            top_k=top_k * 3,
            contract_id=contract_id,
            chunk_type='parent',
            include_parent_context=False
        )
        
        # Filter by article number
        filtered = [r for r in results if r.article_number == article_number]
        
        return filtered[:top_k]
    
    def get_article_content(
        self,
        contract_id: str,
        article_number: int
    ) -> Optional[str]:
        """
        Get the full content of a specific article from a contract.
        """
        results = self.search(
            query=f"Điều {article_number}",
            top_k=1,
            contract_id=contract_id,
            chunk_type='parent',
            include_parent_context=False
        )
        
        for r in results:
            if r.article_number == article_number:
                return r.content
        
        return None
    
    def find_similar_clauses(
        self,
        clause_content: str,
        exclude_contract_id: Optional[str] = None,
        top_k: int = 10
    ) -> List[EnrichedSearchResult]:
        """
        Find similar clauses across all contracts.
        Useful for comparing terms across different contracts.
        """
        results = self.search(
            query=clause_content,
            top_k=top_k + 5,  # Get extra to account for exclusion
            include_parent_context=True
        )
        
        if exclude_contract_id:
            results = [r for r in results if r.contract_id != exclude_contract_id]
        
        return results[:top_k]
