# Multi-document Summarization Workflow (Map-Reduce)
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import logging
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from llama_index.llms.ollama import Ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

from src.storage.metadata_store import MetadataStore
from src.retrieval.hybrid_search import HybridSearchEngine
from config.prompts import MAP_SUMMARIZE_PROMPT, REDUCE_SUMMARIZE_PROMPT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MultiDocResponse:
    """Response from multi-document summarization"""
    summary: str
    contracts_analyzed: int
    contract_summaries: List[Dict[str, Any]]
    total_value: Optional[float]
    query: str


class MultiDocWorkflow:
    """
    Workflow for queries requiring analysis across multiple contracts.
    Uses Map-Reduce pattern:
    
    1. Filter contracts by metadata
    2. MAP: Extract relevant info from each contract
    3. REDUCE: Synthesize into final summary
    """
    
    def __init__(
        self,
        search_engine: HybridSearchEngine,
        metadata_store: MetadataStore,
        llm_model: str = "gpt-oss:20b",
        batch_size: int = 5,
        max_contracts: int = 50
    ):
        self.search_engine = search_engine
        self.metadata_store = metadata_store
        self.batch_size = batch_size
        self.max_contracts = max_contracts
        
        # Initialize LLM
        self.llm = None
        if OLLAMA_AVAILABLE:
            try:
                self.llm = Ollama(
                    model=llm_model,
                    request_timeout=180.0,
                    context_window=32768
                )
                logger.info(f"Multi-doc workflow using LLM: {llm_model}")
            except Exception as e:
                logger.error(f"Could not initialize LLM: {e}")
    
    async def run(
        self,
        query: str,
        partner_name: Optional[str] = None,
        year: Optional[int] = None,
        contract_type: Optional[str] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None
    ) -> MultiDocResponse:
        """
        Execute multi-document summarization workflow.
        
        Args:
            query: User's question/request
            partner_name: Filter by partner
            year: Filter by year
            contract_type: Filter by contract type
            min_value: Minimum contract value
            max_value: Maximum contract value
            
        Returns:
            MultiDocResponse with summary and analysis
        """
        logger.info(f"Multi-doc query: {query[:50]}...")
        
        # Step 1: Filter contracts by metadata
        contracts = self.metadata_store.search_contracts(
            partner_name=partner_name,
            year=year,
            contract_type=contract_type,
            min_value=min_value,
            max_value=max_value,
            limit=self.max_contracts
        )
        
        if not contracts:
            return MultiDocResponse(
                summary="Không tìm thấy hợp đồng nào phù hợp với tiêu chí tìm kiếm.",
                contracts_analyzed=0,
                contract_summaries=[],
                total_value=None,
                query=query
            )
        
        logger.info(f"Found {len(contracts)} contracts to analyze")
        
        # Calculate total value
        total_value = sum(c.get('total_value', 0) or 0 for c in contracts)
        
        # Step 2: MAP - Extract relevant info from each contract
        contract_summaries = await self._map_phase(query, contracts)
        
        # Step 3: REDUCE - Synthesize final summary
        final_summary = await self._reduce_phase(query, contract_summaries)
        
        return MultiDocResponse(
            summary=final_summary,
            contracts_analyzed=len(contracts),
            contract_summaries=contract_summaries,
            total_value=total_value,
            query=query
        )
    
    def run_sync(
        self,
        query: str,
        partner_name: Optional[str] = None,
        year: Optional[int] = None,
        contract_type: Optional[str] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None
    ) -> MultiDocResponse:
        """Synchronous version of run()"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(
            self.run(query, partner_name, year, contract_type, min_value, max_value)
        )
    
    async def _map_phase(
        self,
        query: str,
        contracts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        MAP phase: Extract relevant information from each contract.
        Processes contracts in batches for efficiency.
        """
        summaries = []
        
        # Process in batches
        for i in range(0, len(contracts), self.batch_size):
            batch = contracts[i:i + self.batch_size]
            batch_summaries = await self._process_batch(query, batch)
            summaries.extend(batch_summaries)
            logger.info(f"Processed {min(i + self.batch_size, len(contracts))}/{len(contracts)} contracts")
        
        return summaries
    
    async def _process_batch(
        self,
        query: str,
        contracts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Process a batch of contracts concurrently"""
        tasks = [self._process_single_contract(query, contract) for contract in contracts]
        results = await asyncio.gather(*tasks)
        return results

    async def _process_single_contract(
        self,
        query: str,
        contract: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process a single contract"""
        contract_id = contract.get('id', '')
        
        # Search for relevant content in this contract
        # Run synchronous search in thread pool to avoid blocking
        search_results = await asyncio.to_thread(
            self.search_engine.search,
            query=query,
            top_k=3,
            contract_id=contract_id,
            include_parent_context=True
        )
        
        if search_results:
            # Combine relevant chunks
            content_parts = []
            for r in search_results:
                content = r.parent_content or r.content
                content_parts.append(content)
            
            contract_content = "\n\n".join(content_parts)
            
            # Generate summary for this contract
            summary = await self._summarize_contract(query, contract, contract_content)
        else:
            summary = f"Không tìm thấy nội dung liên quan trong hợp đồng {contract_id}"
        
        return {
            'contract_id': contract_id,
            'contract_number': contract.get('contract_number'),
            'partner_name': contract.get('partner_name'),
            'sign_date': contract.get('sign_date'),
            'total_value': contract.get('total_value'),
            'summary': summary
        }
    
    async def _summarize_contract(
        self,
        query: str,
        contract: Dict[str, Any],
        content: str
    ) -> str:
        """Generate summary for a single contract"""
        if not self.llm:
            return f"Tóm tắt tự động không khả dụng. Nội dung: {content[:300]}..."
        
        # Limit content length to avoid token limits
        max_content_length = 4000
        if len(content) > max_content_length:
            content = content[:max_content_length] + "..."
        
        prompt = MAP_SUMMARIZE_PROMPT.format(
            query=query,
            contract_content=f"Hợp đồng: {contract.get('contract_number', 'N/A')}\n"
                           f"Đối tác: {contract.get('partner_name', 'N/A')}\n"
                           f"Ngày ký: {contract.get('sign_date', 'N/A')}\n"
                           f"Giá trị: {contract.get('total_value', 'N/A'):,.0f} VNĐ\n\n"
                           f"Nội dung:\n{content}"
        )
        
        try:
            response = await self.llm.acomplete(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error summarizing contract: {e}")
            return f"Lỗi khi tóm tắt: {e}"
    
    async def _reduce_phase(
        self,
        query: str,
        contract_summaries: List[Dict[str, Any]]
    ) -> str:
        """
        REDUCE phase: Synthesize individual summaries into final report.
        """
        if not self.llm:
            # Fallback: simple concatenation
            parts = []
            for s in contract_summaries:
                parts.append(f"**{s['contract_id']}** ({s['partner_name']}): {s['summary']}")
            return "## Tóm tắt các hợp đồng\n\n" + "\n\n".join(parts)
        
        # Build summaries text - include ALL contracts
        summaries_text = ""
        for i, s in enumerate(contract_summaries, 1):
            summaries_text += f"""
### Hợp đồng {i}: {s.get('contract_number', s['contract_id'])}
- Đối tác: {s.get('partner_name', 'N/A')}
- Ngày ký: {s.get('sign_date', 'N/A')}
- Giá trị: {s.get('total_value', 0):,.0f} VNĐ
- Tóm tắt: {s['summary'][:500]}

"""
        
        # Increased limit to handle more contracts (up to ~20 contracts)
        max_length = 16000
        if len(summaries_text) > max_length:
            # Truncate individual summaries instead of cutting contracts
            summaries_text = ""
            for i, s in enumerate(contract_summaries, 1):
                summary_short = s['summary'][:200] + "..." if len(s['summary']) > 200 else s['summary']
                summaries_text += f"""
### Hợp đồng {i}: {s.get('contract_number', s['contract_id'])}
- Đối tác: {s.get('partner_name', 'N/A')}
- Giá trị: {s.get('total_value', 0):,.0f} VNĐ
- Tóm tắt: {summary_short}

"""
        
        prompt = REDUCE_SUMMARIZE_PROMPT.format(
            query=query,
            summaries=summaries_text
        )
        
        try:
            response = await self.llm.acomplete(prompt)
            result = response.text.strip()
            
            # Enrich with default suggestions if missing
            if "Gợi ý" not in result and "câu hỏi tiếp theo" not in result.lower():
                result += """

---
**Gợi ý câu hỏi tiếp theo:**
1. Xem chi tiết danh sách hợp đồng
2. Phân tích xu hướng theo thời gian
3. Thống kê theo đối tác"""
            
            return result
        except Exception as e:
            logger.error(f"Error in reduce phase: {e}")
            return f"Lỗi khi tổng hợp: {e}\n\n{summaries_text[:2000]}"
    
    async def get_statistics(
        self,
        partner_name: Optional[str] = None,
        year: Optional[int] = None,
        contract_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get aggregate statistics without LLM (direct from metadata).
        """
        contracts = self.metadata_store.search_contracts(
            partner_name=partner_name,
            year=year,
            contract_type=contract_type,
            limit=1000
        )
        
        if not contracts:
            return {'total_contracts': 0, 'total_value': 0}
        
        total_value = sum(c.get('total_value', 0) or 0 for c in contracts)
        
        # Group by partner
        by_partner = {}
        for c in contracts:
            partner = c.get('partner_name', 'Unknown')
            if partner not in by_partner:
                by_partner[partner] = {'count': 0, 'value': 0}
            by_partner[partner]['count'] += 1
            by_partner[partner]['value'] += c.get('total_value', 0) or 0
        
        return {
            'total_contracts': len(contracts),
            'total_value': total_value,
            'by_partner': by_partner,
            'contracts': [
                {
                    'id': c['id'],
                    'number': c.get('contract_number'),
                    'partner': c.get('partner_name'),
                    'value': c.get('total_value'),
                    'date': c.get('sign_date')
                }
                for c in contracts
            ]
        }
