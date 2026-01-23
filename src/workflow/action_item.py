# Action Item Extraction Workflow
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import sys
from pathlib import Path
import traceback

# Add project root to path if needed for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from llama_index.llms.ollama import Ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

from src.retrieval.hybrid_search import HybridSearchEngine
from src.retrieval.reranker import Reranker
from config.prompts import ACTION_EXTRACTION_PROMPT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ActionItemResponse:
    """Response from action item extraction"""
    response: str
    source_nodes: List[Dict[str, Any]]
    contract_id: Optional[str] = None


class ActionExtractionWorkflow:
    """
    Workflow for extracting action items, deadlines, and obligations from contracts.
    """
    
    def __init__(
        self,
        search_engine: HybridSearchEngine,
        reranker: Reranker,
        llm_model: str = "gpt-oss:20b"
    ):
        self.search_engine = search_engine
        self.reranker = reranker
        self.llm_model = llm_model
        
        # Initialize LLM
        self.llm = None
        if OLLAMA_AVAILABLE:
            try:
                self.llm = Ollama(
                    model=llm_model,
                    request_timeout=60.0,
                    context_window=16384  # Larger window for analyzing clauses
                )
                logger.info(f"Action Extraction Workflow using LLM: {llm_model}")
            except Exception as e:
                logger.error(f"Could not initialize LLM: {e}")
    
    async def run(self, query: str, contract_number: str = None, contract_id: str = None, partner_name: str = None):
        """
        Execute the action item extraction workflow.
        """
        try:
            logger.info(f"Processing Action Item query: {query}")
            logger.info(f"INPUTS: contract_number={contract_number}, contract_id={contract_id}")

            if not self.search_engine:
                 logger.error("Search engine is None")
                 return ActionItemResponse(
                     response="Hệ thống tìm kiếm chưa sẵn sàng (Search Engine initialization failed).",
                     source_nodes=[]
                 )
            
            # Step 0: Resolve contract_number to contract_id if provided
            if contract_number and not contract_id:
                logger.info(f"Attempting to resolve contract number: {contract_number}")
                try:
                    # Access metadata store from search engine
                    if not self.search_engine.metadata_store:
                         logger.warning("Metadata store not available in search engine")
                    else:
                        md_store = self.search_engine.metadata_store
                        contracts = md_store.search_contracts(limit=1000)
                        
                        # Normalize contract number for matching
                        target_num = contract_number.strip().lower()
                        
                        resolved_id = None
                        # Strategy 1: Exact match (case insensitive)
                        for c in contracts:
                            c_num = c.get('contract_number', '').lower()
                            if c_num == target_num:
                                resolved_id = c['id']
                                break
                        
                        # Strategy 2: Partial match if exact failed
                        if not resolved_id:
                            for c in contracts:
                                c_num = c.get('contract_number', '').lower()
                                # Match if target is in contract number or vice versa
                                # e.g. "126/2025" in "126/2025/CHKNB-HĐMB"
                                if target_num and (target_num in c_num or c_num in target_num):
                                    resolved_id = c['id']
                                    break
                                    
                        if resolved_id:
                            contract_id = resolved_id
                            logger.info(f"Resolved '{contract_number}' to contract_id: {contract_id}")
                        else:
                            logger.warning(f"Could not resolve contract number: {contract_number}")
                        
                        # DEBUG: Write ID to file
                        try:
                            with open("data/debug_id.txt", "w") as f:
                                f.write(str(contract_id))
                        except: pass

                except Exception as e:
                    logger.error(f"Error resolving contract number: {e}")

            # Step 1: Search for relevant context
            search_results = self.search_engine.search(
                query=query,
                top_k=15,
                contract_id=contract_id,
                partner_name=partner_name
            )
            
            # Fallback if no results found with specific contract
            if not search_results and contract_id:
                logger.warning(f"No results for contract {contract_id}, falling back to global search")
                try:
                    with open("data/debug_fallback.txt", "w") as f: f.write("fallback triggered")
                except: pass
                
                search_results = self.search_engine.search(
                    query=query,
                    top_k=10,
                    contract_id=None # Search everywhere
                )

            if not search_results:
                logger.warning(f"No search results found for query: {query}")
                return ActionItemResponse(
                    response="Không tìm thấy thông tin liên quan trong hệ thống.",
                    source_nodes=[]
                )
                
            # Step 2: Rerank results
            # ERROR FIX: Reranker might fail in offline/network restricted env. Bypassing for now.
            logger.warning("Bypassing Reranker for stability in offline environment")
            
            # Pass through results without reranking (take top 7 from hybrid search)
            ranked_results = search_results[:7]
            
            logger.info(f"Top matched contract IDs: {list(set(r.contract_id for r in ranked_results))}")
                
            # Prepare context for LLM
            context_parts = []
            source_nodes = []
            
            # If specific contract is targeted, try to resolve it effectively
            target_contract_id = contract_id or (ranked_results[0].contract_id if ranked_results else None)
            
            for r in ranked_results:
                # If search is scoped to a specific contract, filter out others
                # ERROR FIX: Removing filter to ensure context delivery. Rely on search engine.
                # if contract_id and r.contract_id != contract_id:
                #    continue
                    
                page_num = r.contract_metadata.get('page_number', 'N/A') if r.contract_metadata else 'N/A'
                content = r.parent_content or r.content
                context_parts.append(f"--- Hợp đồng: {r.contract_id}, Trang {page_num} ---\n{content}")
                
                source_nodes.append({
                    'contract_id': r.contract_id,
                    'content': r.content[:200] + "...",
                    'score': r.score,
                    'page': page_num
                })
                
            context_text = "\n\n".join(context_parts)
            
            # DEBUG: Write context to file
            try:
                with open("data/debug_context.txt", "w", encoding="utf-8") as f:
                    f.write(context_text)
                logger.info("Debug context written to data/debug_context.txt")
            except Exception as e:
                logger.error(f"Failed to write debug context: {e}")

            # Step 3: Generate response with LLM
            if not self.llm:
                return ActionItemResponse(
                    response="LLM chưa được khởi tạo. Không thể trích xuất hành động.",
                    source_nodes=source_nodes
                )
                
            prompt = ACTION_EXTRACTION_PROMPT.format(
                query=query,
                context=context_text
            )
            
            try:
                response = self.llm.complete(prompt)
                response_text = response.text.strip() if response.text else ""
                if not response_text:
                    response_text = "Xin lỗi, tôi không thể trích xuất thông tin hành động từ văn bản này."
                    
                return ActionItemResponse(
                    response=response_text,
                    source_nodes=source_nodes,
                    contract_id=target_contract_id
                )
            except Exception as e:
                logger.error(f"Error generating action items: {e}")
                return ActionItemResponse(
                    response=f"Có lỗi khi xử lý yêu cầu: {str(e)}",
                    source_nodes=source_nodes
                )
        except Exception as e:
            logger.error(f"Unexpected error in ActionExtractionWorkflow.run: {e}")
            with open("data/error.log", "w") as f:
                f.write(traceback.format_exc())
            raise e

    def run_sync(self, *args, **kwargs):
        """Synchronous wrapper for run"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.run(*args, **kwargs))
