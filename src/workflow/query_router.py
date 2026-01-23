# Query Router for Contract RAG
from enum import Enum
from typing import Optional, Tuple
import re
import logging
import sys
from pathlib import Path
import unicodedata  # Moved import to top level

# Fix import path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.prompts import QUERY_ROUTER_PROMPT

try:
    from llama_index.llms.ollama import Ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of queries the system can handle"""
    SINGLE_HOP = "single_hop"      # Query about a specific contract/clause
    MULTI_DOC = "multi_doc"        # Query requiring aggregation across contracts
    METADATA = "metadata"          # Query about contract metadata only
    ACTION_ITEM = "action_item"    # Query about obligations, deadlines, and todos
    UNKNOWN = "unknown"


class QueryRouter:
    """
    Routes queries to appropriate workflow based on query analysis.
    Uses pattern matching first, then LLM for ambiguous cases.
    """
    
    def __init__(self, llm_model: str = "gpt-oss:7b", use_llm: bool = True):
        self.use_llm = use_llm
        self.llm = None
        
        if use_llm and OLLAMA_AVAILABLE:
            try:
                self.llm = Ollama(model=llm_model, request_timeout=30.0)
                logger.info(f"Query router using LLM: {llm_model}")
            except Exception as e:
                logger.warning(f"Could not initialize LLM: {e}")
                self.use_llm = False
        
        # Patterns for different query types
        self.patterns = {
            QueryType.ACTION_ITEM: [
                # Deadlines and timeline
                re.compile(r'(hạn|thời hạn|mốc|tiến độ|lịch)\s+(chót|hoàn thành|thanh toán|giao hàng|thực hiện|nghiệm thu|bắt đầu|kết thúc)', re.IGNORECASE),
                re.compile(r'bao giờ|khi nào\s+(phải|cần|thì|được|bắt đầu|kết thúc)', re.IGNORECASE),
                re.compile(r'(thời gian|ngày)\s+(thực hiện|nghiệm thu|bàn giao)', re.IGNORECASE),
                
                # Obligations and responsibilities
                re.compile(r'(trách nhiệm|nghĩa vụ|nhiệm vụ|công việc)\s+(của|thuộc về|phải làm)', re.IGNORECASE),
                re.compile(r'các mốc thực hiện', re.IGNORECASE),
                re.compile(r'quy định về thời gian', re.IGNORECASE),
                re.compile(r'(phải|cần)\s+(làm gì|thực hiện|tuân thủ)', re.IGNORECASE),
                re.compile(r'bên\s+[AB]\s+(phải|cần|có trách nhiệm)', re.IGNORECASE),
                re.compile(r'các bước\s+(thực hiện|nghiệm thu|thanh toán)', re.IGNORECASE),
            ],
            QueryType.MULTI_DOC: [
                # Aggregation keywords
                re.compile(r'tổng\s+(hợp|giá trị|cộng|số)', re.IGNORECASE),
                re.compile(r'thống kê', re.IGNORECASE),
                re.compile(r'so sánh', re.IGNORECASE),
                re.compile(r'liệt kê\s+(các|tất cả)', re.IGNORECASE),
                re.compile(r'bao nhiêu hợp đồng', re.IGNORECASE),
                re.compile(r'các hợp đồng', re.IGNORECASE),
                re.compile(r'tất cả (các )?hợp đồng', re.IGNORECASE),
                # Time-based aggregation
                re.compile(r'năm\s+\d{4}', re.IGNORECASE),
                re.compile(r'trong\s+(tháng|năm|quý)', re.IGNORECASE),
            ],
            QueryType.METADATA: [
                # Pure metadata queries
                re.compile(r'danh sách (hợp đồng|đối tác)', re.IGNORECASE),
                re.compile(r'có bao nhiêu', re.IGNORECASE),
                re.compile(r'đối tác nào', re.IGNORECASE),
                re.compile(r'công ty nào', re.IGNORECASE),
            ],
            QueryType.SINGLE_HOP: [
                # Specific contract references
                re.compile(r'hợp đồng\s+(số\s+)?[\d/]+', re.IGNORECASE),
                re.compile(r'điều\s+\d+', re.IGNORECASE),
                re.compile(r'khoản\s+\d+', re.IGNORECASE),
                # Specific questions
                re.compile(r'điều kiện (phạt|bảo hành|thanh toán)', re.IGNORECASE),
                re.compile(r'thời hạn (bảo hành)', re.IGNORECASE),
                re.compile(r'nội dung (của )?hợp đồng', re.IGNORECASE),
            ]
        }
    
    def route(self, query: str) -> Tuple[QueryType, dict]:
        """
        Route a query to the appropriate workflow.
        
        Args:
            query: User's natural language query
            
        Returns:
            Tuple of (QueryType, extracted_params dict)
        """
        # Normalize unicode to NFC
        query = unicodedata.normalize('NFC', query)
        
        # Sanitize hyphens global check
        query = re.sub(r'[\u2010-\u2015\u2212\uFE58\uFE63\uFF0D]', '-', query)

        # Log routing decision
        logger.info(f"Routing query: {query}")
        
        # Extract parameters from query
        params = self._extract_params(query)
        
        # Try pattern matching first
        query_type = self._pattern_match(query)
        
        if query_type != QueryType.UNKNOWN:
            logger.info(f"Routed via pattern matching: {query_type.value}")
            return query_type, params
        
        # Use LLM for ambiguous cases
        if self.use_llm and self.llm:
            query_type = self._llm_classify(query)
            logger.info(f"Routed via LLM: {query_type.value}")
            return query_type, params
        
        # Default to single-hop for unknown queries
        logger.info("Unknown query type, defaulting to single-hop")
        return QueryType.SINGLE_HOP, params

    def _pattern_match(self, query: str) -> QueryType:
        """Match query against known patterns"""
        # Check Action Item first (very specific intent)
        for pattern in self.patterns.get(QueryType.ACTION_ITEM, []):
            if pattern.search(query):
                return QueryType.ACTION_ITEM
        
        # Check multi-doc patterns
        for pattern in self.patterns[QueryType.MULTI_DOC]:
            if pattern.search(query):
                return QueryType.MULTI_DOC
        
        # Check metadata patterns
        for pattern in self.patterns[QueryType.METADATA]:
            if pattern.search(query):
                return QueryType.METADATA
        
        # Check single-hop patterns
        for pattern in self.patterns[QueryType.SINGLE_HOP]:
            if pattern.search(query):
                return QueryType.SINGLE_HOP
        
        return QueryType.UNKNOWN
    
    def _extract_params(self, query: str) -> dict:
        """Extract structured parameters from query"""
        params = {}
        
        # Extract contract number (support alphanumeric and various separators)
        contract_num_match = re.search(
            r'hợp đồng\s+(số\s+)?([\w\.\/\-]+)', query, re.IGNORECASE
        )
        if contract_num_match:
            params['contract_number'] = contract_num_match.group(2)
        
        # Extract year
        year_match = re.search(r'năm\s+(\d{4})', query, re.IGNORECASE)
        if year_match:
            params['year'] = int(year_match.group(1))
        
        # Extract partner/company name
        partner_match = re.search(
            r'(công ty|đối tác)\s+([A-Za-zÀ-ỹ\s]+?)(?:\s+(?:năm|trong|từ|đến)|$)',
            query, re.IGNORECASE
        )
        if partner_match:
            params['partner_name'] = partner_match.group(2).strip()
        
        # Extract article number
        article_match = re.search(r'điều\s+(\d+)', query, re.IGNORECASE)
        if article_match:
            params['article_number'] = int(article_match.group(1))
        
        return params
    
    def _llm_classify(self, query: str) -> QueryType:
        """Use LLM to classify query type"""
        if not self.llm:
            return QueryType.UNKNOWN
        
        prompt = QUERY_ROUTER_PROMPT.format(question=query)

        try:
            response = self.llm.complete(prompt)
            response_text = response.text.strip().upper()
            
            if "ACTION" in response_text:
                return QueryType.ACTION_ITEM
            elif "SINGLE" in response_text:
                return QueryType.SINGLE_HOP
            elif "MULTI" in response_text:
                return QueryType.MULTI_DOC
            elif "METADATA" in response_text:
                return QueryType.METADATA
        except Exception as e:
            logger.error(f"LLM classification error: {e}")
        
        return QueryType.UNKNOWN


# Test
if __name__ == "__main__":
    router = QueryRouter(use_llm=False)  # Pattern matching only for testing
    
    test_queries = [
        "Điều kiện phạt của Hợp đồng số 112/2024 là gì?",
        "Tổng giá trị các hợp đồng năm 2024 là bao nhiêu?",
        "Danh sách các đối tác đã ký hợp đồng",
        "Thời hạn bảo hành trong hợp đồng với công ty Elcom?",
        "So sánh điều khoản thanh toán giữa các hợp đồng mua sắm",
        "Tôi phải làm gì để được thanh toán đợt 2 của hợp đồng 112?",
        "Các mốc thanh toán trong hợp đồng 456 là khi nào?",
    ]
    
    for query in test_queries:
        query_type, params = router.route(query)
        print(f"\\nQuery: {query}")
        print(f"  Type: {query_type.value}")
        print(f"  Params: {params}")
