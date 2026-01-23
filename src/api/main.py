# FastAPI Backend for Contract RAG System
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from pathlib import Path
import logging
import asyncio
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings, SOURCE_CONTRACTS_DIR, PROCESSED_DATA_DIR, METADATA_DIR
from src.data_pipeline.converter import ContractConverter
from src.data_pipeline.extractor import MetadataExtractor
from src.data_pipeline.chunker import HierarchicalChunker
from src.storage.vector_store import VectorStore
from src.storage.metadata_store import MetadataStore
from src.retrieval.hybrid_search import HybridSearchEngine
from src.retrieval.reranker import Reranker
from src.workflow.query_router import QueryRouter, QueryType
from src.workflow.single_hop import SingleHopWorkflow
from src.workflow.multi_doc import MultiDocWorkflow
from src.workflow.action_item import ActionExtractionWorkflow  # New Workflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Contract RAG System",
    description="Hệ thống Quản lý Hợp đồng Thông minh với Advanced RAG",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances (initialized on startup)
vector_store: Optional[VectorStore] = None
metadata_store: Optional[MetadataStore] = None
search_engine: Optional[HybridSearchEngine] = None
reranker: Optional[Reranker] = None
query_router: Optional[QueryRouter] = None
single_hop_workflow: Optional[SingleHopWorkflow] = None
multi_doc_workflow: Optional[MultiDocWorkflow] = None
action_extraction_workflow: Optional[ActionExtractionWorkflow] = None  # New Workflow


# Request/Response Models
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, description="Câu hỏi về hợp đồng")
    contract_id: Optional[str] = Field(None, description="ID hợp đồng cụ thể")
    partner_name: Optional[str] = Field(None, description="Lọc theo tên đối tác")
    year: Optional[int] = Field(None, description="Lọc theo năm")
    contract_type: Optional[str] = Field(None, description="Lọc theo loại hợp đồng")


class QueryResponse(BaseModel):
    answer: str
    query_type: str
    sources: List[Dict[str, Any]] = []
    contracts_analyzed: int = 0
    total_value: Optional[float] = None
    confidence: float = 0.0


class IndexRequest(BaseModel):
    source_dir: Optional[str] = Field(None, description="Đường dẫn đến thư mục chứa file hợp đồng")
    recreate: bool = Field(False, description="Xóa và tạo lại index")


class IndexResponse(BaseModel):
    success: bool
    message: str
    contracts_indexed: int = 0
    chunks_created: int = 0


class ContractListResponse(BaseModel):
    total: int
    contracts: List[Dict[str, Any]]


class StatsResponse(BaseModel):
    total_contracts: int
    total_value: float
    by_year: List[Dict[str, Any]]
    by_type: List[Dict[str, Any]]
    top_partners: List[Dict[str, Any]]


# Startup and Shutdown
@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    global vector_store, metadata_store, search_engine, reranker
    global query_router, single_hop_workflow, multi_doc_workflow, action_extraction_workflow
    
    logger.info("Initializing Contract RAG System...")
    logger.info("Applying patches for Query Router (Pattern Fix)...")  # Trigger Reload
    
    try:
        # Initialize stores
        metadata_store = MetadataStore(db_path=str(PROJECT_ROOT / "data" / "contracts.db"))
        
        # Try to initialize vector store (requires Qdrant to be running)
        try:
            vector_store = VectorStore(
                host=settings.qdrant.host,
                port=settings.qdrant.port,
                collection_name=settings.qdrant.collection_name
            )
            logger.info("Vector store initialized")
        except Exception as e:
            logger.warning(f"Vector store initialization failed (is Qdrant running?): {e}")
            vector_store = None
        
        # Initialize reranker
        try:
            reranker = Reranker()
            logger.info("Reranker initialized")
        except Exception as e:
            logger.warning(f"Reranker initialization failed: {e}")
            reranker = Reranker()  # Will use fallback
        
        # Initialize search engine if vector store available
        if vector_store:
            search_engine = HybridSearchEngine(
                vector_store=vector_store,
                metadata_store=metadata_store
            )
            
            # Initialize workflows
            single_hop_workflow = SingleHopWorkflow(
                search_engine=search_engine,
                reranker=reranker,
                llm_model=settings.llm.model_name
            )
            
            multi_doc_workflow = MultiDocWorkflow(
                search_engine=search_engine,
                metadata_store=metadata_store,
                llm_model=settings.llm.model_name
            )
            
            action_extraction_workflow = ActionExtractionWorkflow(  # Initialize new workflow
                search_engine=search_engine,
                reranker=reranker,
                llm_model=settings.llm.model_name
            )
        
        # Initialize query router
        # Use a smaller/faster model for routing if available, or the main model
        router_model = settings.llm.model_name
        # If we have a dedicated router model in settings, use it (optional enhancement)
        # For now, we'll use the configured model but enable LLM
        query_router = QueryRouter(
            use_llm=True,
            llm_model=router_model
        )
        
        logger.info("Contract RAG System initialized successfully!")
        
    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise


# API Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "vector_store": vector_store is not None,
        "metadata_store": metadata_store is not None,
        "search_engine": search_engine is not None
    }


@app.post("/query", response_model=QueryResponse)
async def query_contracts(request: QueryRequest):
    """
    Main query endpoint - automatically routes to appropriate workflow.
    """
    if not query_router:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    # Route query
    query_type, params = query_router.route(request.query)
    
    # Merge request params with extracted params
    contract_id = request.contract_id or params.get('contract_number')
    partner_name = request.partner_name or params.get('partner_name')
    year = request.year or params.get('year')
    
    try:
        if query_type == QueryType.METADATA:
            # Direct metadata query
            if not metadata_store:
                raise HTTPException(status_code=503, detail="Metadata store not available")
            
            contracts = metadata_store.search_contracts(
                partner_name=partner_name,
                year=year,
                contract_type=request.contract_type,
                limit=100
            )
            
            return QueryResponse(
                answer=f"Tìm thấy {len(contracts)} hợp đồng.",
                query_type="metadata",
                sources=[{"contract_id": c['id'], "partner": c.get('partner_name')} for c in contracts[:10]],
                contracts_analyzed=len(contracts),
                total_value=sum(c.get('total_value', 0) or 0 for c in contracts)
            )
        
        elif query_type == QueryType.MULTI_DOC:
            # Multi-document summarization
            if not multi_doc_workflow:
                raise HTTPException(status_code=503, detail="Multi-doc workflow not available")
            
            response = await multi_doc_workflow.run(
                query=request.query,
                partner_name=partner_name,
                year=year,
                contract_type=request.contract_type
            )
            
            return QueryResponse(
                answer=response.summary,
                query_type="multi_doc",
                sources=response.contract_summaries,
                contracts_analyzed=response.contracts_analyzed,
                total_value=response.total_value
            )
            
        elif query_type == QueryType.ACTION_ITEM:
            # Action Item Extraction
            if not action_extraction_workflow:
                raise HTTPException(status_code=503, detail="Action extraction workflow not available")
                
            response = await action_extraction_workflow.run(
                query=request.query,
                contract_id=contract_id,
                contract_number=contract_id, # Use contract_id as number if available
                partner_name=partner_name
            )
            
            return QueryResponse(
                answer=response.response,
                query_type="action_item",
                sources=response.source_nodes,
                confidence=0.85 # Placeholder confidence
            )
        
        else:
            # Single-hop query (default)
            if not single_hop_workflow:
                raise HTTPException(status_code=503, detail="Single-hop workflow not available")
            
            response = await single_hop_workflow.run(
                query=request.query,
                contract_id=contract_id,
                partner_name=partner_name,
                year=year
            )
            
            return QueryResponse(
                answer=response.answer,
                query_type="single_hop",
                sources=response.sources,
                confidence=response.confidence
            )
    
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index", response_model=IndexResponse)
async def index_contracts(request: IndexRequest, background_tasks: BackgroundTasks):
    """
    Index contracts from source directory.
    """
    source_dir = Path(request.source_dir) if request.source_dir else SOURCE_CONTRACTS_DIR
    
    if not source_dir.exists():
        raise HTTPException(status_code=400, detail=f"Directory not found: {source_dir}")
    
    # Run indexing in background
    background_tasks.add_task(run_indexing, source_dir, request.recreate)
    
    return IndexResponse(
        success=True,
        message=f"Đang xử lý các file từ {source_dir}. Kiểm tra logs để theo dõi tiến độ."
    )


async def run_indexing(source_dir: Path, recreate: bool):
    """Background task to run indexing"""
    global vector_store, metadata_store
    
    try:
        logger.info(f"Starting indexing from {source_dir}")
        
        # Step 1: Convert TXT to MD
        converter = ContractConverter()
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        converted = converter.convert_directory(source_dir, PROCESSED_DATA_DIR)
        logger.info(f"Converted {len(converted)} files to Markdown")
        
        # Step 2: Extract metadata
        extractor = MetadataExtractor()
        METADATA_DIR.mkdir(parents=True, exist_ok=True)
        metadata_list = extractor.extract_from_directory(
            source_dir,
            METADATA_DIR / "contracts.json"
        )
        logger.info(f"Extracted metadata from {len(metadata_list)} contracts")
        
        # Step 3: Store metadata in SQLite
        if metadata_store:
            metadata_store.insert_many([m.to_dict() for m in metadata_list])
        
        # Step 4: Chunk documents
        chunker = HierarchicalChunker()
        parent_chunks, child_chunks = chunker.chunk_directory(source_dir)
        logger.info(f"Created {len(parent_chunks)} parent chunks and {len(child_chunks)} child chunks")
        
        # Step 5: Index in vector store
        if vector_store:
            if recreate:
                vector_store.create_collection(recreate=True)
            
            # Index all chunks
            all_chunks = [c.to_dict() for c in parent_chunks + child_chunks]
            vector_store.index_chunks(all_chunks)
            logger.info(f"Indexed {len(all_chunks)} chunks in vector store")
        
        logger.info("Indexing completed successfully!")
        
    except Exception as e:
        logger.error(f"Indexing error: {e}")


@app.get("/contracts", response_model=ContractListResponse)
async def list_contracts(
    partner: Optional[str] = Query(None, description="Filter by partner name"),
    year: Optional[int] = Query(None, description="Filter by year"),
    contract_type: Optional[str] = Query(None, description="Filter by type"),
    limit: int = Query(50, ge=1, le=200)
):
    """List contracts with optional filters"""
    if not metadata_store:
        raise HTTPException(status_code=503, detail="Metadata store not available")
    
    contracts = metadata_store.search_contracts(
        partner_name=partner,
        year=year,
        contract_type=contract_type,
        limit=limit
    )
    
    return ContractListResponse(
        total=len(contracts),
        contracts=contracts
    )


@app.get("/contracts/{contract_id}")
async def get_contract(contract_id: str):
    """Get a specific contract by ID"""
    if not metadata_store:
        raise HTTPException(status_code=503, detail="Metadata store not available")
    
    contract = metadata_store.get_contract(contract_id)
    if not contract:
        raise HTTPException(status_code=404, detail="Contract not found")
    
    return contract


@app.get("/stats", response_model=StatsResponse)
async def get_statistics():
    """Get aggregate statistics about contracts"""
    if not metadata_store:
        raise HTTPException(status_code=503, detail="Metadata store not available")
    
    stats = metadata_store.get_statistics()
    return StatsResponse(**stats)


@app.get("/partners")
async def get_partners():
    """Get list of all partners"""
    if not metadata_store:
        raise HTTPException(status_code=503, detail="Metadata store not available")
    
    return {"partners": metadata_store.get_all_partners()}


@app.get("/years")
async def get_years():
    """Get list of all years"""
    if not metadata_store:
        raise HTTPException(status_code=503, detail="Metadata store not available")
    
    return {"years": metadata_store.get_all_years()}


# Serve static files for web interface
STATIC_DIR = PROJECT_ROOT / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the web interface"""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse("<h1>Contract RAG System API</h1><p>Web interface not found. Use /docs for API documentation.</p>")


# Run with: uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
