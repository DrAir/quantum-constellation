# Contract RAG System Configuration
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# Base Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"

# Source Data Path (User's contract files)
SOURCE_CONTRACTS_DIR = Path("/Users/sonnguyen/Downloads/HDTXT")


@dataclass
class QdrantConfig:
    """Qdrant Vector Database Configuration"""
    host: str = "localhost"
    port: int = 6333
    collection_name: str = "contracts"
    vector_size: int = 1024  # BGE-M3 embedding size
    distance: str = "Cosine"


@dataclass
class EmbeddingConfig:
    """Embedding Model Configuration"""
    model_name: str = "BAAI/bge-m3"
    device: str = "cpu"  # or "cuda" if GPU available
    max_length: int = 8192  # BGE-M3 supports long context
    batch_size: int = 32


@dataclass
class RerankerConfig:
    """Reranker Model Configuration"""
    model_name: str = "BAAI/bge-reranker-v2-m3"
    device: str = "cpu"
    top_k: int = 10  # Initial retrieval count
    final_k: int = 3  # After reranking


@dataclass
class LLMConfig:
    """LLM Configuration for Ollama"""
    model_name: str = "gpt-oss:20b"  # Default, can be changed
    base_url: str = "http://localhost:11434"
    temperature: float = 0.1
    max_tokens: int = 4096
    context_window: int = 32768


@dataclass
class ChunkingConfig:
    """Hierarchical Chunking Configuration"""
    # Parent chunk: Full "Điều" (Article)
    # Child chunk: Individual clauses within an Article
    parent_separator: str = r"Điều\s+\d+"
    child_separator: str = r"\d+\.\d+"
    chunk_overlap: int = 100
    min_chunk_size: int = 100
    max_chunk_size: int = 2000


@dataclass
class MetadataFields:
    """Metadata fields to extract from contracts"""
    contract_id: str = "contract_id"
    contract_name: str = "contract_name"
    contract_number: str = "contract_number"
    partner_name: str = "partner_name"  # BÊN B
    sign_date: str = "sign_date"
    total_value: str = "total_value"
    contract_type: str = "contract_type"
    file_path: str = "file_path"


@dataclass
class Settings:
    """Main Settings Container"""
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    metadata_fields: MetadataFields = field(default_factory=MetadataFields)
    
    # Search settings
    hybrid_search_alpha: float = 0.5  # Balance between vector and keyword search
    
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000


# Global settings instance
settings = Settings()


def ensure_directories():
    """Create necessary directories if they don't exist"""
    for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, METADATA_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)


# Initialize directories on import
ensure_directories()
