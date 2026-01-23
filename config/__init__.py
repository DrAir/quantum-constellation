# Config package
from .settings import settings, Settings, ensure_directories
from .prompts import (
    METADATA_EXTRACTION_PROMPT,
    SINGLE_HOP_QUERY_PROMPT,
    MAP_SUMMARIZE_PROMPT,
    REDUCE_SUMMARIZE_PROMPT,
    QUERY_ROUTER_PROMPT,
    CLAUSE_EXTRACTION_PROMPT
)

__all__ = [
    'settings',
    'Settings',
    'ensure_directories',
    'METADATA_EXTRACTION_PROMPT',
    'SINGLE_HOP_QUERY_PROMPT',
    'MAP_SUMMARIZE_PROMPT',
    'REDUCE_SUMMARIZE_PROMPT',
    'QUERY_ROUTER_PROMPT',
    'CLAUSE_EXTRACTION_PROMPT'
]
