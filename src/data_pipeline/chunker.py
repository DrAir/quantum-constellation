# Hierarchical Chunker for Vietnamese Contracts
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import logging
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Represents a chunk of text from a contract"""
    chunk_id: str
    content: str
    chunk_type: str  # 'parent' or 'child'
    parent_id: Optional[str] = None
    
    # Metadata
    contract_id: str = ""
    article_number: Optional[int] = None
    article_title: Optional[str] = None
    section_number: Optional[str] = None
    
    # Position info
    start_char: int = 0
    end_char: int = 0
    
    # For embedding
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'chunk_id': self.chunk_id,
            'content': self.content,
            'chunk_type': self.chunk_type,
            'parent_id': self.parent_id,
            'contract_id': self.contract_id,
            'article_number': self.article_number,
            'article_title': self.article_title,
            'section_number': self.section_number,
            'start_char': self.start_char,
            'end_char': self.end_char,
        }


class HierarchicalChunker:
    """
    Implements hierarchical chunking for Vietnamese contracts.
    
    Strategy:
    - Parent Chunk: Full content of a "Điều" (Article)
    - Child Chunk: Individual sections within an Article (e.g., 1.1, 1.2)
    
    This allows:
    - Precise retrieval via child chunks
    - Full context via parent chunks
    """
    
    def __init__(
        self,
        min_chunk_size: int = 100,
        max_chunk_size: int = 2000,
        chunk_overlap: int = 50
    ):
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Patterns for Vietnamese contracts
        self.patterns = {
            # Match "Điều X" or "Điều X." or "Điều X:"
            'article': re.compile(
                r'^[\s]*(Điều\s+(\d+)[.:]?\s*([^\n\r]*))',
                re.IGNORECASE | re.MULTILINE
            ),
            # Match sections like "1.1", "1.2", "2.1" etc.
            'section': re.compile(
                r'^[\s]*(\d+\.\d+)[.:]?\s*',
                re.MULTILINE
            ),
            # Match sub-sections like "1.1.1", "1.1.2"
            'subsection': re.compile(
                r'^[\s]*(\d+\.\d+\.\d+)[.:]?\s*',
                re.MULTILINE
            ),
        }
    
    def chunk_document(self, content: str, contract_id: str) -> Tuple[List[Chunk], List[Chunk]]:
        """
        Chunk a contract document into parent and child chunks.
        
        Args:
            content: Full contract text
            contract_id: Unique identifier for the contract
            
        Returns:
            Tuple of (parent_chunks, child_chunks)
        """
        parent_chunks = []
        child_chunks = []
        
        # Find all articles
        articles = self._extract_articles(content)
        
        if not articles:
            # If no articles found, treat the whole document as one chunk
            logger.warning(f"No articles found in {contract_id}, treating as single chunk")
            parent_chunk = self._create_chunk(
                content=content,
                contract_id=contract_id,
                chunk_type='parent',
                article_number=0,
                article_title="Full Document"
            )
            parent_chunks.append(parent_chunk)
            return parent_chunks, child_chunks
        
        for article in articles:
            article_num = article['number']
            article_title = article['title']
            article_content = article['content']
            start_pos = article['start']
            end_pos = article['end']
            
            # Create parent chunk (full article)
            parent_chunk = self._create_chunk(
                content=article_content,
                contract_id=contract_id,
                chunk_type='parent',
                article_number=article_num,
                article_title=article_title,
                start_char=start_pos,
                end_char=end_pos
            )
            parent_chunks.append(parent_chunk)
            
            # Extract child chunks (sections within article)
            sections = self._extract_sections(article_content, article_num)
            
            for section in sections:
                child_chunk = self._create_chunk(
                    content=section['content'],
                    contract_id=contract_id,
                    chunk_type='child',
                    parent_id=parent_chunk.chunk_id,
                    article_number=article_num,
                    article_title=article_title,
                    section_number=section['number'],
                    start_char=start_pos + section['start'],
                    end_char=start_pos + section['end']
                )
                child_chunks.append(child_chunk)
        
        logger.info(
            f"Contract {contract_id}: {len(parent_chunks)} parent chunks, "
            f"{len(child_chunks)} child chunks"
        )
        
        return parent_chunks, child_chunks
    
    def chunk_file(self, file_path: Path) -> Tuple[List[Chunk], List[Chunk]]:
        """Chunk a contract file"""
        content = self._read_file(file_path)
        contract_id = file_path.stem
        return self.chunk_document(content, contract_id)
    
    def chunk_directory(self, dir_path: Path) -> Tuple[List[Chunk], List[Chunk]]:
        """Chunk all contract files in a directory"""
        all_parent_chunks = []
        all_child_chunks = []
        
        for file_path in list(dir_path.glob("*.txt")) + list(dir_path.glob("*.md")):
            try:
                parents, children = self.chunk_file(file_path)
                all_parent_chunks.extend(parents)
                all_child_chunks.extend(children)
            except Exception as e:
                logger.error(f"Error chunking {file_path.name}: {e}")
        
        logger.info(
            f"Total: {len(all_parent_chunks)} parent chunks, "
            f"{len(all_child_chunks)} child chunks"
        )
        
        return all_parent_chunks, all_child_chunks
    
    def _extract_articles(self, content: str) -> List[Dict[str, Any]]:
        """Extract all articles (Điều) from content"""
        articles = []
        
        # Find all article matches
        matches = list(self.patterns['article'].finditer(content))
        
        for i, match in enumerate(matches):
            article_num = int(match.group(2))
            article_title = match.group(3).strip() if match.group(3) else ""
            
            start_pos = match.start()
            
            # End position is start of next article or end of document
            if i + 1 < len(matches):
                end_pos = matches[i + 1].start()
            else:
                end_pos = len(content)
            
            article_content = content[start_pos:end_pos].strip()
            
            # Skip if too short
            if len(article_content) < self.min_chunk_size:
                continue
            
            articles.append({
                'number': article_num,
                'title': article_title,
                'content': article_content,
                'start': start_pos,
                'end': end_pos
            })
        
        return articles
    
    def _extract_sections(self, article_content: str, article_num: int) -> List[Dict[str, Any]]:
        """Extract sections within an article"""
        sections = []
        
        # Find all section matches
        matches = list(self.patterns['section'].finditer(article_content))
        
        if not matches:
            # No sections found, don't create child chunks
            return sections
        
        for i, match in enumerate(matches):
            section_num = match.group(1)
            
            start_pos = match.start()
            
            # End position is start of next section or end of article
            if i + 1 < len(matches):
                end_pos = matches[i + 1].start()
            else:
                end_pos = len(article_content)
            
            section_content = article_content[start_pos:end_pos].strip()
            
            # Skip if too short
            if len(section_content) < self.min_chunk_size // 2:
                continue
            
            # If section is too long, split it
            if len(section_content) > self.max_chunk_size:
                sub_sections = self._split_long_section(section_content, section_num)
                for j, sub in enumerate(sub_sections):
                    sections.append({
                        'number': f"{section_num}.{j+1}",
                        'content': sub['content'],
                        'start': start_pos + sub['start'],
                        'end': start_pos + sub['end']
                    })
            else:
                sections.append({
                    'number': section_num,
                    'content': section_content,
                    'start': start_pos,
                    'end': end_pos
                })
        
        return sections
    
    def _split_long_section(self, content: str, section_num: str) -> List[Dict[str, Any]]:
        """Split a long section into smaller chunks with overlap"""
        chunks = []
        
        # Try to split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', content)
        
        current_chunk = ""
        current_start = 0
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > self.max_chunk_size:
                if current_chunk:
                    chunks.append({
                        'content': current_chunk.strip(),
                        'start': current_start,
                        'end': current_start + len(current_chunk)
                    })
                    # Overlap: keep last part of current chunk
                    overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else ""
                    current_start = current_start + len(current_chunk) - len(overlap_text)
                    current_chunk = overlap_text
            
            current_chunk += " " + sentence
        
        # Add remaining content
        if current_chunk.strip():
            chunks.append({
                'content': current_chunk.strip(),
                'start': current_start,
                'end': current_start + len(current_chunk)
            })
        
        return chunks
    
    def _create_chunk(
        self,
        content: str,
        contract_id: str,
        chunk_type: str,
        article_number: Optional[int] = None,
        article_title: Optional[str] = None,
        section_number: Optional[str] = None,
        parent_id: Optional[str] = None,
        start_char: int = 0,
        end_char: int = 0
    ) -> Chunk:
        """Create a Chunk object with a unique ID"""
        # Generate unique chunk ID
        id_string = f"{contract_id}_{article_number}_{section_number}_{start_char}"
        chunk_id = hashlib.md5(id_string.encode()).hexdigest()[:12]
        
        return Chunk(
            chunk_id=chunk_id,
            content=content,
            chunk_type=chunk_type,
            parent_id=parent_id,
            contract_id=contract_id,
            article_number=article_number,
            article_title=article_title,
            section_number=section_number,
            start_char=start_char,
            end_char=end_char
        )
    
    def _read_file(self, file_path: Path) -> str:
        """Read file with fallback encodings"""
        encodings = ['utf-8', 'utf-16', 'cp1252', 'latin-1']
        
        for encoding in encodings:
            try:
                return file_path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
        
        raise ValueError(f"Could not decode file: {file_path}")


# CLI usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python chunker.py <input_file_or_dir>")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    chunker = HierarchicalChunker()
    
    if input_path.is_file():
        parents, children = chunker.chunk_file(input_path)
    else:
        parents, children = chunker.chunk_directory(input_path)
    
    print(f"\n=== Results ===")
    print(f"Parent chunks: {len(parents)}")
    print(f"Child chunks: {len(children)}")
    
    if parents:
        print(f"\nSample parent chunk (Article):")
        print(f"  ID: {parents[0].chunk_id}")
        print(f"  Article: Điều {parents[0].article_number}")
        print(f"  Content preview: {parents[0].content[:200]}...")
    
    if children:
        print(f"\nSample child chunk (Section):")
        print(f"  ID: {children[0].chunk_id}")
        print(f"  Parent ID: {children[0].parent_id}")
        print(f"  Section: {children[0].section_number}")
        print(f"  Content preview: {children[0].content[:150]}...")
