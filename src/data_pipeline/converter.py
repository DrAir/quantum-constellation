# TXT to Markdown Converter for Vietnamese Contracts
import re
from pathlib import Path
from typing import List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContractConverter:
    """
    Converts raw TXT contract files to structured Markdown format.
    Preserves and enhances document structure for better LLM understanding.
    """
    
    def __init__(self):
        # Patterns for Vietnamese contracts
        self.patterns = {
            # Main article headers (Điều 1, Điều 2, etc.)
            'article': re.compile(r'^[\s]*Điều\s+(\d+)[.:]\s*(.*)$', re.IGNORECASE | re.MULTILINE),
            
            # Sub-sections (1.1, 1.2, 2.1, etc.)
            'subsection': re.compile(r'^[\s]*(\d+\.\d+)[.:]?\s*(.*)$', re.MULTILINE),
            
            # Deep sub-sections (1.1.1, 1.1.2, etc.)
            'sub_subsection': re.compile(r'^[\s]*(\d+\.\d+\.\d+)[.:]?\s*(.*)$', re.MULTILINE),
            
            # List items with letters (a., b., c., etc.)
            'list_letter': re.compile(r'^[\s]*([a-z])[.)]\s*(.*)$', re.MULTILINE),
            
            # List items with dashes
            'list_dash': re.compile(r'^[\s]*[-+•]\s*(.*)$', re.MULTILINE),
            
            # Party headers (BÊN A, BÊN B)
            'party': re.compile(r'^[\s]*(BÊN\s+[AB])[:.]?\s*(.*)$', re.IGNORECASE | re.MULTILINE),
            
            # Contract title patterns
            'title': re.compile(r'(HỢP ĐỒNG.*?)(?=\n)', re.IGNORECASE),
            
            # Package/Item headers
            'package': re.compile(r'^[\s]*(Gói thầu|Hạng mục)[.:]\s*(.*)$', re.IGNORECASE | re.MULTILINE),
            
            # Contract number
            'contract_number': re.compile(r'Số[.:]\s*(\S+)', re.IGNORECASE),
        }
    
    def convert_file(self, input_path: Path, output_path: Path = None) -> str:
        """
        Convert a single TXT file to Markdown format.
        
        Args:
            input_path: Path to input TXT file
            output_path: Optional path for output MD file
            
        Returns:
            Converted markdown content
        """
        # Read the file with appropriate encoding
        content = self._read_file(input_path)
        
        # Convert to markdown
        markdown_content = self._convert_to_markdown(content)
        
        # Write output if path provided
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(markdown_content, encoding='utf-8')
            logger.info(f"Converted: {input_path.name} -> {output_path.name}")
        
        return markdown_content
    
    def convert_directory(self, input_dir: Path, output_dir: Path) -> List[Tuple[Path, Path]]:
        """
        Convert all TXT files in a directory to Markdown.
        
        Args:
            input_dir: Directory containing TXT files
            output_dir: Directory for output MD files
            
        Returns:
            List of (input_path, output_path) tuples for converted files
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        converted_files = []
        
        for txt_file in input_dir.glob("*.txt"):
            output_file = output_dir / f"{txt_file.stem}.md"
            self.convert_file(txt_file, output_file)
            converted_files.append((txt_file, output_file))
        
        logger.info(f"Converted {len(converted_files)} files")
        return converted_files
    
    def _read_file(self, file_path: Path) -> str:
        """Read file with fallback encodings"""
        encodings = ['utf-8', 'utf-16', 'cp1252', 'latin-1']
        
        for encoding in encodings:
            try:
                return file_path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
        
        raise ValueError(f"Could not decode file: {file_path}")
    
    def _convert_to_markdown(self, content: str) -> str:
        """
        Convert contract content to structured Markdown.
        """
        lines = content.split('\n')
        markdown_lines = []
        
        # Track state
        in_article = False
        current_article = None
        
        for i, line in enumerate(lines):
            converted_line = self._convert_line(line, i == 0)
            markdown_lines.append(converted_line)
        
        # Join and clean up
        markdown_content = '\n'.join(markdown_lines)
        
        # Clean up excessive whitespace
        markdown_content = re.sub(r'\n{4,}', '\n\n\n', markdown_content)
        
        # Add metadata header
        markdown_content = self._add_metadata_header(markdown_content)
        
        return markdown_content
    
    def _convert_line(self, line: str, is_first: bool = False) -> str:
        """Convert a single line to markdown format"""
        original_line = line
        line = line.strip()
        
        if not line:
            return ''
        
        # Check for contract title (usually at the beginning)
        if 'HỢP ĐỒNG' in line.upper() and len(line) < 200:
            return f"# {line}"
        
        # Check for party headers
        party_match = self.patterns['party'].match(line)
        if party_match:
            party = party_match.group(1).upper()
            rest = party_match.group(2) if party_match.group(2) else ""
            return f"### {party}: {rest}" if rest else f"### {party}"
        
        # Check for main articles (Điều)
        article_match = self.patterns['article'].match(line)
        if article_match:
            article_num = article_match.group(1)
            article_title = article_match.group(2) if article_match.group(2) else ""
            return f"## Điều {article_num}. {article_title}"
        
        # Check for sub-subsections (x.x.x) first (more specific)
        sub_sub_match = self.patterns['sub_subsection'].match(line)
        if sub_sub_match:
            section_num = sub_sub_match.group(1)
            section_content = sub_sub_match.group(2) if sub_sub_match.group(2) else ""
            return f"#### {section_num} {section_content}"
        
        # Check for subsections (x.x)
        sub_match = self.patterns['subsection'].match(line)
        if sub_match:
            section_num = sub_match.group(1)
            section_content = sub_match.group(2) if sub_match.group(2) else ""
            return f"### {section_num} {section_content}"
        
        # Check for letter lists
        letter_match = self.patterns['list_letter'].match(line)
        if letter_match:
            letter = letter_match.group(1)
            content = letter_match.group(2)
            return f"   {letter}. {content}"
        
        # Check for dash lists
        dash_match = self.patterns['list_dash'].match(line)
        if dash_match:
            content = dash_match.group(1)
            return f"- {content}"
        
        # Check for package/item headers
        package_match = self.patterns['package'].match(line)
        if package_match:
            package_type = package_match.group(1)
            package_content = package_match.group(2)
            return f"**{package_type}:** {package_content}"
        
        # Regular text - preserve as is
        return line
    
    def _add_metadata_header(self, content: str) -> str:
        """Add YAML frontmatter for metadata"""
        # Extract contract number if present
        contract_num_match = self.patterns['contract_number'].search(content)
        contract_num = contract_num_match.group(1) if contract_num_match else "N/A"
        
        header = f"""---
type: contract
contract_number: "{contract_num}"
processed: true
---

"""
        return header + content


# CLI usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python converter.py <input_dir> <output_dir>")
        sys.exit(1)
    
    input_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    
    converter = ContractConverter()
    converted = converter.convert_directory(input_dir, output_dir)
    print(f"Converted {len(converted)} files")
