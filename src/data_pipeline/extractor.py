# Metadata Extractor for Vietnamese Contracts
import re
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ContractMetadata:
    """Structured metadata for a contract"""
    contract_id: str  # Unique identifier (filename-based)
    contract_number: Optional[str] = None
    contract_name: Optional[str] = None
    partner_name: Optional[str] = None  # BÊN B
    party_a_name: Optional[str] = None  # BÊN A
    sign_date: Optional[str] = None
    total_value: Optional[float] = None
    total_value_text: Optional[str] = None
    contract_type: Optional[str] = None
    file_path: Optional[str] = None
    year: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MetadataExtractor:
    """
    Extracts structured metadata from Vietnamese contract documents.
    Uses regex patterns first, with optional LLM fallback for complex cases.
    """
    
    def __init__(self):
        self.patterns = {
            # Contract number patterns
            'contract_number': [
                re.compile(r'Số[.:]\s*([^\n\r]+)', re.IGNORECASE),
                re.compile(r'HỢP ĐỒNG\s+SỐ[.:]\s*([^\n\r]+)', re.IGNORECASE),
            ],
            
            # Contract name/title
            'contract_name': [
                re.compile(r'(HỢP ĐỒNG\s+[^\n\r]{5,100})', re.IGNORECASE),
                re.compile(r'Gói thầu[.:]\s*["""]?([^"""\n\r]+)["""]?', re.IGNORECASE),
            ],
            
            # Party B (partner)
            'partner_name': [
                re.compile(r'BÊN\s+B[.:]\s*([^\n\r]+)', re.IGNORECASE),
                re.compile(r'BÊN\s+B\s*\([^)]*\)[.:]\s*([^\n\r]+)', re.IGNORECASE),
            ],
            
            # Party A
            'party_a_name': [
                re.compile(r'BÊN\s+A[.:]\s*([^\n\r]+)', re.IGNORECASE),
                re.compile(r'BÊN\s+A\s*\([^)]*\)[.:]\s*([^\n\r]+)', re.IGNORECASE),
            ],
            
            # Sign date
            'sign_date': [
                re.compile(r'ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})', re.IGNORECASE),
                re.compile(r'Hôm nay,?\s+ngày\s+(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})', re.IGNORECASE),
                re.compile(r'(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})'),
            ],
            
            # Total value
            'total_value': [
                re.compile(r'(?:Tổng\s+)?[Gg]iá\s+trị\s+[Hh]ợp\s+đồng[^:]*[.:]\s*([\d.,]+)\s*(?:đồng|VNĐ|VND)?', re.IGNORECASE),
                re.compile(r'(?:Tổng\s+)?[Gg]iá\s+trị[^:]*[.:]\s*([\d.,]+)\s*(?:đồng|VNĐ|VND)?', re.IGNORECASE),
                re.compile(r'([\d.,]+)\s*(?:đồng|VNĐ|VND)', re.IGNORECASE),
            ],
            
            # Value in words (for verification)
            'total_value_text': [
                re.compile(r'[Bb]ằng\s+chữ[.:]\s*([^\n\r.]+)', re.IGNORECASE),
            ],
            
            # Contract type detection
            'contract_type_keywords': {
                'Mua bán hàng hóa': ['mua bán', 'mua sắm', 'cung cấp hàng', 'mua hàng'],
                'Cung cấp dịch vụ': ['cung cấp dịch vụ', 'dịch vụ'],
                'Thuê dịch vụ': ['thuê dịch vụ', 'thuê'],
                'Xây dựng': ['xây dựng', 'thi công', 'xây lắp'],
                'Tư vấn': ['tư vấn', 'consulting'],
                'Bảo trì': ['bảo trì', 'bảo dưỡng', 'sửa chữa'],
            }
        }
    
    def extract_from_file(self, file_path: Path) -> ContractMetadata:
        """Extract metadata from a contract file"""
        content = self._read_file(file_path)
        
        metadata = ContractMetadata(
            contract_id=file_path.stem,
            file_path=str(file_path)
        )
        
        # Extract each field
        metadata.contract_number = self._extract_contract_number(content)
        metadata.contract_name = self._extract_contract_name(content)
        metadata.partner_name = self._extract_partner_name(content)
        metadata.party_a_name = self._extract_party_a_name(content)
        
        sign_date = self._extract_sign_date(content)
        if sign_date:
            metadata.sign_date = sign_date
            try:
                metadata.year = int(sign_date.split('-')[0])
            except:
                pass
        
        value_result = self._extract_total_value(content)
        if value_result:
            metadata.total_value = value_result[0]
            metadata.total_value_text = value_result[1]
        
        metadata.contract_type = self._detect_contract_type(content, metadata.contract_name)
        
        # Try to extract year from filename if not in content
        if not metadata.year:
            year_match = re.search(r'_(\d{4})_', file_path.stem)
            if year_match:
                metadata.year = int(year_match.group(1))
        
        return metadata
    
    def extract_from_directory(self, dir_path: Path, output_file: Path = None) -> List[ContractMetadata]:
        """Extract metadata from all files in a directory"""
        all_metadata = []
        
        for file_path in dir_path.glob("*.txt"):
            try:
                metadata = self.extract_from_file(file_path)
                all_metadata.append(metadata)
                logger.info(f"Extracted metadata from: {file_path.name}")
            except Exception as e:
                logger.error(f"Error extracting from {file_path.name}: {e}")
        
        # Also check for .md files
        for file_path in dir_path.glob("*.md"):
            try:
                metadata = self.extract_from_file(file_path)
                all_metadata.append(metadata)
                logger.info(f"Extracted metadata from: {file_path.name}")
            except Exception as e:
                logger.error(f"Error extracting from {file_path.name}: {e}")
        
        # Save to JSON if output path provided
        if output_file:
            self._save_to_json(all_metadata, output_file)
        
        return all_metadata
    
    def _read_file(self, file_path: Path) -> str:
        """Read file with fallback encodings"""
        encodings = ['utf-8', 'utf-16', 'cp1252', 'latin-1']
        
        for encoding in encodings:
            try:
                return file_path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
        
        raise ValueError(f"Could not decode file: {file_path}")
    
    def _extract_contract_number(self, content: str) -> Optional[str]:
        """Extract contract number"""
        for pattern in self.patterns['contract_number']:
            match = pattern.search(content)
            if match:
                number = match.group(1).strip()
                # Clean up common artifacts
                number = re.sub(r'[\r\n].*', '', number)
                return number[:100]  # Limit length
        return None
    
    def _extract_contract_name(self, content: str) -> Optional[str]:
        """Extract contract name/title"""
        for pattern in self.patterns['contract_name']:
            match = pattern.search(content)
            if match:
                name = match.group(1).strip()
                # Clean up quotes and extra whitespace
                name = re.sub(r'["""]', '', name)
                name = re.sub(r'\s+', ' ', name)
                return name[:300]  # Limit length
        return None
    
    def _extract_partner_name(self, content: str) -> Optional[str]:
        """Extract BÊN B (partner) name - improved version"""
        # Look for company name patterns after BÊN B
        company_patterns = [
            # BÊN B: CÔNG TY...
            re.compile(r'BÊN\s+B[.:]\s*(CÔNG TY[^,\n\r]{10,150})', re.IGNORECASE),
            # BÊN B (Bên bán): CÔNG TY...
            re.compile(r'BÊN\s+B\s*\([^)]*\)[.:]\s*(CÔNG TY[^,\n\r]{10,150})', re.IGNORECASE),
            # - Tên đơn vị: CÔNG TY... (after BÊN B section)
            re.compile(r'Tên\s+(?:đơn vị|công ty|doanh nghiệp)[.:]\s*(CÔNG TY[^,\n\r]{10,150})', re.IGNORECASE),
        ]
        
        for pattern in company_patterns:
            match = pattern.search(content)
            if match:
                name = match.group(1).strip()
                name = self._clean_company_name(name)
                if len(name) > 10 and self._is_valid_company_name(name):
                    return name[:200]
        
        # Fallback: original patterns but with stricter validation
        for pattern in self.patterns['partner_name']:
            match = pattern.search(content)
            if match:
                name = match.group(1).strip()
                name = self._clean_company_name(name)
                if len(name) > 5 and self._is_valid_company_name(name):
                    return name[:200]
        return None
    
    def _clean_company_name(self, name: str) -> str:
        """Clean up company name by removing common suffixes and artifacts"""
        # Stop at common delimiters
        name = re.split(r'[,\n\r]', name)[0].strip()
        
        # Remove "(gọi tắt là bên B/A)" and similar
        name = re.sub(r'\s*\(gọi tắt[^)]*\)\s*', '', name, flags=re.IGNORECASE)
        name = re.sub(r'\s*\(sau đây[^)]*\)\s*', '', name, flags=re.IGNORECASE)
        name = re.sub(r'\s*\(bên [AB]\)\s*', '', name, flags=re.IGNORECASE)
        
        # Remove other parenthetical notes at the end
        name = re.sub(r'\s*\([^)]*\)\s*$', '', name)
        
        # Remove trailing punctuation
        name = name.rstrip('.,;: ')
        
        return name
    
    def _is_valid_company_name(self, name: str) -> bool:
        """Validate that extracted name looks like a company name"""
        # Must contain company-like keywords
        company_keywords = ['CÔNG TY', 'CTY', 'TNHH', 'CỔ PHẦN', 'TẬP ĐOÀN', 'DOANH NGHIỆP']
        name_upper = name.upper()
        has_company_keyword = any(kw in name_upper for kw in company_keywords)
        
        # Reject if starts with numbers (like "5.5. Trong vòng...")
        starts_with_number = bool(re.match(r'^\d+\.', name))
        
        # Reject if looks like article text
        article_keywords = ['trong vòng', 'điều', 'khoản', 'căn cứ', 'theo']
        looks_like_article = any(kw in name.lower() for kw in article_keywords)
        
        return has_company_keyword and not starts_with_number and not looks_like_article
    
    def _extract_party_a_name(self, content: str) -> Optional[str]:
        """Extract BÊN A name"""
        for pattern in self.patterns['party_a_name']:
            match = pattern.search(content)
            if match:
                name = match.group(1).strip()
                name = re.split(r'[-–—:]', name)[0].strip()
                name = re.sub(r'\s*\([^)]*\)\s*$', '', name)
                if len(name) > 5:
                    return name[:200]
        return None
    
    def _extract_sign_date(self, content: str) -> Optional[str]:
        """Extract signing date in YYYY-MM-DD format"""
        for pattern in self.patterns['sign_date']:
            match = pattern.search(content)
            if match:
                groups = match.groups()
                if len(groups) == 3:
                    day, month, year = groups
                    try:
                        day = int(day)
                        month = int(month)
                        year = int(year)
                        # Validate date
                        date_obj = datetime(year, month, day)
                        return date_obj.strftime('%Y-%m-%d')
                    except (ValueError, TypeError):
                        continue
        return None
    
    def _extract_total_value(self, content: str) -> Optional[tuple]:
        """Extract total contract value"""
        value = None
        value_text = None
        
        # Try to find value in numbers
        for pattern in self.patterns['total_value']:
            match = pattern.search(content)
            if match:
                value_str = match.group(1)
                # Clean up number formatting
                value_str = value_str.replace('.', '').replace(',', '')
                try:
                    value = float(value_str)
                    if value > 1000:  # Reasonable contract value
                        break
                except ValueError:
                    continue
        
        # Try to find value in words
        for pattern in self.patterns['total_value_text']:
            match = pattern.search(content)
            if match:
                value_text = match.group(1).strip()
                break
        
        if value or value_text:
            return (value, value_text)
        return None
    
    def _detect_contract_type(self, content: str, contract_name: Optional[str] = None) -> str:
        """Detect contract type from content keywords"""
        text_to_check = content.lower()
        if contract_name:
            text_to_check = contract_name.lower() + " " + text_to_check
        
        for contract_type, keywords in self.patterns['contract_type_keywords'].items():
            for keyword in keywords:
                if keyword in text_to_check:
                    return contract_type
        
        return "Khác"
    
    def _save_to_json(self, metadata_list: List[ContractMetadata], output_file: Path):
        """Save metadata list to JSON file"""
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        data = [m.to_dict() for m in metadata_list]
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(data)} metadata records to {output_file}")


# CLI usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python extractor.py <input_dir> [output_json]")
        sys.exit(1)
    
    input_dir = Path(sys.argv[1])
    output_file = Path(sys.argv[2]) if len(sys.argv) > 2 else input_dir / "metadata.json"
    
    extractor = MetadataExtractor()
    metadata = extractor.extract_from_directory(input_dir, output_file)
    
    for m in metadata:
        print(f"\n{m.contract_id}:")
        print(f"  Number: {m.contract_number}")
        print(f"  Partner: {m.partner_name}")
        print(f"  Value: {m.total_value:,.0f} VNĐ" if m.total_value else "  Value: N/A")
        print(f"  Date: {m.sign_date}")
        print(f"  Type: {m.contract_type}")
