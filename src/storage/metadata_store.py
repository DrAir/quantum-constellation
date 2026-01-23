# Metadata Store using SQLite for Contract RAG
import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from contextlib import contextmanager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetadataStore:
    """
    SQLite-based metadata store for contract filtering.
    Enables fast filtering before vector search.
    """
    
    def __init__(self, db_path: str = "data/contracts.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize the database schema"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Create contracts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS contracts (
                    id TEXT PRIMARY KEY,
                    contract_number TEXT,
                    contract_name TEXT,
                    partner_name TEXT,
                    party_a_name TEXT,
                    sign_date TEXT,
                    total_value REAL,
                    total_value_text TEXT,
                    contract_type TEXT,
                    file_path TEXT,
                    year INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for common queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_partner_name 
                ON contracts(partner_name)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_year 
                ON contracts(year)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_contract_type 
                ON contracts(contract_type)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_sign_date 
                ON contracts(sign_date)
            """)
            
            conn.commit()
            logger.info("Database initialized")
    
    @contextmanager
    def _get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def insert_contract(self, metadata: Dict[str, Any]) -> bool:
        """Insert or update a contract's metadata"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO contracts 
                    (id, contract_number, contract_name, partner_name, party_a_name,
                     sign_date, total_value, total_value_text, contract_type, 
                     file_path, year, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metadata.get('contract_id'),
                    metadata.get('contract_number'),
                    metadata.get('contract_name'),
                    metadata.get('partner_name'),
                    metadata.get('party_a_name'),
                    metadata.get('sign_date'),
                    metadata.get('total_value'),
                    metadata.get('total_value_text'),
                    metadata.get('contract_type'),
                    metadata.get('file_path'),
                    metadata.get('year'),
                    datetime.now().isoformat()
                ))
                conn.commit()
                return True
            except Exception as e:
                logger.error(f"Error inserting contract: {e}")
                return False
    
    def insert_many(self, metadata_list: List[Dict[str, Any]]) -> int:
        """Insert multiple contracts' metadata"""
        success_count = 0
        for metadata in metadata_list:
            if self.insert_contract(metadata):
                success_count += 1
        logger.info(f"Inserted {success_count}/{len(metadata_list)} contracts")
        return success_count
    
    def get_contract(self, contract_id: str) -> Optional[Dict[str, Any]]:
        """Get a single contract by ID"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM contracts WHERE id = ?", (contract_id,))
            row = cursor.fetchone()
            
            if row:
                return dict(row)
            return None
    
    def search_contracts(
        self,
        partner_name: Optional[str] = None,
        year: Optional[int] = None,
        contract_type: Optional[str] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        keyword: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Search contracts with various filters.
        
        Args:
            partner_name: Filter by partner name (partial match)
            year: Filter by year
            contract_type: Filter by contract type
            min_value: Minimum contract value
            max_value: Maximum contract value
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            keyword: Keyword search in contract name
            limit: Maximum results to return
            
        Returns:
            List of matching contracts
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM contracts WHERE 1=1"
            params = []
            
            if partner_name:
                query += " AND partner_name LIKE ?"
                params.append(f"%{partner_name}%")
            
            if year:
                query += " AND year = ?"
                params.append(year)
            
            if contract_type:
                query += " AND contract_type = ?"
                params.append(contract_type)
            
            if min_value is not None:
                query += " AND total_value >= ?"
                params.append(min_value)
            
            if max_value is not None:
                query += " AND total_value <= ?"
                params.append(max_value)
            
            if start_date:
                query += " AND sign_date >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND sign_date <= ?"
                params.append(end_date)
            
            if keyword:
                query += " AND contract_name LIKE ?"
                params.append(f"%{keyword}%")
            
            query += " ORDER BY sign_date DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            return [dict(row) for row in rows]
    
    def get_contract_ids(
        self,
        partner_name: Optional[str] = None,
        year: Optional[int] = None,
        contract_type: Optional[str] = None
    ) -> List[str]:
        """Get list of contract IDs matching filters (for vector search pre-filtering)"""
        contracts = self.search_contracts(
            partner_name=partner_name,
            year=year,
            contract_type=contract_type,
            limit=1000
        )
        return [c['id'] for c in contracts]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregate statistics about contracts"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            stats = {}
            
            # Total count
            cursor.execute("SELECT COUNT(*) as count FROM contracts")
            stats['total_contracts'] = cursor.fetchone()['count']
            
            # Total value
            cursor.execute("SELECT SUM(total_value) as total FROM contracts")
            result = cursor.fetchone()['total']
            stats['total_value'] = result if result else 0
            
            # By year
            cursor.execute("""
                SELECT year, COUNT(*) as count, SUM(total_value) as total
                FROM contracts
                WHERE year IS NOT NULL
                GROUP BY year
                ORDER BY year DESC
            """)
            stats['by_year'] = [dict(row) for row in cursor.fetchall()]
            
            # By type
            cursor.execute("""
                SELECT contract_type, COUNT(*) as count, SUM(total_value) as total
                FROM contracts
                GROUP BY contract_type
                ORDER BY count DESC
            """)
            stats['by_type'] = [dict(row) for row in cursor.fetchall()]
            
            # Top partners by value
            cursor.execute("""
                SELECT partner_name, COUNT(*) as count, SUM(total_value) as total
                FROM contracts
                WHERE partner_name IS NOT NULL
                GROUP BY partner_name
                ORDER BY total DESC
                LIMIT 10
            """)
            stats['top_partners'] = [dict(row) for row in cursor.fetchall()]
            
            return stats
    
    def get_all_partners(self) -> List[str]:
        """Get list of all unique partner names"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT DISTINCT partner_name 
                FROM contracts 
                WHERE partner_name IS NOT NULL
                ORDER BY partner_name
            """)
            return [row['partner_name'] for row in cursor.fetchall()]
    
    def get_all_years(self) -> List[int]:
        """Get list of all years with contracts"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT DISTINCT year 
                FROM contracts 
                WHERE year IS NOT NULL
                ORDER BY year DESC
            """)
            return [row['year'] for row in cursor.fetchall()]
    
    def delete_contract(self, contract_id: str) -> bool:
        """Delete a contract by ID"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM contracts WHERE id = ?", (contract_id,))
            conn.commit()
            return cursor.rowcount > 0
    
    def clear_all(self) -> int:
        """Delete all contracts (use with caution)"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM contracts")
            conn.commit()
            count = cursor.rowcount
            logger.info(f"Deleted {count} contracts")
            return count


# CLI usage
if __name__ == "__main__":
    import sys
    
    store = MetadataStore()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "stats":
            stats = store.get_statistics()
            print(json.dumps(stats, indent=2, ensure_ascii=False))
        elif sys.argv[1] == "search" and len(sys.argv) > 2:
            results = store.search_contracts(keyword=sys.argv[2])
            for r in results:
                print(f"{r['id']}: {r['contract_name'][:50]}... - {r['total_value']:,.0f} VNĐ")
    else:
        print("Usage:")
        print("  python metadata_store.py stats")
        print("  python metadata_store.py search <keyword>")
