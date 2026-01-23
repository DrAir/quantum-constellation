import sys
from pathlib import Path
import sqlite3
import re
import unicodedata

# Setup path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# DB Path
DB_PATH = PROJECT_ROOT / "data" / "contracts.db"

def inspect_db(target_number_part="126/2025"):
    print(f"--- Inspecting DB for '{target_number_part}' ---")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, contract_number FROM contracts")
    rows = cursor.fetchall()
    conn.close()
    
    found = False
    for r_id, r_num in rows:
        if target_number_part in r_num:
            print(f"FOUND: ID='{r_id}', Number='{r_num}'")
            # Analyze characters in number
            print(f"Hex dump of Number: {[hex(ord(c)) for c in r_num]}")
            found = True
    
    if not found:
        print("NOT FOUND in DB")

def test_regex(query):
    print(f"\n--- Testing Regex on query: '{query}' ---")
    # Using the regex from query_router.py
    normalized_query = unicodedata.normalize('NFC', query)
    print(f"Normalized query: {normalized_query}")
    
    # Updated regex to include alphanumeric and common separators more broadly
    # The original regex was: r'hợp đồng\s+(số\s+)?([\d/\-]+)'
    pattern = re.compile(r'hợp đồng\s+(số\s+)?([A-Za-z0-9\-\/\\\.]+)', re.IGNORECASE)
    
    match = pattern.search(normalized_query)
    if match:
        extracted = match.group(2)
        print(f"Extracted: '{extracted}'")
        print(f"Hex dump: {[hex(ord(c)) for c in extracted]}")
    else:
        print("NO MATCH for contract number")

if __name__ == "__main__":
    query_str = "các mốc thực hiện của hợp đồng 126/2025/CHKNB‑HĐMB"
    inspect_db()
    test_regex(query_str)
