import requests
import json
import sys

def test_api():
    url = "http://localhost:8000/query"
    payload = {
        "query": "các mốc thực hiện của hợp đồng 126/2025/CHKNB‑HĐMB"
    }
    headers = {"Content-Type": "application/json"}
    
    print(f"Sending Query: {payload['query']}")
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        
        data = response.json()
        answer = data.get("answer", "")
        
        print("\n=== API RESPONSE ===")
        print(answer)
        print("====================")
        
        if "180 ngày" in answer and "|" in answer:
            print("\n✅ SUCCESS: Found '180 ngày' and Markdown Table.")
        elif "Không tìm thấy" in answer:
            print("\n❌ FAILURE: Model rejected response.")
        else:
            print("\n⚠️ UNCERTAIN: Check output manually.")
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")

if __name__ == "__main__":
    test_api()
