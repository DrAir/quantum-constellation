# Contract RAG System - Hệ thống Quản lý Hợp đồng Thông minh

Hệ thống quản lý và truy xuất thông tin từ hợp đồng sử dụng công nghệ **Advanced RAG + Agentic Workflow**.

## ✨ Tính năng

- 🔍 **Truy xuất chính xác** - Tìm kiếm thông tin chi tiết từ từng điều khoản (độ chính xác > 95%)
- 📊 **Tổng hợp thông minh** - Phân tích và báo cáo từ nhiều hợp đồng cùng lúc
- 🚀 **Hybrid Search** - Kết hợp Vector Search + Keyword Search (BM25)
- 🎯 **Reranking** - Sử dụng BGE-Reranker-v2-m3 để tăng độ chính xác
- 🌐 **Web Interface** - Giao diện chat trực quan

## 🏗️ Kiến trúc

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Web Interface │────▶│   FastAPI        │────▶│  Query Router   │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                        ┌─────────────────────────────────┼─────────────────────────────────┐
                        │                                 │                                 │
                        ▼                                 ▼                                 ▼
               ┌────────────────┐              ┌────────────────┐              ┌────────────────┐
               │  Single-hop    │              │   Multi-doc    │              │   Metadata     │
               │   Workflow     │              │   Workflow     │              │    Query       │
               └───────┬────────┘              └───────┬────────┘              └───────┬────────┘
                       │                               │                               │
                       ▼                               ▼                               ▼
               ┌────────────────┐              ┌────────────────┐              ┌────────────────┐
               │ Hybrid Search  │              │  Map-Reduce    │              │    SQLite      │
               │  + Reranker    │              │  Summarization │              │   Metadata     │
               └───────┬────────┘              └───────┬────────┘              └────────────────┘
                       │                               │
                       ▼                               ▼
               ┌────────────────────────────────────────────┐
               │              Qdrant Vector DB              │
               │         (BGE-M3 Dense + Sparse)            │
               └────────────────────────────────────────────┘
```

## 🚀 Bắt đầu

### 1. Cài đặt dependencies

```bash
cd quantum-constellation
pip install -r requirements.txt
```

### 2. Khởi động Qdrant (Vector Database)

```bash
docker-compose up -d
```

### 3. Khởi động Ollama (nếu chưa có)

```bash
# Cài đặt Ollama từ https://ollama.ai
ollama pull gpt-oss:20b
ollama serve
```

### 4. Chạy ứng dụng

```bash
# Chạy API server
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Truy cập

- **Web Interface**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## 📁 Cấu trúc dự án

```
quantum-constellation/
├── config/
│   ├── settings.py          # Cấu hình hệ thống
│   └── prompts.py            # LLM prompts
├── data/
│   ├── raw/                  # File hợp đồng gốc
│   ├── processed/            # File đã xử lý (.md)
│   └── metadata/             # Metadata JSON
├── src/
│   ├── data_pipeline/
│   │   ├── converter.py      # TXT → MD converter
│   │   ├── extractor.py      # Metadata extraction
│   │   └── chunker.py        # Hierarchical chunking
│   ├── storage/
│   │   ├── vector_store.py   # Qdrant operations
│   │   └── metadata_store.py # SQLite operations
│   ├── retrieval/
│   │   ├── hybrid_search.py  # Vector + BM25 search
│   │   └── reranker.py       # BGE reranker
│   ├── workflow/
│   │   ├── query_router.py   # Query classification
│   │   ├── single_hop.py     # Single contract queries
│   │   └── multi_doc.py      # Multi-doc summarization
│   └── api/
│       └── main.py           # FastAPI application
├── static/
│   └── index.html            # Web interface
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## 📖 API Endpoints

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| POST | `/query` | Hỏi đáp về hợp đồng |
| POST | `/index` | Index hợp đồng mới |
| GET | `/contracts` | Danh sách hợp đồng |
| GET | `/contracts/{id}` | Chi tiết hợp đồng |
| GET | `/stats` | Thống kê tổng quan |
| GET | `/health` | Health check |

## 🔧 Cấu hình

Chỉnh sửa file `config/settings.py`:

```python
# LLM Model
llm_model = "gpt-oss:20b"  # hoặc model khác

# Qdrant
qdrant_host = "localhost"
qdrant_port = 6333

# Embedding
embedding_model = "BAAI/bge-m3"
```

## 📝 Ví dụ sử dụng

### Query đơn (Single-hop)
```
Điều kiện phạt của Hợp đồng số 112/2024 là gì?
```

### Query tổng hợp (Multi-doc)
```
Tổng giá trị các hợp đồng năm 2024 là bao nhiêu?
```

### Query thống kê
```
Danh sách các đối tác đã ký hợp đồng
```

## 🛠️ Phát triển

```bash
# Chạy tests
pytest tests/ -v

# Type checking
mypy src/
```

## 📄 License

MIT License
