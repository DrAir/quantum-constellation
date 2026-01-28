# Contract RAG System - Hệ thống Quản lý Hợp đồng Thông minh

Hệ thống quản lý và truy xuất thông tin từ hợp đồng sử dụng công nghệ **Advanced RAG + Agentic Workflow**, được tối ưu hóa cho việc xử lý văn bản pháp lý tiếng Việt và trích xuất dữ liệu có cấu trúc.

## ✨ Tính năng Nổi bật

- 🔍 **Truy xuất ngữ nghĩa (Semantic Search)** - Tìm kiếm thông tin chính xác > 95% nhờ Hybrid Search (Vector + BM25).
- 📊 **Tổng hợp đa văn bản (Multi-doc)** - Tự động tổng hợp dữ liệu từ hàng chục hợp đồng cùng lúc.
- � **Trích xuất Action Items (Mới)** - Tự động nhận diện Timeline, Deadline, Mốc thanh toán và Nghĩa vụ từ văn bản hợp đồng.
- 🛡️ **Cơ chế Fallback Thông minh** - Tự động chuyển đổi giữa tìm kiếm cụ thể và tìm kiếm toàn cục để đảm bảo luôn có kết quả.
- 🎯 **Advanced Reranking** - Tích hợp BAAI/bge-reranker-v2-m3 (có chế độ bypass khi mạng yếu).
- ⚡ **Hiệu năng cao** - Sử dụng Qdrant cho Vector Store và SQLite cho Metadata quản lý hàng triệu bản ghi.

## 🏗️ Kiến trúc Hệ thống

```mermaid
graph TD
    Client[Web Interface / API] --> Router[Query Router AI]
    
    Router -->|Hỏi cụ thể| Single[Single-hop Workflow]
    Router -->|Tổng hợp| Multi[Multi-doc Workflow]
    Router -->|Tiến độ/Deadline| Action[Action Extraction Workflow]
    Router -->|Thống kê| Meta[Metadata Query]
    
    subgraph Core Engine
        Single & Multi & Action --> Search[Hybrid Search Engine]
        Search -->|Vector| Qdrant[Qdrant DB]
        Search -->|Keyword| BM25[BM25 Sparse]
        Search -->|Filter| SQLite[SQLite Metadata]
        
        Search --> Rerank[Reranker Model]
        Rerank --> LLM[LLM Generator (Ollama)]
    end
    
    LLM --> Response[Final Answer]
```

## 🚀 Cài đặt & Triển khai

### 1. Yêu cầu hệ thống
- Python 3.10+
- Docker & Docker Compose
- RAM: Tối thiểu 16GB (để chạy LLM local)

### 2. Cài đặt dependencies
```bash
cd quantum-constellation
pip install -r requirements.txt
```

### 3. Khởi động Infrastructure
```bash
# Khởi động Qdrant Vector DB
docker-compose up -d
```

### 4. Cài đặt LLM (Ollama)
```bash
# Tải model (Khuyến nghị gpt-oss:20b hoặc Qwen2.5-14b cho tiếng Việt tốt nhất)
ollama pull gpt-oss:20b
ollama serve
```

### 5. Chạy ứng dụng
```bash
# Khởi động API Server
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

## 📁 Cấu trúc Project

```
quantum-constellation/
├── config/
│   ├── settings.py           # Cấu hình Global & Env vars
│   └── prompts.py            # System Prompts tối ưu cho RAG
├── data/
│   ├── contracts.db          # SQLite Database
│   ├── raw/                  # Thư mục chứa file gốc
│   └── processed/            # File đã xử lý (Markdown)
├── src/
│   ├── data_pipeline/        # Pipeline xử lý dữ liệu đầu vào
│   ├── storage/              # Kết nối Qdrant & SQLite
│   ├── retrieval/            # Logic tìm kiếm (Hybrid + Rerank)
│   ├── workflow/             # Các luồng xử lý chính
│   │   ├── query_router.py   # Phân loại câu hỏi
│   │   ├── single_hop.py     # Hỏi đáp thông thường
│   │   ├── multi_doc.py      # Tổng hợp nhiều văn bản
│   │   └── action_item.py    # Trích xuất nhiệm vụ/tiến độ (NEW)
│   └── api/                  # FastAPI Endpoints
└── static/                   # Giao diện Web (Chat UI)
```

## 📖 Hướng dẫn sử dụng

### 1. Truy cập
- **Web Chat**: http://localhost:8000
- **API Swagger**: http://localhost:8000/docs

### 2. Các loại câu hỏi hỗ trợ

#### 🔹 Hỏi đáp chi tiết (Single-hop)
> "Điều kiện thanh toán tạm ứng của hợp đồng 126/2025 là gì?"
> "Quy định về bảo hành trong hợp đồng mua sắm máy in?"

#### 🔹 Trích xuất tiến độ (Action Items)
> "Các mốc thực hiện của hợp đồng 126/2025/CHKNB-HĐMB"
> "Liệt kê deadline giao hàng và nghiệm thu của công ty Bầu Trời Việt"

#### 🔹 Tổng hợp thông tin (Multi-doc)
> "Tổng giá trị các hợp đồng đã ký với đối tác Elcom trong năm 2024?"
> "Tóm tắt các điều khoản phạt chậm tiến độ của tất cả hợp đồng CNTT."

## 🔧 Cơ chế Debus & Logging

Hệ thống có tích hợp sẵn các công cụ debug trong thư mục `data/`:
- `error.log`: Ghi nhận chi tiết lỗi Runtime (Stacktrace).
- `debug_context.txt`: Kiểm tra nội dung văn bản được gửi vào LLM.
- `debug_id.txt`: Kiểm tra Contract ID đã được resolve.
- `debug_fallback.txt`: Ghi nhận khi hệ thống kích hoạt chế độ Fallback Search.

## 🤝 Đóng góp
Sử dụng `pytest` để chạy kiểm thử trước khi commit:
```bash
pytest tests/ -v
```

## 📄 License
MIT License
