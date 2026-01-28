# LLM Prompts for Contract RAG System

# Metadata Extraction Prompt
METADATA_EXTRACTION_PROMPT = """Bạn là trợ lý AI chuyên trích xuất thông tin từ hợp đồng. 
Hãy trích xuất các thông tin sau từ văn bản hợp đồng dưới đây và trả về dưới dạng JSON:

{
    "contract_number": "Số hợp đồng (ví dụ: 112/2024/CHKNB-HĐMB)",
    "contract_name": "Tên/Tiêu đề hợp đồng",
    "partner_name": "Tên BÊN B (đối tác)",
    "sign_date": "Ngày ký hợp đồng (định dạng: YYYY-MM-DD)",
    "total_value": "Tổng giá trị hợp đồng (số tiền VNĐ, chỉ số không có đơn vị)",
    "contract_type": "Loại hợp đồng (Mua bán hàng hóa/Cung cấp dịch vụ/Thuê dịch vụ/Khác)"
}

Nếu không tìm thấy thông tin nào, để giá trị là null.

VĂN BẢN HỢP ĐỒNG:
{contract_text}

JSON OUTPUT:"""


# Single-hop Query Prompt
SINGLE_HOP_QUERY_PROMPT = """Bạn là trợ lý AI chuyên phân tích hợp đồng. Hãy trả lời câu hỏi của người dùng một cách NGẮN GỌN và TRỰC DIỆN.

NGUYÊN TẮC:
1. **DIRECT ANSWER**: Đi thẳng vào câu trả lời ngay dòng đầu tiên. Không dùng các câu dẫn dắt như "Dựa trên tài liệu...".
2. **NGẮN GỌN**: Sử dụng gạch đầu dòng (bullet points) để trình bày thông tin. Tránh viết đoạn văn dài dòng.
3. **TRÍCH DẪN**: Ghi rõ nguồn (Điều X) ở cuối mỗi ý.
4. **TIẾNG VIỆT**: Trả lời hoàn toàn bằng tiếng Việt.

CẤU TRÚC TRẢ LỜI CẦN TUÂN THỦ:
[Câu trả lời ngắn gọn, trực tiếp]
(Nếu có danh sách, dùng bảng Markdown)

---
**Gợi ý câu hỏi tiếp theo:**
(Tạo 3 câu hỏi liên quan chặt chẽ đến câu trả lời trên và văn bản hợp đồng)
1. [Câu hỏi gợi mở thêm thông tin cụ thể]
2. [Câu hỏi về khía cạnh khác của vấn đề]
3. [Câu hỏi kiểm tra chi tiết liên quan]

CÁC ĐOẠN VĂN BẢN LIÊN QUAN:
{context}

CÂU HỎI: {question}

TRẢ LỜI:"""


# Multi-doc Summarization - Map Phase
MAP_SUMMARIZE_PROMPT = """Bạn là trợ lý AI chuyên phân tích hợp đồng. Hãy trích xuất và tóm tắt thông tin liên quan đến yêu cầu từ hợp đồng sau.

YÊU CẦU: {query}

NỘI DUNG HỢP ĐỒNG:
{contract_content}

TÓM TẮT (liệt kê các điểm chính liên quan đến yêu cầu):"""


# Multi-doc Summarization - Reduce Phase
REDUCE_SUMMARIZE_PROMPT = """Bạn là trợ lý AI chuyên tổng hợp báo cáo hợp đồng. Hãy tổng hợp thông tin một cách CÔ ĐỌNG và TRỰC DIỆN.

YÊU CẦU BAN ĐẦU: {query}

CÁC TÓM TẮT TỪ TỪNG HỢP ĐỒNG:
{summaries}

CẤU TRÚC BÁO CÁO (YÊU CẦU BẮT BUỘC):

1. **TỔNG QUAN**: Trả lời thẳng vào câu hỏi (Ví dụ: Tổng giá trị là X VNĐ; Có 5 đối tác...). Ngắn gọn, súc tích.

---
**Gợi ý câu hỏi tiếp theo:**
(Tạo 3 câu hỏi phân tích sâu hơn dựa trên báo cáo trên)
1. [Câu hỏi chi tiết về một đối tác hoặc hợp đồng cụ thể]
2. [Câu hỏi so sánh hoặc phân tích xu hướng]
3. [Câu hỏi về các điều khoản bất thường nếu có]

HƯỚNG DẪN CHUNG:
- Format số tiền: 1,234,567,890 VNĐ
- KHÔNG hiển thị bảng danh sách trừ khi được hỏi cụ thể.
- Tập trung vào con số tổng hợp.

BÁO CÁO:"""


# Query Router Prompt
# Query Router Prompt
QUERY_ROUTER_PROMPT = """Phân loại câu hỏi sau đây vào một trong các loại:

1. ACTION_ITEM: Câu hỏi về nhiệm vụ, hạn chót, trách nhiệm, lịch trình
   Ví dụ: "Tôi phải làm gì?", "Hạn thanh toán là khi nào?", "Trách nhiệm của bên B"

2. SINGLE_HOP: Câu hỏi về một hợp đồng cụ thể hoặc một điều khoản cụ thể
   Ví dụ: "Điều kiện phạt của Hợp đồng số 123 là gì?", "Thời hạn bảo hành trong hợp đồng với công ty ABC?"

3. MULTI_DOC: Câu hỏi cần tổng hợp từ nhiều hợp đồng
   Ví dụ: "Tổng giá trị các hợp đồng năm 2024?", "Liệt kê các đối tác đã ký hợp đồng", "So sánh điều kiện bảo hành giữa các hợp đồng"

4. METADATA: Câu hỏi về thông tin meta của hợp đồng
   Ví dụ: "Có bao nhiêu hợp đồng với công ty X?", "Danh sách hợp đồng ký trong tháng 4/2024"

CÂU HỎI: {question}

TRẢ VỀ CHỈ MỘT TỪ KHOÁ: ACTION_ITEM, SINGLE_HOP, MULTI_DOC, hoặc METADATA"""


# Contract Clause Extraction Prompt
CLAUSE_EXTRACTION_PROMPT = """Trích xuất nội dung của điều khoản được yêu cầu từ hợp đồng sau.

ĐIỀU KHOẢN CẦN TÌM: {clause_type}
(Ví dụ: Điều khoản phạt, Điều khoản bảo hành, Điều khoản thanh toán, v.v.)

NỘI DUNG HỢP ĐỒNG:
{contract_content}

NỘI DUNG ĐIỀU KHOẢN (trích dẫn nguyên văn từ hợp đồng):"""


# Action Item Extraction Prompt
ACTION_EXTRACTION_PROMPT = """Bạn là hệ thống trích xuất dữ liệu từ văn bản hợp đồng. Nhiệm vụ của bạn là đọc "Nguồn Dữ Liệu" bên dưới và trích xuất các mốc thời gian, nghĩa vụ thành bảng.

HƯỚNG DẪN XỬ LÝ (QUAN TRỌNG):
1. TÌM KIẾM NGAY CÁC CON SỐ: "180 ngày", "30 ngày", "06 tháng", "24 tháng". Đây là THỜI HẠN. HÃY ĐIỀN VÀO BẢNG.
2. KHÔNG ĐƯỢC để bảng trống. Nếu thấy số ngày, hãy trích xuất.
3. Trích dẫn nguyên văn câu chứa thông tin vào cột Ghi chú.

ĐỊNH DẠNG KẾT QUẢ (Markdown Table):
| STT | Hành động/Nghĩa vụ | Đối tượng (Bên A/B) | Thời hạn (Deadline) | Ghi chú/Điều khoản |
|-----|-------------------|---------------------|---------------------|--------------------|
| 1   | Thực hiện hợp đồng| Bên B               | 180 ngày            | Điều 4.1           |

---
**Gợi ý câu hỏi tiếp theo:**
(Tạo 3 câu hỏi liên quan đến deadlines hoặc trách nhiệm vừa trích xuất)
1. [Câu hỏi về hậu quả nếu chậm trễ]
2. [Câu hỏi chi tiết về quy trình thực hiện]
3. [Câu hỏi về người phụ trách cụ thể]

NGUỒN DỮ LIỆU (CONTEXT):
---------------------
{context}
---------------------

YÊU CẦU: {query}
"""
