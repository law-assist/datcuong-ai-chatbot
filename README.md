# AI Chatbot — Legal Question Answering Service

Microservice chatbot tư vấn pháp lý sử dụng RAG (Retrieval-Augmented Generation) với LangChain, LangGraph, Ollama và ChromaDB.

---

## Mục lục

- [Tổng quan dự án](#tổng-quan-dự-án)
- [Tech Stack](#tech-stack)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Hướng dẫn chạy local](#hướng-dẫn-chạy-local)
- [Biến môi trường](#biến-môi-trường)
- [API Endpoints](#api-endpoints)
- [Ghi chú bổ sung](#ghi-chú-bổ-sung)

---

## Tổng quan dự án

Service này cung cấp khả năng hỏi đáp pháp lý tự động bằng tiếng Việt. Người dùng đặt câu hỏi pháp lý và hệ thống trả lời dựa trên nội dung các văn bản pháp luật đã được lập chỉ mục trong vector database (ChromaDB).

**Chức năng chính:**

- **Hỏi đáp pháp lý (RAG):** Nhận câu hỏi từ người dùng, tìm kiếm các đoạn văn bản pháp luật liên quan nhất trong ChromaDB, sau đó dùng LLM (Llama 3 chạy local qua Ollama) tổng hợp câu trả lời bằng tiếng Việt.
- **Pipeline LangGraph:** Luồng xử lý được tổ chức thành đồ thị trạng thái với 4 bước tuần tự: dịch truy vấn → phân tích truy vấn → truy xuất tài liệu → sinh câu trả lời.
- **Vector search:** Dùng embedding model `all-MiniLM-L6-v2` (HuggingFace) để mã hóa câu hỏi và tìm kiếm similarity trong ChromaDB.
- **Tiền xử lý dữ liệu:** Module `data_processing.py` chuyển đổi cây nội dung văn bản từ MongoDB thành văn bản phẳng để đưa vào ChromaDB.

**Kiến trúc xử lý (RAG pipeline):**

```
Câu hỏi người dùng
        │
        ▼
[1] query_translation   ← hiện tại: pass-through (chưa dịch)
        │
        ▼
[2] query_analysis      ← hiện tại: pass-through (chưa phân tích)
        │
        ▼
[3] retrieve            ← ChromaDB similarity_search, k=5
        │   vector store: ./database/all-MiniLM-L6-v2/3000_300
        │   embedding: all-MiniLM-L6-v2
        ▼
[4] generate            ← Llama 3 (Ollama local)
        │   prompt: "Bạn là một luật sư giàu kinh nghiệm..."
        ▼
Câu trả lời tiếng Việt + 5 đoạn context
```

---

## Tech Stack

### Runtime & Framework

| Công nghệ | Mục đích |
|---|---|
| Python 3.x | Runtime |
| FastAPI | REST API framework |
| Uvicorn | ASGI server (port 28080) |

### AI / LLM

| Công nghệ | Mục đích |
|---|---|
| LangChain | Orchestration framework (PromptTemplate, ChatPromptTemplate) |
| LangGraph | Pipeline dạng đồ thị trạng thái (StateGraph) |
| LangChain-Ollama | Kết nối với Ollama local LLM |
| Ollama + Llama 3 | LLM chạy local (`llama3`) |
| LangChain-HuggingFace | Embedding model từ HuggingFace |
| `all-MiniLM-L6-v2` | Sentence embedding model (384 chiều) |
| LangSmith | Observability / tracing (tùy chọn) |

### Vector Database

| Công nghệ | Mục đích |
|---|---|
| ChromaDB | Vector store lưu embedding văn bản pháp luật |
| LangChain-Chroma | Wrapper ChromaDB cho LangChain |

### Database

| Công nghệ | Mục đích |
|---|---|
| MongoDB | Nguồn dữ liệu văn bản pháp luật gốc |
| PyMongo | MongoDB client |

### Tiện ích

| Công nghệ | Mục đích |
|---|---|
| python-dotenv | Đọc biến môi trường từ `.env` |
| Pydantic | Validation request body (`QueryQuestion`) |

---

## Cấu trúc dự án

```
datcuong-ai-chatbot/
├── api/
│   ├── __init__.py
│   └── api.py                  # FastAPI app: khởi tạo models, định nghĩa RAG pipeline (LangGraph), endpoint
│
├── components/
│   ├── __init__.py
│   ├── query_translation.py    # Bước 1: Dịch/chuẩn hóa câu hỏi (hiện là pass-through)
│   ├── query_analysis.py       # Bước 2: Phân tích câu hỏi thành structured query (hiện là pass-through)
│   ├── document_ranking.py     # Xếp hạng tài liệu (file rỗng, chưa triển khai)
│   └── router.py               # Định tuyến query đến collection phù hợp (stub: luôn trả về "legislation")
│
├── utils/
│   ├── __init__.py
│   ├── dto.py                  # Pydantic schema: QueryQuestion { query: str }
│   ├── mongo_handler.py        # Kết nối MongoDB: get_legislation_by_query(), convert_document_from_db_to_available_json()
│   └── data_processing.py      # Chuyển đổi cây nội dung MongoDB → văn bản phẳng cho ChromaDB
│
├── database/                   # Thư mục lưu ChromaDB (cần tạo thủ công hoặc build từ MongoDB)
│   └── all-MiniLM-L6-v2/
│       └── 3000_300/           # Vector store: chunk_size=3000, chunk_overlap=300 (tên thư mục theo quy ước)
│           └── ...             # ChromaDB files (collection: "legislation")
│
└── requirements.txt
```

---

## Hướng dẫn chạy local

### Điều kiện tiên quyết

- **Python 3.9+**
- **Ollama** đã cài đặt và đang chạy, với model `llama3` đã tải về
- **MongoDB** đang chạy và có dữ liệu văn bản pháp luật (collection `laws`)
- **ChromaDB** đã được build sẵn tại `./database/all-MiniLM-L6-v2/3000_300/`

### 1. Clone repository

```bash
git clone <repository-url>
cd datcuong-ai-chatbot
```

### 2. Tạo môi trường ảo và cài dependencies

```bash
python -m venv venv
source venv/bin/activate       # Linux/macOS
# hoặc: venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

### 3. Cài đặt và khởi động Ollama

```bash
# Cài Ollama (macOS)
brew install ollama

# Tải model llama3
ollama pull llama3

# Khởi động Ollama server (nếu chưa chạy)
ollama serve
```

Ollama mặc định lắng nghe tại `http://localhost:11434`.

### 4. Tạo file `.env`

Tạo file `.env` ở thư mục `api/` (vì `api.py` đọc `dotenv_path=".env"` tương đối với nơi chạy):

```env
# MongoDB
MONGO_URI=mongodb://localhost:27017
MONGO_DBNAME=law_linking

# LangSmith (tùy chọn — dùng để theo dõi pipeline)
LANGSMITH_TRACING=false
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=your_project_name
```

### 5. Build ChromaDB từ MongoDB (nếu chưa có)

Vector store tại `./database/all-MiniLM-L6-v2/3000_300/` cần được tạo trước khi chạy API. Hiện không có script build sẵn trong repo — cần tự viết script sử dụng `utils/data_processing.py` và `utils/mongo_handler.py`:

```python
from utils.mongo_handler import get_legislation_by_query
from utils.data_processing import build_chroma_document_from_mongo_document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Lấy tất cả văn bản từ MongoDB
result = get_legislation_by_query({})
documents = result["data"]

# Chuyển đổi sang định dạng ChromaDB
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma(
    persist_directory="./database/all-MiniLM-L6-v2/3000_300",
    collection_name="legislation",
    embedding_function=embedding_model,
)

for doc in documents:
    chroma_doc = build_chroma_document_from_mongo_document(doc)
    vector_store.add_texts(
        texts=[chroma_doc["documents"]],
        metadatas=[chroma_doc["metadata"]]
    )
```

### 6. Chạy ứng dụng

```bash
# Chạy từ thư mục gốc dự án
cd api
python api.py
```

Service sẽ khởi động tại `http://localhost:28080`.

Swagger UI: `http://localhost:28080/docs`

---

## Biến môi trường

File `.env` được đọc tại `api/.env` (tương đối theo vị trí chạy script).

| Biến | Mô tả | Bắt buộc | Ví dụ |
|---|---|---|---|
| `MONGO_URI` | MongoDB connection string | ✅ | `mongodb://localhost:27017` |
| `MONGO_DBNAME` | Tên database MongoDB | ❌ | `law_linking` (mặc định) |
| `LANGSMITH_TRACING` | Bật/tắt LangSmith tracing | ❌ | `false` |
| `LANGSMITH_ENDPOINT` | URL LangSmith API | ❌ | `https://api.smith.langchain.com` |
| `LANGSMITH_API_KEY` | API key của LangSmith | ❌ (chỉ khi tracing bật) | `ls__...` |
| `LANGSMITH_PROJECT` | Tên project trên LangSmith | ❌ | `law-chatbot` |

---

## API Endpoints

### `POST /agents/question-answering`

Nhận câu hỏi pháp lý bằng tiếng Việt, thực thi RAG pipeline, trả về câu trả lời kèm theo các đoạn văn bản nguồn được dùng để sinh câu trả lời.

**Request Body:**

```json
{
  "query": "Điều kiện để được cấp giấy chứng nhận quyền sử dụng đất là gì?"
}
```

| Trường | Kiểu | Mô tả |
|---|---|---|
| `query` | string | Câu hỏi pháp lý bằng tiếng Việt |

**Response:**

```json
{
  "context": [
    { "0": "Điều 98. Điều kiện cấp Giấy chứng nhận quyền sử dụng đất..." },
    { "1": "Điều 100. Cấp Giấy chứng nhận quyền sử dụng đất cho hộ gia đình..." },
    { "2": "..." },
    { "3": "..." },
    { "4": "..." }
  ],
  "answer": "Theo quy định của pháp luật hiện hành, điều kiện để được cấp giấy chứng nhận quyền sử dụng đất bao gồm..."
}
```

| Trường response | Mô tả |
|---|---|
| `context` | Danh sách 5 đoạn văn bản pháp luật liên quan nhất được truy xuất từ ChromaDB (top-k=5) |
| `answer` | Câu trả lời tổng hợp bằng tiếng Việt từ Llama 3, chỉ dựa trên `context` |

---

## Ghi chú bổ sung

### Luồng xử lý chi tiết (LangGraph StateGraph)

Pipeline được xây dựng bằng `LangGraph.StateGraph` với 4 node nối tiếp nhau:

```
translation → analysis → retrieve → generate
```

**State schema:**

| Trường | Kiểu | Mô tả |
|---|---|---|
| `raw_question` | str | Câu hỏi gốc từ người dùng |
| `query` | str | Câu hỏi sau bước translation |
| `structured_query` | str | Câu hỏi sau bước analysis |
| `context` | List[Document] | 5 tài liệu được retrieve |
| `answer` | str | Câu trả lời cuối cùng |

**Prompt template cho LLM:**

```
Bạn là một luật sư giàu kinh nghiệm với vai trò tư vấn pháp lý cho khách hàng.
Hãy trả lời câu hỏi của khách hàng bằng tiếng Việt CHỈ dựa trên các tài liệu đã cung cấp:
{context}
Câu hỏi: {question}
```

---

### Tiền xử lý dữ liệu (`utils/data_processing.py`)

Hàm `content_processing(content)` duyệt đệ quy cây nội dung văn bản (cấu trúc lồng nhau `header`, `description`, `mainContent`, `footer`) và nối tất cả trường `value` thành một chuỗi văn bản phẳng duy nhất. Kết quả được xóa khoảng trắng thừa trước khi đưa vào ChromaDB.

**Metadata được lưu kèm mỗi document trong ChromaDB:**

```json
{
  "id": "mongodb_object_id",
  "name": "LUẬT ĐẤT ĐAI",
  "category": "Luật",
  "department": "Quốc hội",
  "numberDoc": "45/2013/QH13",
  "fields": ["Đất đai"]
}
```

---

### Cấu hình ChromaDB

- **Thư mục lưu trữ:** `./database/all-MiniLM-L6-v2/3000_300/`
- **Tên collection:** `legislation`
- **Embedding model:** `all-MiniLM-L6-v2` (384 chiều, đa ngữ)
- **Quy ước đặt tên thư mục:** `<embedding_model>/<chunk_size>_<chunk_overlap>` — gợi ý chunk_size=3000, chunk_overlap=300 (không có text splitter trong code hiện tại, văn bản được đưa vào nguyên bản)

---

### Observability với LangSmith

Khi `LANGSMITH_TRACING=true`, toàn bộ các bước trong pipeline (translation, analysis, retrieve, generate) được tự động ghi lại trên LangSmith dashboard — hữu ích để debug và đánh giá chất lượng RAG.

---

### Các hạn chế đã biết

- **`query_translation` và `query_analysis` là pass-through:** Cả hai hàm hiện tại chỉ trả lại nguyên câu hỏi mà không xử lý gì. Các tính năng như dịch sang tiếng Anh, mở rộng truy vấn (query expansion), hay phân tích ý định chưa được triển khai.
- **`document_ranking.py` rỗng:** File tồn tại nhưng không có nội dung — reranking tài liệu chưa được triển khai.
- **`router.py` là stub:** Luôn trả về `"legislation"` bất kể nội dung câu hỏi — không có logic định tuyến thực sự.
- **Không có script build ChromaDB:** Phải tự viết script để đưa dữ liệu từ MongoDB vào ChromaDB trước khi chạy service.
- **Embedding đơn ngữ:** `all-MiniLM-L6-v2` là mô hình tiếng Anh — khả năng embedding tiếng Việt có thể kém hơn các mô hình đa ngữ chuyên biệt như `paraphrase-multilingual-MiniLM-L12-v2` hoặc PhoBERT.
- **LLM phụ thuộc Ollama local:** Service yêu cầu Ollama đang chạy với model `llama3` được tải sẵn. Nếu Ollama không sẵn sàng, toàn bộ pipeline sẽ lỗi.
- **Không có xử lý lỗi cho ChromaDB:** Nếu thư mục `./database/all-MiniLM-L6-v2/3000_300/` không tồn tại, service lỗi ngay khi khởi động mà không có thông báo rõ ràng.
- **`mongo_handler.py` comment out datetime conversion:** Các trường `dateApproved`, `createdAt`, `updatedAt` không được chuyển đổi kiểu — có thể gây lỗi serialization nếu dùng kết quả MongoDB trực tiếp.
