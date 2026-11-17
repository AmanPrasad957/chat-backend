# AI Chat API Backend (`chat-backend`)

Production-ready conversational AI backend powered by **FastAPI**, **Google Gemini LLM**, **ChromaDB Vector RAG**, and local **MongoDB** Authentication & Authorization with JWT Access and Refresh Tokens.

---

## 🚀 Features

- **Google Gemini LLM**: Multi-model dynamic conversational intelligence with fallback.
- **ChromaDB Vector Store (RAG)**: Persistent local vector database for grounded knowledge retrieval.
- **Local MongoDB Authentication & Authorization**:
  - Secure user registration with unique email validation.
  - Password hashing using `bcrypt`.
  - JWT Bearer Access Tokens (short-lived, 30 min).
  - JWT Refresh Tokens with database storage, rotation, and revocation on logout.
  - Protected endpoints using `get_current_user` dependency.
- **Thread & Message Management**: Multi-turn conversation state machine, per-thread deletion, and per-message deletion.
- **Auto-generated Documentation**: Swagger UI at `/docs` and ReDoc at `/redoc`.

---

## 📁 Project Structure

```text
chat-backend/
├── app/
│   ├── main.py                  # FastAPI application entrypoint, CORS, and lifespans
│   ├── db/
│   │   └── mongodb.py           # Async Motor client, connection pool, and indexes
│   ├── routes/
│   │   ├── auth.py              # Register, login, refresh, logout, /me
│   │   ├── chat.py              # Chat, threads, messages, and state
│   │   └── knowledge.py         # ChromaDB documents and semantic query
│   ├── controllers/
│   │   └── chat_controller.py   # Request orchestration and response modeling
│   ├── services/
│   │   ├── auth_service.py      # Password hashing, JWT token creation/verification
│   │   ├── llm_service.py       # Google Gemini LLM and RAG prompt injection
│   │   └── chroma_service.py    # Persistent ChromaDB vector store
│   └── schemas/
│       ├── auth_schema.py       # Pydantic auth and token models
│       └── chat_schema.py       # Pydantic request and state models
├── tests/
│   ├── test_auth.py             # Automated MongoDB & JWT auth test suite
│   └── test_chat.py             # Automated Chat & ChromaDB test suite
├── .env.example                 # Environment configuration template
├── .gitignore                   # Ignores .env, venv, chroma_data, and pycache
└── requirements.txt             # Python dependencies
```

---

## 🛠️ Quickstart Guide

### 1. Prerequisites
- Python 3.10+
- Local MongoDB running on `mongodb://127.0.0.1:27017`

### 2. Setup Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env and set your GEMINI_API_KEY
```

### 3. Run Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- Health Check: [http://localhost:8000/health](http://localhost:8000/health)

---

## 🧪 Run Automated Tests

```bash
# Test MongoDB Authentication & JWT Tokens
./venv/bin/python3 tests/test_auth.py

# Test Chat, Gemini & ChromaDB RAG
./venv/bin/python3 tests/test_chat.py
```
