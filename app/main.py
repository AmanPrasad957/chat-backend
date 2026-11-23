"""Main application entrypoint for AI Chat API with MongoDB Authentication."""

import os
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.config.mongodb import connect_to_mongo, close_mongo_connection, check_mongo_health
from app.routes.chat import router as chat_router
from app.routes.knowledge import router as knowledge_router
from app.routes.auth import router as auth_router

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO if os.getenv("DEBUG", "True").lower() == "true" else logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ai_chat_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown events."""
    logger.info("Starting up AI Chat API...")
    provider = os.getenv("LLM_PROVIDER", "gemini")
    gemini_key_set = bool(os.getenv("GEMINI_API_KEY", "").strip())
    logger.info("Configured Provider: %s | Gemini Key Configured: %s | ChromaDB Active: True", provider, gemini_key_set)

    # Initialize MongoDB connection pool
    await connect_to_mongo()

    yield

    logger.info("Shutting down AI Chat API...")
    # Close MongoDB connection pool
    await close_mongo_connection()


app = FastAPI(
    title="AI Chat API",
    version="1.0.0",
    description="Production-ready conversational chatbot API supporting Google Gemini LLM, ChromaDB Vector RAG, local MongoDB Authentication (JWT access & refresh tokens), NLU intent classification, and state tracking.",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS configuration for web clients and Admin Panel UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount chat, knowledge, and auth routes directly and under /api/v1
app.include_router(chat_router)
app.include_router(chat_router, prefix="/api/v1")
app.include_router(knowledge_router)
app.include_router(knowledge_router, prefix="/api/v1")
app.include_router(auth_router)
app.include_router(auth_router, prefix="/api/v1")


@app.get("/ui", tags=["General"])
async def serve_ui():
    """Information endpoint indicating the UI is integrated into the Next.js frontend."""
    return {
        "message": "Chat UI is integrated into the frontend application.",
        "frontend_url": os.getenv("FRONTEND_URL", "http://localhost:3000"),
        "docs": "/docs",
    }


@app.get("/", tags=["General"])
async def root():
    """Root endpoint: returns service details and documentation links."""
    return {
        "service": "AI Chat API",
        "status": "online",
        "version": "1.0.0",
        "docs": "/docs",
        "frontend": os.getenv("FRONTEND_URL", "http://localhost:3000"),
        "endpoints": {
            "chat": "POST /chat",
            "chat_stream": "POST /chat/stream",
            "api_v1_chat": "POST /api/v1/chat",
            "knowledge": "GET /knowledge",
            "knowledge_upload": "POST /knowledge/upload",
            "auth_register": "POST /auth/register",
            "auth_login": "POST /auth/login",
            "auth_refresh": "POST /auth/refresh",
            "auth_logout": "POST /auth/logout",
            "auth_me": "GET /auth/me",
            "health": "GET /health",
        },
    }


@app.get("/health", tags=["General"])
async def health_check():
    """Health check endpoint with MongoDB status."""
    mongo_ok = await check_mongo_health()
    return {
        "status": "healthy",
        "service": "ai-chat-api",
        "provider": os.getenv("LLM_PROVIDER", "gemini"),
        "gemini_configured": bool(os.getenv("GEMINI_API_KEY", "").strip()),
        "chromadb_configured": True,
        "mongodb_configured": True,
        "mongodb_connected": mongo_ok,
    }


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    debug = os.getenv("DEBUG", "True").lower() == "true"
    uvicorn.run("app.main:app", host=host, port=port, reload=debug)
