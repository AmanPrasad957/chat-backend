"""Database connection configuration package.

Contains connection managers and clients for:
- MongoDB (Document DB for users, auth, tokens): mongodb.py
- ChromaDB (Vector DB for RAG, knowledge embeddings): chromadb.py
"""

from app.config.mongodb import (
    connect_to_mongo,
    close_mongo_connection,
    get_database,
    check_mongo_health,
)
from app.config.chromadb import (
    ChromaService,
    get_chroma_db,
)

__all__ = [
    "connect_to_mongo",
    "close_mongo_connection",
    "get_database",
    "check_mongo_health",
    "ChromaService",
    "get_chroma_db",
]
