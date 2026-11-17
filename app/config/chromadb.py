"""ChromaDB Vector Store database connection and collection configuration."""

import os
import uuid
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path
import chromadb
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("ai_chat_api.config.chromadb")

# Base directory for persistent ChromaDB vector storage
DEFAULT_PERSIST_DIR = str(Path(__file__).resolve().parent.parent.parent / "chroma_data")

_chroma_instance: Optional["ChromaService"] = None


class ChromaService:
    """Connection manager and query client for ChromaDB persistent vector database."""

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> None:
        self.persist_directory = persist_directory or os.getenv("CHROMA_PERSIST_DIRECTORY", DEFAULT_PERSIST_DIR)
        self.collection_name = collection_name or os.getenv("CHROMA_COLLECTION_NAME", "knowledge_base")

        # Ensure persist directory exists
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB persistent client
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        logger.info(
            "ChromaDB connected at '%s' with collection '%s'. Existing document count: %d",
            self.persist_directory,
            self.collection_name,
            self.collection.count(),
        )

        # Seed default business knowledge if collection is empty
        if self.collection.count() == 0:
            self.seed_default_knowledge()

    def seed_default_knowledge(self) -> None:
        """Seed default knowledge base documents for out-of-the-box RAG testing."""
        default_docs = [
            {
                "id": "kb_shipping_01",
                "text": "Standard shipping takes 2 to 3 business days for domestic delivery. Expedited shipping delivers within 24 hours.",
                "metadata": {"category": "shipping", "title": "Shipping Timelines", "source": "shipping_policy.pdf"},
            },
            {
                "id": "kb_returns_01",
                "text": "Items can be returned within 30 days of delivery in their original condition and packaging. Full refunds are processed to the original payment method within 3-5 business days.",
                "metadata": {"category": "returns", "title": "Return & Refund Policy", "source": "refund_guidelines.pdf"},
            },
            {
                "id": "kb_order_tracking_01",
                "text": "Order numbers follow the format 'ORD' followed by digits (e.g. ORD987124). Customers can track live status, carrier info, and ETA online or via the chat assistant.",
                "metadata": {"category": "orders", "title": "Order Tracking System", "source": "order_help.md"},
            },
            {
                "id": "kb_support_hours_01",
                "text": "Customer support is available 24/7 via the AI chat assistant, or Monday to Friday from 9 AM to 6 PM EST via support@example.com.",
                "metadata": {"category": "support", "title": "Support Hours & Contact", "source": "contact_info.md"},
            },
        ]

        logger.info("Seeding %d default knowledge documents into ChromaDB...", len(default_docs))
        self.add_documents(
            documents=[d["text"] for d in default_docs],
            metadatas=[d["metadata"] for d in default_docs],
            ids=[d["id"] for d in default_docs],
        )

    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Add text documents/chunks into the ChromaDB vector store."""
        if not documents:
            return []

        doc_ids = ids or [f"doc_{uuid.uuid4().hex[:10]}" for _ in documents]
        meta_list = metadatas or [{} for _ in documents]

        self.collection.add(
            documents=documents,
            metadatas=meta_list,
            ids=doc_ids,
        )
        logger.info("Added %d documents to ChromaDB collection '%s'.", len(documents), self.collection_name)
        return doc_ids

    def query_similar(
        self,
        query_text: str,
        n_results: int = 3,
        threshold: float = 1.6,
    ) -> List[Dict[str, Any]]:
        """Query ChromaDB for documents semantically similar to query_text."""
        if not query_text or self.collection.count() == 0:
            return []

        actual_k = min(n_results, self.collection.count())
        if actual_k <= 0:
            return []

        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=actual_k,
            )

            hits = []
            if results and results.get("documents") and len(results["documents"]) > 0:
                docs = results["documents"][0]
                metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
                ids = results["ids"][0] if results.get("ids") else [""] * len(docs)
                dists = results["distances"][0] if results.get("distances") else [0.0] * len(docs)

                for doc, meta, doc_id, dist in zip(docs, metas, ids, dists):
                    if dist <= threshold:
                        hits.append({
                            "id": doc_id,
                            "text": doc,
                            "metadata": meta,
                            "distance": round(float(dist), 4),
                        })
            return hits
        except Exception as exc:
            logger.error("Error querying ChromaDB: %s", exc)
            return []

    def list_documents(self) -> List[Dict[str, Any]]:
        """Retrieve all documents stored in the ChromaDB collection."""
        try:
            data = self.collection.get()
            docs = data.get("documents") or []
            metas = data.get("metadatas") or [{}] * len(docs)
            ids = data.get("ids") or []

            items = []
            for doc_id, doc, meta in zip(ids, docs, metas):
                items.append({
                    "id": doc_id,
                    "text": doc,
                    "metadata": meta or {},
                })
            return items
        except Exception as exc:
            logger.error("Error listing ChromaDB documents: %s", exc)
            return []

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document by its ID from ChromaDB."""
        try:
            self.collection.delete(ids=[doc_id])
            logger.info("Deleted document '%s' from ChromaDB.", doc_id)
            return True
        except Exception as exc:
            logger.error("Error deleting document '%s': %s", doc_id, exc)
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        return {
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory,
            "total_documents": self.collection.count(),
        }


def get_chroma_db() -> ChromaService:
    """Singleton getter for ChromaDB connection instance."""
    global _chroma_instance
    if _chroma_instance is None:
        _chroma_instance = ChromaService()
    return _chroma_instance
