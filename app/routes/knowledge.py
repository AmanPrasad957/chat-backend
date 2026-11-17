"""FastAPI route definitions for ChromaDB Vector Database and Knowledge Base management."""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from app.services.chroma_service import ChromaService

router = APIRouter(prefix="/knowledge", tags=["Knowledge Base (ChromaDB)"])

# Singleton ChromaService dependency
_chroma_instance: Optional[ChromaService] = None


def get_chroma_service() -> ChromaService:
    """Dependency provider for ChromaService instance."""
    global _chroma_instance
    if _chroma_instance is None:
        _chroma_instance = ChromaService()
    return _chroma_instance


class DocumentItem(BaseModel):
    """Schema for adding a document to ChromaDB."""

    text: str = Field(..., description="Document content text to embed and store.", examples=["Returns are accepted within 30 days."])
    title: Optional[str] = Field(default="Untitled Document", description="Title of document.")
    category: Optional[str] = Field(default="general", description="Category/tag for filtering.")
    source: Optional[str] = Field(default="manual_input", description="Source document or URL.")
    id: Optional[str] = Field(default=None, description="Optional custom document ID.")


class QueryRequest(BaseModel):
    """Schema for querying ChromaDB via semantic similarity."""

    query: str = Field(..., description="Query text to find semantically similar documents.", examples=["How do returns work?"])
    n_results: int = Field(default=3, ge=1, le=10, description="Number of similar chunks to retrieve.")


@router.get(
    "",
    summary="List all documents in ChromaDB",
    description="Returns all knowledge base documents currently indexed in ChromaDB vector store.",
)
async def list_knowledge(
    chroma: ChromaService = Depends(get_chroma_service),
) -> Dict[str, Any]:
    """List all documents currently indexed in ChromaDB."""
    documents = chroma.list_documents()
    stats = chroma.get_stats()
    return {
        "status": "success",
        "total_documents": len(documents),
        "stats": stats,
        "documents": documents,
    }


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    summary="Add a document to ChromaDB",
    description="Embeds and stores a new document into ChromaDB for Retrieval-Augmented Generation (RAG).",
)
async def add_knowledge(
    item: DocumentItem,
    chroma: ChromaService = Depends(get_chroma_service),
) -> Dict[str, Any]:
    """Add a new document chunk to ChromaDB."""
    doc_ids = chroma.add_documents(
        documents=[item.text],
        metadatas=[{
            "title": item.title,
            "category": item.category,
            "source": item.source,
        }],
        ids=[item.id] if item.id else None,
    )
    return {
        "status": "success",
        "message": "Document indexed successfully in ChromaDB.",
        "doc_id": doc_ids[0] if doc_ids else None,
    }


@router.post(
    "/query",
    summary="Semantic vector search",
    description="Queries ChromaDB using semantic embedding similarity to find the most relevant document chunks.",
)
async def query_knowledge(
    request: QueryRequest,
    chroma: ChromaService = Depends(get_chroma_service),
) -> Dict[str, Any]:
    """Query ChromaDB for relevant knowledge base documents."""
    results = chroma.query_similar(request.query, n_results=request.n_results)
    return {
        "query": request.query,
        "results_count": len(results),
        "results": results,
    }


@router.delete(
    "/{doc_id}",
    summary="Delete a document from ChromaDB",
    description="Removes a specific document from the ChromaDB vector collection.",
)
async def delete_knowledge(
    doc_id: str,
    chroma: ChromaService = Depends(get_chroma_service),
) -> Dict[str, Any]:
    """Delete a document by its ID from ChromaDB."""
    deleted = chroma.delete_document(doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found or could not be deleted.")
    return {
        "status": "success",
        "doc_id": doc_id,
        "message": f"Document '{doc_id}' deleted from ChromaDB.",
    }


@router.get(
    "/stats",
    summary="ChromaDB Collection Statistics",
    description="Returns collection name, persistence path, and document count.",
)
async def get_chroma_stats(
    chroma: ChromaService = Depends(get_chroma_service),
) -> Dict[str, Any]:
    """Retrieve ChromaDB vector store statistics."""
    return chroma.get_stats()
