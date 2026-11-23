"""FastAPI route definitions for ChromaDB Vector Database and Knowledge Base management."""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from pydantic import BaseModel, Field
from app.config.chromadb import ChromaService, get_chroma_db
from app.services.document_parser import DocumentParserService, SUPPORTED_EXTENSIONS

router = APIRouter(prefix="/knowledge", tags=["Knowledge Base (ChromaDB)"])


def get_chroma_service() -> ChromaService:
    """Dependency provider for ChromaService instance."""
    return get_chroma_db()


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
    "/upload",
    status_code=status.HTTP_201_CREATED,
    summary="Upload and index document into ChromaDB",
    description="Parses PDF, TXT, MD, CSV, or JSON, splits into overlapping chunks, and embeds into ChromaDB for RAG.",
)
async def upload_document(
    file: UploadFile = File(..., description="Document file to parse and index (.pdf, .txt, .md, .csv, .json)"),
    category: Optional[str] = Form(default="documents", description="Category tag for filtering"),
    title: Optional[str] = Form(default=None, description="Optional custom document title"),
    chunk_size: int = Form(default=600, description="Maximum characters per chunk"),
    chunk_overlap: int = Form(default=80, description="Overlapping characters between adjacent chunks"),
    chroma: ChromaService = Depends(get_chroma_service),
) -> Dict[str, Any]:
    """Parse an uploaded file into semantic text chunks and index into ChromaDB."""
    filename = file.filename or "uploaded_file"
    if not DocumentParserService.is_supported(filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{filename}'. Allowed formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    # 15MB limit check
    if len(file_bytes) > 15 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File size exceeds the 15MB limit.",
        )

    try:
        extracted_text, metadata = DocumentParserService.extract_text(file_bytes, filename)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to parse document: {str(exc)}",
        )

    if not extracted_text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not extract any readable text from the uploaded document.",
        )

    chunks = DocumentParserService.chunk_text(
        text=extracted_text,
        chunk_size=max(200, min(chunk_size, 2000)),
        overlap=max(0, min(chunk_overlap, 300)),
    )

    doc_title = title or filename
    total_chunks = len(chunks)
    metadatas = []
    for i in range(total_chunks):
        chunk_meta = {
            "title": f"{doc_title} (Part {i+1}/{total_chunks})" if total_chunks > 1 else doc_title,
            "category": category or "documents",
            "source": filename,
            "chunk_index": i + 1,
            "total_chunks": total_chunks,
            "file_type": metadata.get("file_type", "unknown"),
        }
        metadatas.append(chunk_meta)

    doc_ids = chroma.add_documents(documents=chunks, metadatas=metadatas)

    return {
        "status": "success",
        "message": f"Successfully parsed '{filename}' into {total_chunks} chunk(s) and indexed into ChromaDB vector store.",
        "filename": filename,
        "file_type": metadata.get("file_type", "unknown"),
        "total_chunks": total_chunks,
        "doc_ids": doc_ids,
        "stats": chroma.get_stats(),
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
