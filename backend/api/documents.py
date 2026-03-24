from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
from backend.schemas.document import DocumentUploadResponse, DocumentInfo, DocumentStatus
from backend.services.ingestion_service import ingestion_service
from backend.services.pipeline_service import pipeline_service

router = APIRouter(prefix="/documents", tags=["documents"])

@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith((".pdf", ".txt", ".md")):
        raise HTTPException(status_code=400, detail="Unsupported file type. Use PDF, TXT, or MD.")
    
    content = await file.read()
    document_id = await ingestion_service.upload_document(content, file.filename)
    
    return DocumentUploadResponse(document_id=document_id, status="processing")

@router.get("", response_model=List[DocumentInfo])
async def list_documents():
    return pipeline_service.list_documents()

@router.get("/{document_id}/status", response_model=DocumentStatus)
async def get_document_status(document_id: str):
    info = pipeline_service.get_document_info(document_id)
    if not info:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")
    
    return DocumentStatus(
        status=info["status"],
        num_chunks=info["num_chunks"],
        num_nodes=info["num_nodes"]
    )
