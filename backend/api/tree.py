from fastapi import APIRouter, HTTPException
from backend.schemas.tree import TreeSummaryResponse
from backend.services.tree_service import tree_service

router = APIRouter(prefix="/documents", tags=["tree"])

@router.get("/{document_id}/tree/summary", response_model=TreeSummaryResponse)
async def get_tree_summary(document_id: str):
    try:
        return tree_service.get_tree_summary(document_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
