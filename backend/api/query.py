from fastapi import APIRouter, HTTPException
from backend.schemas.query import QueryRequest, QueryResponse, RetrievedNodesResponse
from backend.services.generation_service import generation_service
from backend.services.retrieval_service import retrieval_service
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/query", tags=["query"])

@router.post("", response_model=QueryResponse)
async def ask_query(request: QueryRequest):
    try:
        return await generation_service.process_query(request)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Query processing failed")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{query_id}/retrieved-nodes", response_model=RetrievedNodesResponse)
async def get_retrieved_nodes(query_id: str):
    try:
        return retrieval_service.format_retrieved_nodes(query_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
