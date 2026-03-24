from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from backend.services.pipeline_service import pipeline_service

router = APIRouter(prefix="/query", tags=["analytics"])

@router.get("/{query_id}/retrieval-summary", response_model=Dict[str, Any])
async def get_retrieval_summary(query_id: str):
    query_data = pipeline_service.get_query(query_id)
    if not query_data:
        raise HTTPException(status_code=404, detail=f"Query '{query_id}' not found.")
    
    nodes = query_data["nodes"]
    
    # Analyze nodes by level
    level_counts = {}
    for node in nodes:
        lvl = node.level
        level_counts[lvl] = level_counts.get(lvl, 0) + 1
        
    levels_summary = [
        {"level": lvl, "num_nodes": count}
        for lvl, count in sorted(level_counts.items(), reverse=True)
    ]
    
    return {
        "levels": levels_summary,
        "total_retrieved": len(nodes)
    }
