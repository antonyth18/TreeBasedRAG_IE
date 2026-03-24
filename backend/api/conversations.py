from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
import uuid
from backend.services.pipeline_service import pipeline_service

router = APIRouter(prefix="/conversations", tags=["conversations"])

@router.post("", response_model=Dict[str, str])
async def create_conversation():
    conv_id = str(uuid.uuid4())
    pipeline_service.conversations[conv_id] = []
    return {"conversation_id": conv_id}

@router.get("", response_model=List[str])
async def list_conversations():
    return list(pipeline_service.conversations.keys())

@router.get("/{conversation_id}", response_model=List[Dict[str, Any]])
async def get_conversation_messages(conversation_id: str):
    if conversation_id not in pipeline_service.conversations:
        raise HTTPException(status_code=404, detail=f"Conversation '{conversation_id}' not found.")
    return pipeline_service.conversations[conversation_id]
