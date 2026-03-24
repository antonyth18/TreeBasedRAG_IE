from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    document_id: str
    query: str
    conversation_id: Optional[str] = None

class RetrievedNode(BaseModel):
    node_id: str
    level: int
    text: str
    similarity: float
    title: Optional[str] = None
    summary: Optional[str] = None

class QueryResponse(BaseModel):
    query_id: str
    query_type: str
    retrieved_nodes: List[RetrievedNode]
    answer: str

class LevelNodes(BaseModel):
    level: int
    nodes: List[RetrievedNode]

class RetrievedNodesResponse(BaseModel):
    query_id: str
    total_retrieved: int
    levels: List[LevelNodes]
