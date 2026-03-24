from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class DocumentInfo(BaseModel):
    id: str
    filename: str
    status: str
    upload_time: datetime
    num_chunks: Optional[int] = 0
    num_nodes: Optional[int] = 0

class DocumentUploadResponse(BaseModel):
    document_id: str
    status: str

class DocumentStatus(BaseModel):
    status: str
    num_chunks: int
    num_nodes: int
