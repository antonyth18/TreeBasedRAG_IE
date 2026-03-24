from pydantic import BaseModel
from typing import Optional, Any

class StatusResponse(BaseModel):
    status: str
    message: Optional[str] = None
    data: Optional[Any] = None
