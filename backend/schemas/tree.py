from pydantic import BaseModel
from typing import List

class TreeLevelSummary(BaseModel):
    level: int
    num_clusters: int

class TreeSummaryResponse(BaseModel):
    document_id: str
    levels: List[TreeLevelSummary]
    total_nodes: int
    max_depth: int
