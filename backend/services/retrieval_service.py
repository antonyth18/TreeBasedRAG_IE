from typing import List
import uuid
from backend.services.pipeline_service import pipeline_service
from backend.schemas.query import RetrievedNode, LevelNodes, RetrievedNodesResponse

class RetrievalService:
    def format_retrieved_nodes(self, query_id: str) -> RetrievedNodesResponse:
        query_data = pipeline_service.get_query(query_id)
        if not query_data:
            raise ValueError(f"Query '{query_id}' not found.")
            
        nodes = query_data["nodes"]
        # Group by level
        level_map = {}
        for node in nodes:
            lvl = node.layer
            if lvl not in level_map:
                level_map[lvl] = []
            
            level_map[lvl].append(RetrievedNode(
                node_id=str(node.index),
                level=node.layer,
                text=node.text,
                similarity=getattr(node, "last_score", 0.0),
                summary=node.text if len(node.children) > 0 else None
            ))
            
        levels = [
            LevelNodes(level=lvl, nodes=level_nodes)
            for lvl, level_nodes in sorted(level_map.items(), reverse=True)
        ]
        
        return RetrievedNodesResponse(
            query_id=query_id,
            total_retrieved=len(nodes),
            levels=levels
        )

retrieval_service = RetrievalService()
