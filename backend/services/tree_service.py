from typing import List, Dict, Any
from backend.services.pipeline_service import pipeline_service
from backend.schemas.tree import TreeLevelSummary, TreeSummaryResponse

class TreeService:
    def get_tree_summary(self, document_id: str) -> TreeSummaryResponse:
        pipeline = pipeline_service.get_pipeline(document_id)
        if not pipeline or not pipeline._tree:
            raise ValueError(f"Tree for document '{document_id}' not found or not built.")
        
        tree = pipeline._tree
        all_nodes = tree.all_nodes_flat()
        
        # Count clusters per level
        level_counts = {}
        max_depth = 0
        for node in all_nodes:
            is_cluster = len(node.children) > 0 # Summary nodes are clusters
            if is_cluster:
                level_counts[node.layer] = level_counts.get(node.layer, 0) + 1
            max_depth = max(max_depth, node.layer)
            
        levels = [
            TreeLevelSummary(level=lvl, num_clusters=count)
            for lvl, count in sorted(level_counts.items())
        ]
        
        return TreeSummaryResponse(
            document_id=document_id,
            levels=levels,
            total_nodes=len(all_nodes),
            max_depth=max_depth
        )

tree_service = TreeService()
