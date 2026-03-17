import logging
from typing import List, Optional

import numpy as np

from node import RaptorNode, RaptorTree

logger = logging.getLogger(__name__)

# Collapsed tree retrieval
def retrieve_collapsed_tree(
    tree: RaptorTree,
    query_emb: np.ndarray,
    max_tokens: int = 2000,
) -> List[RaptorNode]:
    all_nodes = tree.all_nodes_flat()
    if not all_nodes:
        return []

    scores        = np.stack([n.embedding for n in all_nodes]) @ query_emb
    ranked        = np.argsort(scores)[::-1]
    selected      = []
    total_tokens  = 0

    for idx in ranked:
        node = all_nodes[idx]
        if total_tokens + node.token_count <= max_tokens:
            selected.append(node)
            total_tokens += node.token_count

    logger.debug(f"collapsed_tree: {len(selected)}/{len(all_nodes)} nodes ({total_tokens}/{max_tokens} tokens)")
    return selected

# Tree traversal retrieval
def retrieve_tree_traversal(
    tree: RaptorTree,
    query_emb: np.ndarray,
    k: int = 3,
    depth: Optional[int] = None,
) -> List[RaptorNode]:
    depth       = depth or tree.num_layers
    current_ids = list(tree.root_ids)
    selected    = []

    for _ in range(depth):
        if not current_ids:
            break

        current_nodes = [tree.nodes[i] for i in current_ids if i in tree.nodes]
        if not current_nodes:
            break

        scores      = np.stack([n.embedding for n in current_nodes]) @ query_emb
        top_nodes   = [current_nodes[i] for i in np.argsort(scores)[::-1][:min(k, len(current_nodes))]]

        selected.extend(top_nodes)
        current_ids = [child for node in top_nodes for child in node.children]

    logger.debug(f"tree_traversal: {len(selected)} nodes selected (k={k})")
    return selected