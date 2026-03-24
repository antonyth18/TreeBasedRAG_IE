import logging
from typing import List, Optional, Tuple

import numpy as np

from tree.node import RaptorNode, RaptorTree

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

# MMR deduplication threshold — nodes with pairwise cosine similarity above
# this are considered redundant. Lower = more aggressive deduplication.
_MMR_SIMILARITY_THRESHOLD = 0.75

# Minimum similarity score for a query to be considered "in scope".
# If the top retrieved node scores below this, the query is likely out of scope.
_MIN_RELEVANCE_THRESHOLD = 0.45

# Minimum similarity score a branch must achieve at any layer to continue traversal.
# If the best available node at a layer scores below this, traversal stops entirely
# rather than following a weak branch deeper.
_LAYER_SCORE_THRESHOLD = 0.3

def _deduplicate_mmr(
    nodes: List[RaptorNode],
    scores: List[float],
    threshold: float = _MMR_SIMILARITY_THRESHOLD,
) -> List[RaptorNode]:
    """
    Post-retrieval deduplication using pairwise cosine similarity.

    Iterates through nodes in score order (best first). For each candidate,
    checks its cosine similarity against all already-selected nodes. If it
    exceeds the threshold against any selected node, it is dropped as redundant.

    Args:
        nodes:      Retrieved nodes in ranked order (best first).
        scores:     Corresponding query similarity scores, same order as nodes.
        threshold:  Cosine similarity above which two nodes are considered
                    redundant. The lower-ranked one is dropped.

    Returns:
        Deduplicated list of nodes, preserving rank order.
    """
    if not nodes:
        return []

    selected: List[RaptorNode] = []
    selected_embeddings: List[np.ndarray] = []

    for node in nodes:
        if not selected:
            # Always keep the top-ranked node
            selected.append(node)
            selected_embeddings.append(node.embedding)
            continue

        # Compute cosine similarity against all already-selected nodes
        # Embeddings are L2-normalized in embedder.py so dot product = cosine sim
        emb_matrix  = np.stack(selected_embeddings)       # (n_selected, dim)
        similarities = emb_matrix @ node.embedding         # (n_selected,)
        max_sim      = float(np.max(similarities))

        if max_sim >= threshold:
            logger.debug(
                f"MMR dropped node (max_sim={max_sim:.3f} >= threshold={threshold}): "
                f"'{node.text[:60]}...'"
            )
            continue

        selected.append(node)
        selected_embeddings.append(node.embedding)

    logger.debug(f"MMR deduplication: {len(nodes)} → {len(selected)} nodes "
                 f"(dropped {len(nodes) - len(selected)})")
    return selected


# ── Out-of-Scope Check ─────────────────────────────────────────────────────────

def _check_relevance_threshold(
    top_score: float,
    query: str,
    threshold: float = _MIN_RELEVANCE_THRESHOLD,
) -> bool:
    """
    Checks whether the best-matching node clears the minimum relevance bar.
    If it doesn't, the query is likely out of scope for this document.

    Returns True if retrieval should proceed, False if query is out of scope.
    """
    if top_score < threshold:
        logger.warning(
            f"Out-of-scope query detected: top similarity score {top_score:.3f} "
            f"is below threshold {threshold}. Query: '{query}'"
        )
        return False
    return True


# ── Smart Retrieval Entry Point ────────────────────────────────────────────────

def retrieve(
    tree: RaptorTree,
    query: str,
    query_emb: np.ndarray,
    query_type: str,
    max_tokens: int = 2000,
    k: int = 3,
    mmr_threshold: float = _MMR_SIMILARITY_THRESHOLD,
    relevance_threshold: float = _MIN_RELEVANCE_THRESHOLD,
    layer_score_threshold: float = _LAYER_SCORE_THRESHOLD,
    force_strategy: Optional[str] = None,
) -> Tuple[List[RaptorNode], str]:
    """
    Smart retrieval entry point that:
      1. Classifies the query as broad or specific
      2. Routes to the appropriate retrieval strategy
      3. Applies out-of-scope detection
      4. Deduplicates results via MMR

    Args:
        tree:                  The RAPTOR tree to retrieve from.
        query:                 Raw query string (used for logging).
        query_emb:             Encoded query embedding.
        query_type:            The categorized query intent (broad, specific, definitional, comparative).
        max_tokens:            Token budget for collapsed retrieval.
        k:                     Top-k branches for traversal retrieval.
        mmr_threshold:         Cosine similarity threshold for deduplication.
        relevance_threshold:   Minimum score for in-scope detection.
        layer_score_threshold: Minimum score to continue layer-by-layer traversal.
        force_strategy:        Optional override — pass "collapsed" or "traversal"
                               to skip the classifier entirely. Useful for testing.

    Returns:
        (nodes, strategy_used) — retrieved deduplicated nodes and the strategy
        name that was used, so the caller can log or display it.
    """
    # ── Step 1: Determine strategy ─────────────────────────────────────────────────
    if force_strategy:
        strategy = force_strategy
    else:
        if query_type == "broad":
            strategy = "collapsed"
        elif query_type in {"specific", "definitional", "comparative"}:
            strategy = "traversal"
        else:
            strategy = "traversal"  # Safe default

    logger.info(f"Query classified as '{query_type}' → using strategy '{strategy}': '{query}'")

    # ── Step 2: Initial retrieval with scores ──────────────────────────────────
    if strategy == "collapsed":
        nodes, scores = _retrieve_collapsed_tree_with_scores(
            tree, query_emb, max_tokens
        )
    else:
        nodes, scores = _retrieve_tree_traversal_with_scores(
            tree, query_emb, k, layer_score_threshold=layer_score_threshold
        )

    if not nodes:
        logger.warning("No nodes retrieved.")
        return [], strategy

    # ── Step 3: Out-of-scope detection ────────────────────────────────────────
    top_score = scores[0] if scores else 0.0
    if not _check_relevance_threshold(top_score, query, relevance_threshold):
        # Return empty list — signals to caller that query is out of scope
        return [], strategy

    # Attach scores to nodes so they can be retrieved later (e.g. by API)
    for node, score in zip(nodes, scores):
        setattr(node, "last_score", float(score))

    # ── Step 4: MMR deduplication ─────────────────────────────────────────────
    deduplicated = _deduplicate_mmr(nodes, scores, mmr_threshold)

    return deduplicated, strategy


# ── Internal Retrieval Functions (with scores) ─────────────────────────────────

def _retrieve_collapsed_tree_with_scores(
    tree: RaptorTree,
    query_emb: np.ndarray,
    max_tokens: int = 2000,
) -> Tuple[List[RaptorNode], List[float]]:
    """
    Flat k-NN retrieval across all tree nodes.
    Returns nodes and their corresponding similarity scores, best first.
    """
    all_nodes = tree.all_nodes_flat()
    if not all_nodes:
        return [], []

    scores_arr    = np.stack([n.embedding for n in all_nodes]) @ query_emb
    ranked        = np.argsort(scores_arr)[::-1]

    selected      : List[RaptorNode] = []
    selected_scores: List[float]     = []
    total_tokens  = 0

    for idx in ranked:
        node = all_nodes[idx]
        if total_tokens + node.token_count <= max_tokens:
            selected.append(node)
            selected_scores.append(float(scores_arr[idx]))
            total_tokens += node.token_count

    logger.debug(
        f"collapsed_tree: {len(selected)}/{len(all_nodes)} nodes "
        f"({total_tokens}/{max_tokens} tokens) "
        f"top_score={selected_scores[0]:.3f}" if selected_scores else ""
    )
    return selected, selected_scores


def _retrieve_tree_traversal_with_scores(
    tree: RaptorTree,
    query_emb: np.ndarray,
    k: int = 3,
    depth: Optional[int] = None,
    layer_score_threshold: float = _LAYER_SCORE_THRESHOLD,
) -> Tuple[List[RaptorNode], List[float]]:
    """
    Top-down branch traversal from root to leaves.
    Stops early if best_score in a layer < layer_score_threshold.
    Returns nodes and their similarity scores in traversal order,
    sorted by score descending before returning.
    """
    depth       = depth or tree.num_layers
    current_ids = list(tree.root_ids)
    selected    : List[RaptorNode] = []
    selected_scores: List[float]   = []

    for layer_idx in range(depth):
        if not current_ids:
            break

        current_nodes = [tree.nodes[i] for i in current_ids if i in tree.nodes]
        if not current_nodes:
            break

        scores_arr  = np.stack([n.embedding for n in current_nodes]) @ query_emb
        best_score = float(scores_arr.max())
        
        if best_score < layer_score_threshold:
            logger.debug(f"Traversal stopped at layer {layer_idx}: best_score={best_score:.3f} < threshold={layer_score_threshold}")
            break
            
        top_indices = np.argsort(scores_arr)[::-1][:min(k, len(current_nodes))]
        top_nodes   = [current_nodes[i] for i in top_indices]
        top_scores  = [float(scores_arr[i]) for i in top_indices]

        selected.extend(top_nodes)
        selected_scores.extend(top_scores)
        current_ids = [child for node in top_nodes for child in node.children]

    # Sort by score descending so MMR and threshold checks see best nodes first
    if selected:
        paired          = sorted(zip(selected_scores, selected), key=lambda x: x[0], reverse=True)
        selected_scores = [s for s, _ in paired]
        selected        = [n for _, n in paired]

    logger.debug(
        f"tree_traversal: {len(selected)} nodes selected (k={k}) "
        f"top_score={selected_scores[0]:.3f}" if selected_scores else ""
    )
    return selected, selected_scores


# ── Original Functions (kept for backward compatibility) ───────────────────────

def retrieve_collapsed_tree(
    tree: RaptorTree,
    query_emb: np.ndarray,
    max_tokens: int = 2000,
) -> List[RaptorNode]:
    """Original interface preserved. Use retrieve() for full functionality."""
    nodes, _ = _retrieve_collapsed_tree_with_scores(tree, query_emb, max_tokens)
    return nodes


def retrieve_tree_traversal(
    tree: RaptorTree,
    query_emb: np.ndarray,
    k: int = 3,
    depth: Optional[int] = None,
    layer_score_threshold: float = _LAYER_SCORE_THRESHOLD,
) -> List[RaptorNode]:
    """Original interface preserved. Use retrieve() for full functionality."""
    nodes, _ = _retrieve_tree_traversal_with_scores(tree, query_emb, k, depth, layer_score_threshold)
    return nodes