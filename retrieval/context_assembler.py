import logging
from typing import List

import tiktoken

from tree.node import RaptorNode

logger = logging.getLogger(__name__)

_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text))


def _sort_nodes_leaf_first(nodes: List[RaptorNode]) -> List[RaptorNode]:
    """
    Sorts nodes so that leaf nodes (layer 0) appear first and higher-level
    summary nodes appear last.

    Within each layer, the original relative order is preserved so that
    the rank ordering from retrieval is maintained as a tiebreaker.

    This ensures that if the token budget is exhausted mid-assembly,
    the truncated nodes are summary nodes rather than the more reliable
    leaf-level source chunks.
    """
    return sorted(nodes, key=lambda n: n.layer)


def assemble_context(
    nodes: List[RaptorNode],
    max_tokens: int,
    separator: str = "\n\n---\n\n",
) -> str:
    """
    Assembles retrieved nodes into a single context string within the token
    budget. Leaf nodes are prioritized over summary nodes — if truncation
    occurs, summary nodes are dropped first.

    Args:
        nodes:      Retrieved nodes in any order (will be sorted internally).
        max_tokens: Maximum token budget for the assembled context string.
        separator:  String used to join node text segments.

    Returns:
        Assembled context string, or empty string if nodes is empty.
    """
    if not nodes:
        return ""

    # Sort leaf nodes first so truncation always drops summary nodes
    sorted_nodes = _sort_nodes_leaf_first(nodes)

    sep_tokens   = _count_tokens(separator)
    parts        = []
    total_tokens = 0
    dropped      = []

    for node in sorted_nodes:
        seg_tokens = _count_tokens(node.text)
        extra      = sep_tokens if parts else 0

        if total_tokens + seg_tokens + extra <= max_tokens:
            parts.append(node.text)
            total_tokens += seg_tokens + extra
        else:
            dropped.append(node)

    # Log what was dropped so you can tune max_tokens if needed
    if dropped:
        dropped_layers = [n.layer for n in dropped]
        logger.debug(
            f"assemble_context: truncated {len(dropped)} nodes "
            f"(layers={dropped_layers}) due to token budget"
        )

    logger.debug(
        f"assemble_context: {len(parts)}/{len(nodes)} nodes included "
        f"({total_tokens}/{max_tokens} tokens) | "
        f"leaf nodes={sum(1 for n in sorted_nodes[:len(parts)] if n.layer == 0)} | "
        f"summary nodes={sum(1 for n in sorted_nodes[:len(parts)] if n.layer > 0)}"
    )

    return separator.join(parts)