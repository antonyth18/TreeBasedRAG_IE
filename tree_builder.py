import logging
from typing import List, Optional

import tiktoken

from clustering import cluster_nodes
from node import RaptorNode, RaptorTree
from summarization import LLMSummarizer
from tree_serializer import save_tree

logger = logging.getLogger(__name__)

_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text))


def build_tree(
    chunks: List[str],
    embedder,
    summarizer: Optional[LLMSummarizer] = None,
    min_cluster_size: int = 2,
    max_cluster_tokens: int = 8000,
    save_path: Optional[str] = None,
    source_pdf: str = "",
) -> RaptorTree:
    if summarizer is None:
        summarizer = LLMSummarizer()

    all_nodes = {}
    node_id   = 0

    # Leaf nodes
    embeddings = embedder.encode(chunks)

    for text, emb in zip(chunks, embeddings):
        all_nodes[node_id] = RaptorNode(
            index=node_id, text=text, embedding=emb,
            layer=0, children=[], token_count=_count_tokens(text),
        )
        node_id += 1

    logger.info(f"Leaf nodes: {len(all_nodes)}")

    current_layer_nodes = list(all_nodes.values())
    layer = 0

    # Build loop
    while len(current_layer_nodes) > min_cluster_size:
        clusters = cluster_nodes(current_layer_nodes, max_cluster_tokens=max_cluster_tokens)

        if len(clusters) == 1:
            logger.info("Clustering produced 1 cluster — terminating build loop.")
            break

        logger.info(f"Layer {layer} → {layer + 1}: {len(clusters)} clusters, summarising...")

        parent_nodes = []
        for c_nodes in clusters.values():
            summary_text = summarizer.summarize([n.text for n in c_nodes])
            parent = RaptorNode(
                index       = node_id,
                text        = summary_text,
                embedding   = embedder.encode([summary_text])[0],
                layer       = layer + 1,
                children    = [n.index for n in c_nodes],
                token_count = _count_tokens(summary_text),
            )
            all_nodes[node_id] = parent
            parent_nodes.append(parent)
            node_id += 1

        layer += 1
        current_layer_nodes = parent_nodes
        logger.info(f"Layer {layer} done: {len(parent_nodes)} nodes | LLM tokens: {summarizer.total_tokens_used}")

    # Assemble + save
    root_ids = [n.index for n in current_layer_nodes]

    tree = RaptorTree(
        nodes      = all_nodes,
        root_ids   = root_ids,
        num_layers = layer + 1,
        embed_model= embedder.MODEL,
        source_pdf = source_pdf,
    )

    logger.info(f"Build complete: {len(all_nodes)} nodes, {tree.num_layers} layers, {len(root_ids)} roots")

    if save_path:
        save_tree(tree, save_path)

    return tree