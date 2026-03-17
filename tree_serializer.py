import json
import logging
import os
from typing import Optional

import numpy as np

from node import RaptorNode, RaptorTree

logger = logging.getLogger(__name__)


class ModelMismatchError(Exception):
    # Raised when the embed_model in a loaded tree doesn't match the active model.
    pass


def save_tree(tree: RaptorTree, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)                      # Ensure output directory exists, else create it.

    # Build metadata dict (everything except embeddings) for tree.json
    nodes_meta = {
        str(idx): {
            "index":       node.index,
            "text":        node.text,
            "layer":       node.layer,
            "children":    node.children,
            "token_count": node.token_count,
        }
        for idx, node in tree.nodes.items()
    }

    # Save metadata as JSON
    with open(os.path.join(output_dir, "tree.json"), "w", encoding="utf-8") as f:
        json.dump({
            "root_ids":    tree.root_ids,
            "num_layers":  tree.num_layers,
            "embed_model": tree.embed_model,
            "source_pdf":  tree.source_pdf,
            "nodes":       nodes_meta,
        }, f, ensure_ascii=False, indent=2)

    node_ids = sorted(tree.nodes.keys())
    
    # Save embeddings as compressed .npz
    np.savez_compressed(
        os.path.join(output_dir, "embeddings.npz"),
        node_ids   = node_ids,
        embeddings = np.stack([tree.nodes[i].embedding for i in node_ids]).astype(np.float32),      # float32 halves file size vs float64
    )

    logger.info(f"Tree saved to '{output_dir}' ({len(tree.nodes)} nodes, {tree.num_layers} layers)")


def load_tree(path: str, active_embed_model: Optional[str] = None) -> RaptorTree:
    # check that the expected files exist before attempting to load
    meta_path = os.path.join(path, "tree.json")
    emb_path  = os.path.join(path, "embeddings.npz")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"tree.json not found in '{path}'")
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"embeddings.npz not found in '{path}'")
    
    # Load and validate metadata
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Validate embed_model compatibility
    stored_model = metadata["embed_model"]
    if active_embed_model is not None and stored_model != active_embed_model:
        raise ModelMismatchError(
            f"Embed model mismatch!\n"
            f"  Built with : '{stored_model}'\n"
            f"  Active     : '{active_embed_model}'"
        )

    # Load embeddings and reconstruct nodes
    emb_data  = np.load(emb_path)
    node_ids  = emb_data["node_ids"].tolist()
    emb_array = emb_data["embeddings"].astype(np.float32)
    id_to_emb = {nid: emb_array[i] for i, nid in enumerate(node_ids)}

    nodes = {
        int(sid): RaptorNode(
            index       = m["index"],
            text        = m["text"],
            embedding   = id_to_emb[int(sid)],
            layer       = m["layer"],
            children    = m["children"],
            token_count = m["token_count"],
        )
        for sid, m in metadata["nodes"].items()
    }

    tree = RaptorTree(
        nodes       = nodes,
        root_ids    = metadata["root_ids"],
        num_layers  = metadata["num_layers"],
        embed_model = stored_model,
        source_pdf  = metadata.get("source_pdf", ""),
    )

    logger.info(f"Tree loaded from '{path}' ({len(nodes)} nodes, {tree.num_layers} layers)")
    return tree