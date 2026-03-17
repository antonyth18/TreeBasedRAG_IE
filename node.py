from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class RaptorNode:
    index:       int                                                                # Unique identifier for the node
    text:        str                                                                # Text content of the node
    embedding:   np.ndarray                                                         # Embedding vector for the node
    layer:       int                                                                # Layer to which the node belongs
    children:    List[int] = field(default_factory=list)                            # IDs of child nodes
    token_count: int = 0                                                            # Number of tokens in the node's text

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return (f"RaptorNode(index={self.index}, layer={self.layer}, "
                f"tokens={self.token_count}, children={len(self.children)}, "
                f'text="{preview}...")')


@dataclass
class RaptorTree:
    nodes:       Dict[int, RaptorNode]                                              # Mapping of node ID to RaptorNode
    root_ids:    List[int]                                                          # IDs of root nodes
    num_layers:  int                                                                # Number of layers in the tree
    embed_model: str                                                                # Name of the embedding model used
    source_pdf:  str = ""                                                           # Path to the source PDF file

    # All nodes as a flat list for easy access
    def all_nodes_flat(self) -> List[RaptorNode]:
        return list(self.nodes.values())

    def children_of(self, node: RaptorNode) -> List[RaptorNode]:
        return [self.nodes[c] for c in node.children if c in self.nodes]

    def __repr__(self) -> str:
        return (f"RaptorTree(nodes={len(self.nodes)}, layers={self.num_layers}, "
                f'model="{self.embed_model}", source="{self.source_pdf}")')