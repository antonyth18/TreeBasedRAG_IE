import logging
from typing import Literal, Optional

import numpy as np

from context_assembler import assemble_context
from retriever import retrieve_collapsed_tree, retrieve_tree_traversal
from node import RaptorTree
from summarization import LLMSummarizer
from tree_builder import build_tree
from tree_serializer import load_tree

logger = logging.getLogger(__name__)


class RaptorPipeline:
    def __init__(
        self,
        embed_model: str = "multi-qa-mpnet-base-cos-v1",
        llm_model: str = "llama3.2",
        strategy: Literal["collapsed", "traversal"] = "collapsed",
        max_tokens: int = 2000,
        use_faiss: bool = False,
    ):
        self.embed_model = embed_model
        self.llm_model   = llm_model
        self.strategy    = strategy
        self.max_tokens  = max_tokens
        self.use_faiss   = use_faiss
        self._tree       = None
        self._embedder   = None
        self._faiss_index = None
        self._faiss_nodes = None

    def build(
        self,
        pdf_path: str,
        save_path: Optional[str] = None,
        min_cluster_size: int = 2,
        max_cluster_tokens: int = 8000,
    ) -> RaptorTree:
        from pdf_parser import parse_pdf

        chunks     = parse_pdf(pdf_path)
        embedder   = self._get_embedder()
        summarizer = LLMSummarizer(model=self.llm_model)

        self._tree = build_tree(
            chunks             = chunks,
            embedder           = embedder,
            summarizer         = summarizer,
            min_cluster_size   = min_cluster_size,
            max_cluster_tokens = max_cluster_tokens,
            save_path          = save_path,
            source_pdf         = pdf_path,
        )

        if self.use_faiss:
            self._build_faiss_index()

        return self._tree

    def load(self, tree_path: str) -> None:
        self._tree = load_tree(tree_path, active_embed_model=self._get_embedder().MODEL)
        if self.use_faiss:
            self._build_faiss_index()
        logger.info(f"Tree loaded: {self._tree}")

    # Retrieval and assembly
    def retrieve(
        self,
        query: str,
        max_tokens: Optional[int] = None,
        k: int = 3,
        depth: Optional[int] = None,
    ) -> str:
        if self._tree is None:
            raise RuntimeError("No tree loaded. Call build() or load() first.")

        max_tokens = max_tokens or self.max_tokens
        query_emb  = self._get_embedder().encode_query(query)

        if self.strategy == "collapsed":
            nodes = (self._retrieve_collapsed_faiss(query_emb, max_tokens)
                     if self.use_faiss and self._faiss_index is not None
                     else retrieve_collapsed_tree(self._tree, query_emb, max_tokens))
        else:
            nodes = retrieve_tree_traversal(self._tree, query_emb, k=k, depth=depth)

        return assemble_context(nodes, max_tokens)

    # FIASS Integration
    def _build_faiss_index(self) -> None:
        try:
            import faiss
            all_nodes        = self._tree.all_nodes_flat()
            embeddings       = np.stack([n.embedding for n in all_nodes]).astype(np.float32)
            index            = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)
            self._faiss_nodes = all_nodes
            self._faiss_index = index
            logger.info(f"FAISS index built: {index.ntotal} vectors")
        except ImportError:
            logger.warning("faiss-cpu not installed. Falling back to numpy.")
            self.use_faiss = False

    def _retrieve_collapsed_faiss(self, query_emb: np.ndarray, max_tokens: int):
        n_candidates = min(len(self._faiss_nodes), max(50, max_tokens // 20))
        _, indices   = self._faiss_index.search(query_emb.reshape(1, -1).astype(np.float32), n_candidates)
        selected, total_tokens = [], 0
        for idx in indices[0]:
            if idx < 0:
                continue
            node = self._faiss_nodes[idx]
            if total_tokens + node.token_count <= max_tokens:
                selected.append(node)
                total_tokens += node.token_count
        return selected


    def _get_embedder(self):
        if self._embedder is None:
            from embedder import SBERTEmbedder
            self._embedder = SBERTEmbedder(self.embed_model)
        return self._embedder

    def __repr__(self) -> str:
        return (f"RaptorPipeline(strategy={self.strategy!r}, max_tokens={self.max_tokens}, "
                f"model={self.embed_model!r}, tree={repr(self._tree) if self._tree else 'none'})")