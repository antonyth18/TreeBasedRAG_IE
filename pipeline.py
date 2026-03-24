import logging
import re
from typing import Literal, Optional

import numpy as np

from generation.generator import LLMGenerator
from retrieval.context_assembler import assemble_context
from retrieval.retriever import retrieve
from retrieval.query_classifier import ZeroShotQueryClassifier
from tree.node import RaptorTree
from tree.summarization import LLMSummarizer
from tree.tree_builder import build_tree
from tree.tree_serializer import load_tree

logger = logging.getLogger(__name__)


class RaptorPipeline:
    def __init__(
        self,
        embed_model: str = "multi-qa-mpnet-base-cos-v1",
        llm_model: str = "llama3.2",
        max_tokens: int = 2000,
        use_faiss: bool = False,
        mmr_threshold: float = 0.75,
        relevance_threshold: float = 0.45,
        layer_score_threshold: float = 0.3,
        enable_generation: bool = True,
    ):
        self.embed_model          = embed_model
        self.llm_model            = llm_model
        self.max_tokens           = max_tokens
        self.use_faiss            = use_faiss
        self.mmr_threshold        = mmr_threshold
        self.relevance_threshold  = relevance_threshold
        self.layer_score_threshold = layer_score_threshold
        self.enable_generation    = enable_generation
        self._tree                = None
        self._embedder            = None
        self._query_classifier    = None
        self._generator           = None
        self._faiss_index         = None
        self._faiss_nodes         = None
        
        # Expose for backend analytics
        self.last_nodes           = []
        self.last_query_type      = None

    def build(
        self,
        pdf_path: str,
        save_path: Optional[str] = None,
        min_cluster_size: int = 2,
        max_cluster_tokens: int = 8000,
        min_chunk_tokens: int = 40,
    ) -> RaptorTree:
        """
        Builds a RAPTOR tree from a PDF document.

        Args:
            pdf_path:           Link to the source PDF.
            save_path:          Optional directory to save the built tree.
            min_cluster_size:   Minimum nodes per cluster.
            max_cluster_tokens: Token budget for a single summary node.
            min_chunk_tokens:   Minimum token count for a chunk to be included.

        Returns:
            The root node of the constructed hierarchical tree.
        """
        # Check if we can reuse an existing tree
        if save_path and os.path.exists(save_path):
            try:
                self.load(save_path)
                return self._tree
            except Exception as e:
                logger.warning(f"Could not load existing tree from {save_path}: {e}. Rebuilding...")

        from ingestion.pdf_parser import parse_pdf

        chunks     = parse_pdf(pdf_path, min_chunk_tokens=min_chunk_tokens)
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

    def query(
        self,
        query: str,
        max_tokens: Optional[int] = None,
        k: int = 3,
        depth: Optional[int] = None,
        force_strategy: Optional[str] = None,
    ) -> str:
        """
        Retrieves relevant context for a query using the smart retrieval pipeline:
          1. Classifies query as broad or specific
          2. Routes to collapsed or traversal retrieval accordingly
          3. Applies out-of-scope detection via similarity threshold
          4. Deduplicates results via MMR
          5. Assembles leaf-first context within token budget

        Args:
            query:          The user query string.
            max_tokens:     Token budget override (defaults to self.max_tokens).
            k:              Top-k branches for traversal retrieval.
            depth:          Max depth for traversal (defaults to tree depth).
            force_strategy: Optional override — "collapsed" or "traversal".
                            Bypasses the query classifier. Useful for testing.

        Returns:
            Assembled context string, or out-of-scope message if query is
            outside the document's content.
        """
        if self._tree is None:
            raise RuntimeError("No tree loaded. Call build() or load() first.")

        max_tokens = max_tokens or self.max_tokens
        query_emb  = self._get_embedder().encode_query(query)

        # Determine query type using the NLI zero-shot classifier
        if force_strategy:
            query_type = "override"
        else:
            query_type, confidence = self._get_classifier().classify_with_confidence(query)

            if confidence < 0.45:
                logger.warning(f"Low classifier confidence ({confidence:.3f}). Defaulting to 'specific'.")
                query_type = "specific"

        # Expose for main.py logging
        self.last_query_type = query_type

        # single traversal for comparative queries causes asymmetric branch selection where only the higher-scoring concept's branch gets followed
        if query_type == "comparative" and force_strategy is None:
            if self.use_faiss and self._faiss_index is not None:
                logger.debug("Comparative decomposition skipped — FAISS path active")
            else:
                sub_queries = self._decompose_comparative(query)
                if len(sub_queries) == 2:
                    sub_max_tokens = max_tokens // 2
                    
                    emb_a = self._get_embedder().encode_query(sub_queries[0])
                    nodes_a, _ = retrieve(
                        tree=self._tree,
                        query=sub_queries[0],
                        query_emb=emb_a,
                        query_type="specific",
                        max_tokens=sub_max_tokens,
                        k=k,
                        mmr_threshold=self.mmr_threshold,
                        relevance_threshold=self.relevance_threshold,
                        layer_score_threshold=self.layer_score_threshold,
                        force_strategy=force_strategy,
                    )
                    
                    emb_b = self._get_embedder().encode_query(sub_queries[1])
                    nodes_b, _ = retrieve(
                        tree=self._tree,
                        query=sub_queries[1],
                        query_emb=emb_b,
                        query_type="specific",
                        max_tokens=sub_max_tokens,
                        k=k,
                        mmr_threshold=self.mmr_threshold,
                        relevance_threshold=self.relevance_threshold,
                        layer_score_threshold=self.layer_score_threshold,
                        force_strategy=force_strategy,
                    )
                    
                    seen_ids = set()
                    merged_nodes = []
                    for node in nodes_a + nodes_b:
                        node_id = id(node)
                        if node_id not in seen_ids:
                            seen_ids.add(node_id)
                            merged_nodes.append(node)
                            
                    self.last_nodes = merged_nodes
                    logger.info(f"Comparative retrieval: {len(nodes_a)} nodes for '{sub_queries[0]}' + {len(nodes_b)} nodes for '{sub_queries[1]}'")
                    
                    if not merged_nodes:
                        logger.warning(f"No relevant nodes returned for query: '{query}'")
                        context = "No relevant information found in the document for this query."
                        if self.enable_generation:
                            return self._get_generator().generate(context, query, query_type=query_type)
                        return context
                        
                    context = assemble_context(merged_nodes, max_tokens)
                    if self.enable_generation:
                        return self._get_generator().generate(context, query, query_type=query_type)
                    return context
                else:
                    query_type = "specific"

        # Route through FAISS first if enabled, then fall back to smart retrieve
        if self.use_faiss and self._faiss_index is not None:
            nodes = self._retrieve_collapsed_faiss(query_emb, max_tokens)
            logger.info("Retrieved via FAISS (smart routing not applied)")
        else:
            nodes, strategy = retrieve(
                tree                 = self._tree,
                query                = query,
                query_emb            = query_emb,
                query_type           = query_type,
                max_tokens           = max_tokens,
                k                    = k,
                mmr_threshold        = self.mmr_threshold,
                relevance_threshold  = self.relevance_threshold,
                layer_score_threshold = self.layer_score_threshold,
                force_strategy       = force_strategy,
            )
            logger.info(f"Retrieved via smart routing | strategy={strategy} | nodes={len(nodes)}")

        self.last_nodes = nodes
        # Empty nodes means out-of-scope query was detected
        if not nodes:
            logger.warning(f"No relevant nodes returned for query: '{query}'")
            context = "No relevant information found in the document for this query."
            if self.enable_generation:
                return self._get_generator().generate(context, query)
            return context

        context = assemble_context(nodes, max_tokens)
        if self.enable_generation:
            return self._get_generator().generate(context, query, query_type=query_type)
        return context

    # ── FAISS Integration ──────────────────────────────────────────────────────

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
        _, indices   = self._faiss_index.search(
            query_emb.reshape(1, -1).astype(np.float32), n_candidates
        )
        selected, total_tokens = [], 0
        for idx in indices[0]:
            if idx < 0:
                continue
            node = self._faiss_nodes[idx]
            if total_tokens + node.token_count <= max_tokens:
                selected.append(node)
                total_tokens += node.token_count
        return selected

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _decompose_comparative(self, query: str) -> list[str]:
    
        lowered = query.lower().strip().rstrip("?")

        def _make_sub_query(concept: str) -> str:
            return (
                f"what are {concept}"
                if concept.rstrip().endswith("s")
                else f"what is {concept}"
            )

        def _expand_elliptical(concept_a: str, concept_b: str) -> str:
            """
            If concept_a has fewer words than concept_b, it likely uses elliptical
            phrasing (e.g. 'rapidly' instead of 'rapidly adapting fibers').
            Append the extra trailing words from concept_b to concept_a.
            """
            words_a = concept_a.split()
            words_b = concept_b.split()
            if len(words_a) < len(words_b):
                extra = words_b[len(words_a):]
                return concept_a + " " + " ".join(extra)
            return concept_a

        # ── Primary: prefix + rsplit ───────────────────────────────────────────
        prefix_pattern = r"(?:difference between|compare|contrast|how do|how does)\s+(.+)"
        prefix_match   = re.search(prefix_pattern, lowered)

        if prefix_match:
            remainder = prefix_match.group(1).strip()
            # Remove trailing "differ/differs" for "how do X and Y differ" form
            remainder = re.sub(r"\s+differs?$", "", remainder).strip()
            parts = remainder.rsplit(" and ", 1)

            if len(parts) == 2:
                concept_a = parts[0].strip()
                concept_b = parts[1].strip()
                # Expand elliptical concept_a if it's shorter than concept_b
                concept_a = _expand_elliptical(concept_a, concept_b)
                sub_a = _make_sub_query(concept_a)
                sub_b = _make_sub_query(concept_b)
                logger.debug(
                    f"Comparative decomposition: '{concept_a}' | '{concept_b}' → "
                    f"'{sub_a}' | '{sub_b}'"
                )
                return [sub_a, sub_b]

        # ── Fallback: vs / versus ──────────────────────────────────────────────
        vs_match = re.search(r"(.+?)\s+(?:vs|versus)\s+(.+)", lowered)
        if vs_match:
            concept_a = vs_match.group(1).strip()
            concept_b = vs_match.group(2).strip()
            sub_a     = _make_sub_query(concept_a)
            sub_b     = _make_sub_query(concept_b)
            logger.debug(
                f"Comparative decomposition: '{concept_a}' | '{concept_b}' → "
                f"'{sub_a}' | '{sub_b}'"
            )
            return [sub_a, sub_b]

        logger.debug(f"Comparative decomposition failed for: '{lowered}'")
        return [query]

    def _get_embedder(self):
        if self._embedder is None:
            from embedding.embedder import SBERTEmbedder
            self._embedder = SBERTEmbedder(self.embed_model)
        return self._embedder

    def _get_classifier(self):
        if self._query_classifier is None:
            self._query_classifier = ZeroShotQueryClassifier()
        return self._query_classifier

    def _get_generator(self):
        if self._generator is None:
            self._generator = LLMGenerator(model=self.llm_model)
        return self._generator

    def __repr__(self) -> str:
        return (
            f"RaptorPipeline("
            f"max_tokens={self.max_tokens}, "
            f"model={self.embed_model!r}, "
            f"mmr_threshold={self.mmr_threshold}, "
            f"relevance_threshold={self.relevance_threshold}, "
            f"layer_score_threshold={self.layer_score_threshold}, "
            f"enable_generation={self.enable_generation}, "
            f"tree={repr(self._tree) if self._tree else 'none'})"
        )