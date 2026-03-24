"""
Query Classifier Module

This module previously used an embedding-based approach (K-Nearest Neighbors using 
cosine similarity to pre-embedded examples) to classify user queries. 
It has been upgraded to a zero-shot NLI (Natural Language Inference) classifier 
using HuggingFace Transformers. The NLI approach allows for robust classification 
without requiring a hardcoded domain-specific example set, and scales better to 
unforeseen query structures.

The older `EmbeddingQueryClassifier` is kept for backward compatibility but is marked as deprecated.
"""

import logging
import re
from typing import Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)

# ── Zero-Shot NLI Classifier ───────────────────────────────────────────────────

_CANDIDATE_LABELS = {
    "definitional question about what something is": "definitional",
    "specific factual question about a single concept": "specific",
    "comparative question about differences or similarities between two things": "comparative",
    "broad overview or summary request covering multiple concepts": "broad"
}

# ── Structured Regex Rules ─────────────────────────────────────────────────────
# Rules are ordered from most-to-least specific. The first rule to match the query wins.
_STRUCTURAL_RULES = [
    # 1. Comparative — must come BEFORE definitional
    (re.compile(r"what is the difference between|how do .+ and .+ differ|"
                r"compare .+ and|contrast .+ and|.+\s+vs\s+.+|.+\s+versus\s+.+", 
                re.IGNORECASE), "comparative"),

    # 2. Broad — explicit plurality signals
    (re.compile(r"what are (all|the different|the main|the various|the types of)|"
                r"list (all|the)|describe all|explain all|overview|summari[sz]e|"
                r"advantages and disadvantages|pros and cons|benefits and drawbacks|"
r"compare .+ types|differences between .+ types"
                r"types of|different types", re.IGNORECASE), "broad"),

    # 3. Definitional — only after comparative and broad are checked
    (re.compile(r"^(what is|what are|define|explain what)", re.IGNORECASE), "definitional"),

    # 4. Specific
    (re.compile(r"^(why|how does|how do|what causes|what role|what happens|"
                r"how is|how are|what is the role|what is the function|"
                r"what is the density|how many|when does)", re.IGNORECASE), "specific"),
]

def _classify_by_structure(query: str) -> Optional[str]:
    """
    Attempts to classify a query based on syntactic structure using regex rules.
    Returns the mapped category label, or None if the query is ambiguous.
    """
    cleaned_query = query.lower().strip().rstrip("?")
    
    for pattern, label in _STRUCTURAL_RULES:
        if pattern.search(cleaned_query):
            return label
            
    return None

class ZeroShotQueryClassifier:
    """
    Classifies queries locally using a zero-shot NLI transformers pipeline.
    """
    def __init__(self):
        self._classifier = None

    def _get_pipeline(self):
        if self._classifier is None:
            model_name = "cross-encoder/nli-MiniLM2-L6-H768"
            logger.info(f"Loading zero-shot NLI model '{model_name}' (this happens only once)...")
            from transformers import pipeline
            self._classifier = pipeline("zero-shot-classification", model=model_name)
        return self._classifier

    def classify(self, query: str) -> str:
        """
        Classifies the query and returns the predicted short label.
        """
        short_label, _ = self.classify_with_confidence(query)
        return short_label

    def classify_with_confidence(self, query: str) -> Tuple[str, float]:
        """
        Evaluates the query against structural pattern rules first. 
        If no rules match, falls back to the zero-shot NLI classifier.
        Returns the top short label and its confidence score.
        """
        # 1. Try structural rules first (Clear-cut cases)
        structural_match = _classify_by_structure(query)
        if structural_match:
            logger.debug(f"Structural rule matched '{query}' → '{structural_match}' (confidence=1.0)")
            return structural_match, 1.0
            
        # 2. Fallback to NLI classifier (Ambiguous cases)
        pipe = self._get_pipeline()
        candidate_descriptions = list(_CANDIDATE_LABELS.keys())
        
        result = pipe(query, candidate_descriptions)
        
        best_desc = result["labels"][0]
        confidence = float(result["scores"][0])
        short_label = _CANDIDATE_LABELS[best_desc]
        
        logger.debug(
            f"ZeroShot classify '{query}': "
            f"won '{best_desc}' -> '{short_label}' "
            f"(score={confidence:.3f})"
        )
        return short_label, confidence


# ── Deprecated Embedding Classifier ────────────────────────────────────────────

LABELED_EXAMPLES = [
    # Specific
    ("what is the exact value of the parameter?", "specific"),
    ("who is the author of this paper?", "specific"),
    ("when was the product launched?", "specific"),
    ("how many people were injured?", "specific"),
    ("what temperature does water boil at?", "specific"),
    ("what is the precise metric used?", "specific"),
    
    # Broad
    ("what are the main topics discussed?", "broad"),
    ("can you summarize the document?", "broad"),
    ("give me an overview of the findings", "broad"),
    ("explain everything in detail", "broad"),
    ("what is the general theme?", "broad"),
    ("describe the overall process", "broad"),
    
    # Comparative
    ("compare the two approaches", "comparative"),
    ("what is the difference between these concepts?", "comparative"),
    ("how does this differ from that?", "comparative"),
    ("contrast the old and new methods", "comparative"),
    ("is method A better than method B?", "comparative"),
    ("what are the advantages of X over Y?", "comparative"),
    
    # Definitional
    ("define the term conceptually", "definitional"),
    ("what is the meaning of this word?", "definitional"),
    ("what does the acronym stand for?", "definitional"),
    ("what is the definition of the concept?", "definitional"),
    ("explain what this term means", "definitional"),
    ("what exactly is a node?", "definitional"),
]

class EmbeddingQueryClassifier:
    """
    [DEPRECATED] Classifies queries locally using nearest neighbors (K=1) 
    against a small labeled example set.
    
    Use `ZeroShotQueryClassifier` instead.
    """
    def __init__(self, embedder):
        self._embedder = embedder
        self._labels = [label for _, label in LABELED_EXAMPLES]
        
        # Pre-embed the examples at init time
        texts = [text for text, _ in LABELED_EXAMPLES]
        self._example_embeddings = self._embedder.encode(texts)
        logger.debug(f"Initialized EmbeddingQueryClassifier with {len(LABELED_EXAMPLES)} labeled examples.")

    def classify(self, query: str) -> str:
        """
        Classifies the query and returns the predicted label.
        """
        label, _ = self.classify_with_confidence(query)
        return label

    def classify_with_confidence(self, query: str) -> Tuple[str, float]:
        """
        Encodes the query and computes dot product similarity against stored examples.
        Returns the top label and its similarity score.
        """
        query_emb = self._embedder.encode_query(query)
        
        # Compute cosine similarities via dot product (embeddings are L2-normalized)
        similarities = self._example_embeddings @ query_emb
        best_idx = np.argmax(similarities)
        
        top_label = self._labels[best_idx]
        confidence = float(similarities[best_idx])
        
        logger.debug(f"Query classified as '{top_label}' with confidence {confidence:.3f}")
        return top_label, confidence
