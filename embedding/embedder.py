import logging
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class SBERTEmbedder:
    MODEL = "BAAI/bge-m3"

    def __init__(self, model_name: str = MODEL):
        self.MODEL  = model_name
        self._model = SentenceTransformer(model_name)
        logger.info(f"SBERTEmbedder loaded: '{model_name}'")

    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,   # L2-normalise: cosine sim = dot product
            show_progress_bar=False,
        )
        return embeddings.astype(np.float32)   # shape (N, 768)

    def encode_query(self, query: str) -> np.ndarray:
        return self.encode([query])[0]   # shape (768,)