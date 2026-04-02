import logging
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class SBERTEmbedder:
    MODEL = "BAAI/bge-small-en-v1.5"

    def __init__(self, model_name: str = MODEL, quantize: bool = True):
        self.MODEL  = model_name
        self._model = SentenceTransformer(model_name)
        self.quantized = False

        if quantize:
            self._maybe_quantize_dynamic()

        logger.info(f"SBERTEmbedder loaded: '{model_name}' | quantized={self.quantized}")

    def _maybe_quantize_dynamic(self) -> None:
        """
        Apply dynamic int8 quantization for CPU inference to reduce memory usage.
        Skip when CUDA is active because this optimization targets CPU-only paths.
        """
        try:
            import torch
            import torch.nn as nn

            if torch.cuda.is_available():
                logger.info("Skipping dynamic int8 quantization because CUDA is available.")
                return

            transformer_module = self._model._first_module()
            auto_model = getattr(transformer_module, "auto_model", None)
            if auto_model is None:
                logger.warning("Could not locate transformer auto_model; quantization skipped.")
                return

            auto_model.eval()
            transformer_module.auto_model = torch.quantization.quantize_dynamic(
                auto_model,
                {nn.Linear},
                dtype=torch.qint8,
            )
            self.quantized = True
        except Exception as exc:
            logger.warning(f"Dynamic quantization skipped due to: {exc}")

    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,   # L2-normalise: cosine sim = dot product
            show_progress_bar=False,
        )
        return embeddings.astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        return self.encode([query])[0]