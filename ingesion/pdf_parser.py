import logging
from typing import List

import fitz  # PyMuPDF
import spacy
import tiktoken

logger = logging.getLogger(__name__)

_NLP       = spacy.load("en_core_web_sm")
_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text))


def parse_pdf(pdf_path: str, chunk_tokens: int = 100) -> List[str]:
    # Extract text
    doc = fitz.open(pdf_path)
    full_text = " ".join(page.get_text() for page in doc)
    doc.close()

    if not full_text.strip():
        raise ValueError(f"No extractable text found in '{pdf_path}'")

    # Split into sentences
    sentences = [s.text.strip() for s in _NLP(full_text).sents if s.text.strip()]

    # Pack sentences into chunks of <= chunk_tokens
    chunks, current, current_tokens = [], [], 0

    for sentence in sentences:
        s_tokens = _count_tokens(sentence)

        if s_tokens > chunk_tokens:
            logger.warning(f"Single sentence exceeds {chunk_tokens} tokens ({s_tokens}). Using as its own chunk.")
            if current:
                chunks.append(" ".join(current))
                current, current_tokens = [], 0
            chunks.append(sentence)
            continue

        if current_tokens + s_tokens > chunk_tokens:
            chunks.append(" ".join(current))
            current, current_tokens = [sentence], s_tokens
        else:
            current.append(sentence)
            current_tokens += s_tokens

    if current:
        chunks.append(" ".join(current))

    logger.info(f"parse_pdf: {len(chunks)} chunks from '{pdf_path}'")
    return chunks