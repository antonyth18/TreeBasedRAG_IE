import logging
from typing import List

import tiktoken

from node import RaptorNode

logger = logging.getLogger(__name__)

_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text))


def assemble_context(
    nodes: List[RaptorNode],
    max_tokens: int,
    separator: str = "\n\n---\n\n",
) -> str:
    if not nodes:
        return ""

    sep_tokens   = _count_tokens(separator)
    parts        = []
    total_tokens = 0

    for node in nodes:
        seg_tokens = _count_tokens(node.text)
        extra      = sep_tokens if parts else 0

        if total_tokens + seg_tokens + extra <= max_tokens:
            parts.append(node.text)
            total_tokens += seg_tokens + extra

    logger.debug(f"assemble_context: {len(parts)}/{len(nodes)} nodes ({total_tokens}/{max_tokens} tokens)")
    return separator.join(parts)