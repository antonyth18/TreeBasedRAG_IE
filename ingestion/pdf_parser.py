"""
PDF parser for extracting chunked text from documents.
Chunking is now section-aware: it prevents mixed-topic chunks from degrading summarization quality in the RAPTOR tree.
"""
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


def _split_into_sections(text: str) -> List[str]:
    lines = text.split('\n')
    is_blank = [not line.strip() for line in lines]
    
    sections = []
    current_section = []
    headings_found = 0
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        is_heading = False
        
        if stripped and len(stripped) < 60:
            if not stripped.endswith('.') and not stripped.endswith(','):
                if not stripped.startswith(('Figure', 'Table')) and not stripped[0].isdigit():
                    if any(c.isalnum() for c in stripped):
                        prev_blank = i > 0 and is_blank[i-1]
                        next_blank = i < len(lines)-1 and is_blank[i+1]
                        if prev_blank or next_blank:
                            is_heading = True
                            headings_found += 1
                            
        if is_heading:
            if current_section:
                sec_text = '\n'.join(current_section).strip()
                if sec_text:
                    sections.append(sec_text)
            current_section = [line]
        else:
            current_section.append(line)
            
    if headings_found == 0:
        return [text]
        
    if current_section:
        sec_text = '\n'.join(current_section).strip()
        if sec_text:
            sections.append(sec_text)
            
    return sections


def _chunk_section(section_text: str, chunk_tokens: int, nlp) -> List[str]:
    sentences = [s.text.strip() for s in nlp(section_text).sents if s.text.strip()]
    chunks = []
    current = []
    current_tokens = 0

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
        
    return chunks


def _extract_body_text(page, margin_threshold: float = 0.75) -> str:
    """
    PyMuPDF extracts text sequentially including margin sidenotes which get interleaved with body text, 
    causing mixed-topic chunks that degrade summarization quality.
    """
    blocks = page.get_text("blocks")
    page_width = page.rect.width
    
    body_blocks = []
    total_blocks = len(blocks)
    
    for block in blocks:
        # block is a tuple of (x0, y0, x1, y1, "text", block_no, block_type)
        x0 = block[0]
        text = block[4]
        block_type = block[6]
        
        # type 0 is text, type 1 is image
        if block_type == 0 and x0 < page_width * margin_threshold and text.strip():
            body_blocks.append(text.strip())
            
    logger.debug(f"Page extracted: {len(body_blocks)} body blocks, {total_blocks - len(body_blocks)} margin blocks filtered")
    return "\n\n".join(body_blocks)


def parse_pdf(pdf_path: str, chunk_tokens: int = 100, min_chunk_tokens: int = 40) -> List[str]:
    # Extract text
    doc = fitz.open(pdf_path)
    full_text = "\n\n".join(_extract_body_text(page) for page in doc)
    doc.close()

    if not full_text.strip():
        raise ValueError(f"No extractable text found in '{pdf_path}'")

    sections = _split_into_sections(full_text)
    
    all_chunks = []
    total_sections = len(sections)
    
    for i, sec_text in enumerate(sections):
        sec_chunks = _chunk_section(sec_text, chunk_tokens, _NLP)
        
        valid_chunks = [c for c in sec_chunks if _count_tokens(c) >= min_chunk_tokens]
        dropped = len(sec_chunks) - len(valid_chunks)
        
        if dropped > 0:
            logger.debug(f"Section {i+1}/{total_sections}: dropped {dropped} micro-chunks below {min_chunk_tokens} tokens")
            
        logger.debug(f"Section {i+1}/{total_sections}: {len(valid_chunks)} chunks (from {len(sec_chunks)}) from {len(sec_text)} chars")
        all_chunks.extend(valid_chunks)

    logger.debug(f"parse_pdf: {total_sections} sections detected, {len(all_chunks)} total chunks produced")
    logger.info(f"parse_pdf: {len(all_chunks)} chunks from '{pdf_path}' ({total_sections} sections, min_chunk_tokens={min_chunk_tokens})")
    return all_chunks