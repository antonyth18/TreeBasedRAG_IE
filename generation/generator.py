import logging
import time

from openai import OpenAI, APIConnectionError, APIError
from backend.core.config import settings

logger = logging.getLogger(__name__)

class GenerationError(Exception):
    pass

class OllamaConnectionError(Exception):
    pass

# ── Prompts ──────────────────────────────────────────────────────────────────
# Single prompt rules conflict when governing both strict factual queries and 
# inferential 'why/how' queries simultaneously. We split them into two variants.

_SYSTEM_PROMPT_FACTUAL = """Answer questions strictly based on the provided context.
Be concise and direct — answer the question asked without restating it.
If the context does not contain information relevant to the question, respond with exactly: "The document does not contain information about this topic."
Never use outside knowledge, never speculate, never infer beyond what the context states.
If the context is partially relevant but incomplete, answer what can be answered and state what is missing.
Do not mention the context, chunks, or retrieval system in the answer — just answer naturally.

- Include all relevant facts from the context that directly answer the question — do not truncate a definition or explanation if the context contains more detail

- When describing multiple distinct entities, keep each entity's properties strictly separate — do not apply properties of one entity to another
- If you are uncertain which entity a property belongs to, omit it rather than risk misattribution

- Do not add mechanistic context, anatomical pathways, or physiological explanations from outside the provided text — even if they seem obviously true
- If the context describes a process partially, describe only what the context states — do not complete the mechanism from your own knowledge
- For questions asking about types, categories, or lists of things, 
- read ALL of the provided context before answering — do not stop 
- after finding the first few items. Your answer must account for
- every distinct item mentioned across all context chunks.
"""

_SYSTEM_PROMPT_INFERENTIAL = """Answer questions strictly based on the provided context.
Be concise and direct — answer the question asked without restating it.
If the context does not contain information relevant to the question, respond with exactly: "The document does not contain information about this topic."
Never use outside knowledge, never speculate, never infer beyond what the context states.
If the context is partially relevant but incomplete, answer what can be answered and state what is missing.
Do not mention the context, chunks, or retrieval system in the answer — just answer naturally.

- Include all relevant facts from the context that directly answer the question — do not truncate a definition or explanation if the context contains more detail

- When describing multiple distinct entities, keep each entity's properties strictly separate — do not apply properties of one entity to another
- If you are uncertain which entity a property belongs to, omit it rather than risk misattribution

- Do not add mechanistic context, anatomical pathways, or physiological explanations from outside the provided text — even if they seem obviously true
- If the context describes a process partially, describe only what the context states — do not complete the mechanism from your own knowledge

- Connecting two facts that are both explicitly stated in the context 
  to answer a 'why' or 'how' question is permitted — this is reasoning 
  from stated facts, not outside knowledge. For example, if the context 
  states that X performs function Y, and also states that X decreases 
  with age, you may conclude that Y is affected by aging.
- Do not extend this reasoning beyond one inferential step — do not 
  chain multiple inferences or add consequences not implied by the 
  stated facts

- If the question's premise assumes a connection that the context does 
  not support even after attempting one inferential step from stated 
  facts, state that the document does not establish this connection"""

_USER_TEMPLATE = """Context:
{context}

Question: {question}

Answer:"""

class LLMGenerator:
    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434/v1",
        max_answer_tokens: int = 800,
        temperature: float = 0.0,
        num_ctx: int = settings.OLLAMA_NUM_CTX,
    ):
        self.model = model
        self.base_url = base_url
        self.max_answer_tokens = max_answer_tokens
        self.temperature = temperature
        self.num_ctx = num_ctx
        self._total_tokens_used = 0
        
        self.max_retries = 3
        self.retry_delay = 2.0
        
        self._client = OpenAI(base_url=self.base_url, api_key="ollama")
        self._verify_connection()

    @property
    def total_tokens_used(self) -> int:
        return self._total_tokens_used

    def generate(self, context: str, question: str, query_type: str = "specific") -> str:
        if not context or context.strip() == "No relevant information found in the document for this query.":
            return "The document does not contain information about this topic."
            
        logger.info(f"Generating answer for question: '{question[:60]}...'")
        
        if query_type in ("specific",) and any(
            question.lower().startswith(w) 
            for w in ("why", "how", "what causes", "what role", "what happens")
        ):
            system_prompt = _SYSTEM_PROMPT_INFERENTIAL
        else:
            system_prompt = _SYSTEM_PROMPT_FACTUAL

        logger.debug(f"Using {'inferential' if system_prompt == _SYSTEM_PROMPT_INFERENTIAL else 'factual'} prompt for query_type='{query_type}'")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": _USER_TEMPLATE.format(context=context, question=question)},
        ]

        last_error = None
        delay = self.retry_delay

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_answer_tokens,
                    temperature=self.temperature,
                    extra_body={"options": {"num_ctx": self.num_ctx}},
                )

                answer = (response.choices[0].message.content or "").strip()
                if not answer:
                    raise GenerationError("Empty answer returned.")

                if response.usage:
                    self._total_tokens_used += response.usage.total_tokens
                    logger.debug(f"Generation token usage: {response.usage.total_tokens}")

                return answer

            except APIConnectionError as e:
                raise OllamaConnectionError(
                    f"Lost connection with Ollama (attempt {attempt}). "
                    f"Is 'ollama serve' still running?\n{e}"
                ) from e

            except APIError as e:
                last_error = e
                logger.warning(
                    f"Ollama API error (attempt {attempt}/{self.max_retries}): {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
                delay *= 2

        raise GenerationError(
            f"Generation failed after {self.max_retries} attempts. "
            f"Last error: {last_error}"
        )

    def _verify_connection(self) -> None:
        try:
            model_ids = [m.id for m in self._client.models.list().data]
            resolved_model = None
            if self.model in model_ids:
                resolved_model = self.model
            else:
                tagged_matches = [mid for mid in model_ids if mid.startswith(f"{self.model}:")]
                if tagged_matches:
                    resolved_model = tagged_matches[0]

            if resolved_model is None:
                raise OllamaConnectionError(
                    f"Model '{self.model}' not found in Ollama.\n"
                    f"Available: {model_ids}\n"
                    f"Run: ollama pull {self.model}"
                )
            if resolved_model != self.model:
                logger.info(f"Resolved Ollama model '{self.model}' -> '{resolved_model}'")
                self.model = resolved_model

            logger.info(f"Ollama OK  |  model='{self.model}'")
        except APIConnectionError as e:
            raise OllamaConnectionError(
                f"Cannot reach Ollama at {self.base_url}.\n"
                f"Run: ollama serve\n{e}"
            ) from e
