import logging
import time

from openai import OpenAI, APIConnectionError, APIError

logger = logging.getLogger(__name__)


class GenerationError(Exception):
    pass


class OllamaConnectionError(Exception):
    pass


# ── Prompts ───────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT_FACTUAL = """/no_think
You are a precise question-answering assistant. Answer using only the provided context.

RULES:
- Answer directly. Do not restate the question or reference the context.
- If the context lacks relevant information, respond exactly: "The document does not contain information about this topic."
- Never use outside knowledge. Never speculate beyond what the context states.
- If context is partially relevant, answer what can be answered and state what is missing.

COMPLETENESS — THIS IS CRITICAL:
- If the question asks for a list, numbered items, or categories (e.g. "what are the four laws", "list all the strategies"), you must read every context chunk before answering.
- Your answer must include every distinct item mentioned across all chunks. Do not stop after finding the first few.

ACCURACY:
- Keep each entity's properties strictly separate. Never apply properties of one entity to another.
- If uncertain which entity a property belongs to, omit it rather than guess.

REASONING:
- Connecting two facts both explicitly stated in the context to answer a "why" or "how" question is permitted.
- Do not chain more than one inferential step.
- If the question assumes a connection the context does not support, state that the document does not establish this connection."""

_SYSTEM_PROMPT_INFERENTIAL = """You are a precise question-answering assistant. Answer using only the provided context.

RULES:
- Answer directly. Do not restate the question or reference the context.
- If the context lacks relevant information, respond exactly: "The document does not contain information about this topic."
- Never use outside knowledge. Never speculate beyond what the context states.
- If context is partially relevant, answer what can be answered and state what is missing.

COMPLETENESS — THIS IS CRITICAL:
- If the question asks for a list, numbered items, or categories, you must read every context chunk before answering.
- Your answer must include every distinct item mentioned across all chunks. Do not stop after finding the first few.

ACCURACY:
- Keep each entity's properties strictly separate. Never apply properties of one entity to another.
- If uncertain which entity a property belongs to, omit it rather than guess.

REASONING:
- You may connect two facts both explicitly stated in the context to answer a "why", "how", or application question — this is reasoning from stated facts, not outside knowledge.
- Do not chain more than one inferential step.
- Do not add consequences or mechanisms not directly implied by what the context states.
- If the question's premise assumes a connection the context does not support even after one inferential step, state that the document does not establish this connection."""

_USER_TEMPLATE = """Context:
{context}

Question: {question}

Answer:"""

_INFERENTIAL_TRIGGERS = (
    "why", "how", "what causes", "what role", "what happens",
    "explain", "describe", "design", "compare", "contrast", "interact",
)


class LLMGenerator:
    def __init__(
        self,
        model: str = "qwen3:8b",
        base_url: str = "http://localhost:11434/v1",
        max_answer_tokens: int = 800,
        num_ctx: int = 8192,
    ):
        self.model             = model
        self.base_url          = base_url
        self.max_answer_tokens = max_answer_tokens
        self.num_ctx           = num_ctx
        self.max_retries       = 3
        self.retry_delay       = 2.0
        self._total_tokens_used = 0
        self._client           = OpenAI(base_url=base_url, api_key="ollama")
        self._verify_connection()

    @property
    def total_tokens_used(self) -> int:
        return self._total_tokens_used

    def generate(self, context: str, question: str, query_type: str = "specific") -> str:
        if not context or context.strip() == "No relevant information found in the document for this query.":
            return "The document does not contain information about this topic."

        use_inferential = (
            query_type in ("broad", "comparative") or
            any(question.lower().startswith(w) for w in _INFERENTIAL_TRIGGERS)
        )

        system_prompt = _SYSTEM_PROMPT_INFERENTIAL if use_inferential else _SYSTEM_PROMPT_FACTUAL
        temperature   = 0.1 if use_inferential else 0.0

        logger.info(f"Generating | query_type='{query_type}' | mode='{'inferential' if use_inferential else 'factual'}' | question='{question[:60]}'")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": _USER_TEMPLATE.format(context=context, question=question)},
        ]

        last_error = None
        delay      = self.retry_delay

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_answer_tokens,
                    temperature=temperature,
                    extra_body={"options": {"num_ctx": self.num_ctx}},
                )

                answer = (response.choices[0].message.content or "").strip()
                if not answer:
                    raise GenerationError("Empty answer returned.")

                if response.usage:
                    self._total_tokens_used += response.usage.total_tokens

                return answer

            except APIConnectionError as e:
                raise OllamaConnectionError(
                    f"Lost connection to Ollama (attempt {attempt}). "
                    f"Is 'ollama serve' still running?\n{e}"
                ) from e

            except APIError as e:
                last_error = e
                logger.warning(f"Ollama API error (attempt {attempt}/{self.max_retries}): {e}. Retrying in {delay:.1f}s...")
                time.sleep(delay)
                delay *= 2

        raise GenerationError(f"Generation failed after {self.max_retries} attempts. Last error: {last_error}")

    def _verify_connection(self) -> None:
        try:
            model_ids = [m.id for m in self._client.models.list().data]

            resolved = None
            if self.model in model_ids:
                resolved = self.model
            else:
                tagged = [m for m in model_ids if m.startswith(f"{self.model}:")]
                if tagged:
                    resolved = tagged[0]

            if resolved is None:
                raise OllamaConnectionError(
                    f"Model '{self.model}' not found in Ollama.\n"
                    f"Available: {model_ids}\n"
                    f"Run: ollama pull {self.model}"
                )

            if resolved != self.model:
                logger.info(f"Resolved model '{self.model}' → '{resolved}'")
                self.model = resolved

            logger.info(f"Ollama OK  |  model='{self.model}'")

        except APIConnectionError as e:
            raise OllamaConnectionError(
                f"Cannot reach Ollama at {self.base_url}.\n"
                f"Run: ollama serve\n{e}"
            ) from e