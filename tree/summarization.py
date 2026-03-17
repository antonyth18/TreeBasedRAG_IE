import logging
import time
from typing import List

from openai import OpenAI, APIConnectionError, APIError

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = "You are an expert at summarizing technical text concisely while preserving key details. Write a summary that captures the main points and important information from the provided context."
_USER_TEMPLATE = "Write a summary of the following, including as many key details as possible: {context}"
_OLLAMA_BASE_URL = "http://localhost:11434/v1"


# Errors
class SummaryError(Exception):
    pass

class OllamaConnectionError(Exception):
    pass


class LLMSummarizer:
    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = _OLLAMA_BASE_URL,
        max_summary_tokens: int = 200,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        self.model              = model
        self.max_summary_tokens = max_summary_tokens
        self.max_retries        = max_retries
        self.retry_delay        = retry_delay
        self._total_tokens_used = 0
        self._client            = OpenAI(base_url=base_url, api_key="ollama")
        self._verify_connection()

    # Summarisation with retires and error handling. Returns summary string.
    def summarize(self, texts: List[str]) -> str:
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": _USER_TEMPLATE.format(context="\n\n".join(texts))},
        ]

        last_error = None
        delay      = self.retry_delay

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_summary_tokens,
                    temperature=0.0,
                )

                summary = (response.choices[0].message.content or "").strip()
                if not summary:
                    raise SummaryError("Empty Summary returned.")

                if response.usage:
                    self._total_tokens_used += response.usage.total_tokens

                return summary

            except APIConnectionError as e:
                raise OllamaConnectionError(
                    f"Lost connection with Ollama (attempt {attempt}). "
                    f"Is 'ollama serve' still running?\n{e}"
                ) from e

            except APIError as e:
                last_error = e
                logger.warning(f"Ollama API error (attempt {attempt}/{self.max_retries}): {e}. Retrying in {delay:.1f}s...")
                time.sleep(delay)
                delay *= 2

        raise SummaryError(f"Summarisation failed after {self.max_retries} attempts. Last error: {last_error}")

    @property
    def total_tokens_used(self) -> int:
        return self._total_tokens_used

    def _verify_connection(self) -> None:
        try:
            model_ids = [m.id for m in self._client.models.list().data]
            if self.model not in model_ids:
                raise OllamaConnectionError(
                    f"Model '{self.model}' not found in Ollama.\n"
                    f"Available: {model_ids}\n"
                    f"Run: ollama pull {self.model}"
                )
            logger.info(f"Ollama OK  |  model='{self.model}'")
        except APIConnectionError as e:
            raise OllamaConnectionError(
                f"Cannot reach Ollama at {_OLLAMA_BASE_URL}.\n"
                f"Run: ollama serve\n{e}"
            ) from e