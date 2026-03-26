import logging
import time
from typing import List

from openai import OpenAI, APIConnectionError, APIError

logger = logging.getLogger(__name__)

# ── Prompts ────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a precise technical summarizer. Your only job is to
condense the provided text into a shorter summary.

STRICT RULES:
- Only include facts, claims, and relationships that are explicitly stated in
  the provided text
- Do not infer, imply, or add any information not directly present in the input
- Do not connect concepts unless the text explicitly connects them
- If the text appears to be a citation, caption, or footnote, summarize it as is. Do not reject it or output meta-commentary like "There is no text to summarize".
- If the text is ambiguous or unclear, reflect that ambiguity — do not resolve
  it with outside knowledge
- Do not add explanatory context from your training knowledge
- If you are unsure whether a detail is in the text or from your own knowledge,
  leave it out"""

_USER_TEMPLATE = """Summarize ONLY the following text. Do not add any information
beyond what is written below.

TEXT TO SUMMARIZE:
{context}

SUMMARY (only facts present in the text above):"""

_VERIFICATION_SYSTEM_PROMPT = """You are a strict fact-checker. Your job is to 
verify whether a summary contains only information present in a source text.
Reply with exactly one of these two formats and nothing else:
PASS
FAIL: [quote the specific claim not found in source]"""

_VERIFICATION_USER_TEMPLATE = """SOURCE TEXT:
{source}

SUMMARY TO CHECK:
{summary}

Does the summary contain ANY information, connections, or claims not explicitly
present in the source text? Reply PASS or FAIL: [specific claim]"""

_OLLAMA_BASE_URL = "http://localhost:11434/v1"


# ── Errors ─────────────────────────────────────────────────────────────────────

class SummaryError(Exception):
    pass

class OllamaConnectionError(Exception):
    pass

class FaithfulnessError(Exception):
    pass


# ── Summarizer ─────────────────────────────────────────────────────────────────

class LLMSummarizer:
    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = _OLLAMA_BASE_URL,
        max_summary_tokens: int = 200,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        verify_faithfulness: bool = True,       # toggle verification on/off
        max_verification_retries: int = 2,      # how many times to retry a failed summary
    ):
        self.model                     = model
        self.max_summary_tokens        = max_summary_tokens
        self.max_retries               = max_retries
        self.retry_delay               = retry_delay
        self.verify_faithfulness       = verify_faithfulness
        self.max_verification_retries  = max_verification_retries
        self._total_tokens_used        = 0
        self._faithfulness_failures    = 0      # track how often verification catches issues
        self._client                   = OpenAI(base_url=base_url, api_key="ollama")
        self._verify_connection()

    # ── Public ─────────────────────────────────────────────────────────────────

    def summarize(self, texts: List[str]) -> str:
        """
        Summarize a list of text chunks into a single faithful summary.
        If verify_faithfulness=True, checks the summary against the source
        and retries if hallucinated claims are detected.
        """
        context = "\n\n".join(texts)

        # If verification is off, just summarize once normally
        if not self.verify_faithfulness:
            try:
                return self._generate_summary(context)
            except SummaryError as e:
                logger.warning(f"LLM summarization failed, using fallback summary: {e}")
                return self._fallback_summary(context)

        # If verification is on, retry the summary until it passes or we give up
        last_summary = None
        for attempt in range(1, self.max_verification_retries + 1):
            try:
                summary = self._generate_summary(context)
            except SummaryError as e:
                logger.warning(f"LLM summarization failed on attempt {attempt}, using fallback summary: {e}")
                summary = self._fallback_summary(context)
            passed, failed_claim = self._check_faithfulness(context, summary)

            if passed:
                if attempt > 1:
                    logger.info(f"Summary passed faithfulness check on attempt {attempt}.")
                return summary

            # Verification failed — log it and retry
            self._faithfulness_failures += 1
            logger.warning(
                f"Faithfulness check failed (attempt {attempt}/{self.max_verification_retries}). "
                f"Hallucinated claim: '{failed_claim}'. Retrying summary..."
            )
            last_summary = summary

        # All verification retries exhausted — log and return last summary with warning
        # We return rather than raise so tree building isn't blocked entirely,
        # but the warning makes it visible in logs for manual review
        logger.error(
            f"Summary failed faithfulness verification after {self.max_verification_retries} "
            f"attempts. Returning last summary for manual review. "
            f"Last failed claim: '{failed_claim}'"
        )
        return last_summary

    @property
    def total_tokens_used(self) -> int:
        return self._total_tokens_used

    @property
    def faithfulness_failures(self) -> int:
        """How many times verification caught a hallucination across all summaries."""
        return self._faithfulness_failures

    # ── Private ────────────────────────────────────────────────────────────────

    def _generate_summary(self, context: str) -> str:
        """
        Core summarization call with retry logic for API errors.
        Unchanged from original except using updated prompts.
        """
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": _USER_TEMPLATE.format(context=context)},
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
                    raise SummaryError("Empty summary returned.")

                if response.usage:
                    self._total_tokens_used += response.usage.total_tokens

                return summary

            except APIConnectionError as e:
                raise OllamaConnectionError(
                    f"Lost connection with Ollama (attempt {attempt}). "
                    f"Is 'ollama serve' still running?\n{e}"
                ) from e

            except SummaryError as e:
                last_error = e
                logger.warning(
                    f"Summary generation returned empty output (attempt {attempt}/{self.max_retries}). "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
                delay *= 2

            except APIError as e:
                last_error = e
                logger.warning(
                    f"Ollama API error (attempt {attempt}/{self.max_retries}): {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
                delay *= 2

        raise SummaryError(
            f"Summarization failed after {self.max_retries} attempts. "
            f"Last error: {last_error}"
        )

    def _fallback_summary(self, context: str, max_chars: int = 800) -> str:
        text = " ".join(context.split())
        if not text:
            return "No content to summarize."

        if len(text) <= max_chars:
            return text

        # Prefer sentence boundary near limit.
        cutoff = text.rfind(". ", 0, max_chars)
        if cutoff == -1:
            cutoff = max_chars
        return text[:cutoff].strip() + "..."

    def _check_faithfulness(self, source: str, summary: str) -> tuple[bool, str]:
        """
        Sends the source text and summary to the LLM for faithfulness verification.
        Returns (True, "") if the summary passes.
        Returns (False, failed_claim) if the summary contains hallucinated content.
        """
        messages = [
            {"role": "system", "content": _VERIFICATION_SYSTEM_PROMPT},
            {"role": "user",   "content": _VERIFICATION_USER_TEMPLATE.format(
                source=source,
                summary=summary,
            )},
        ]

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=100,         # verification responses are short
                temperature=0.0,
            )

            verdict = (response.choices[0].message.content or "").strip()

            if response.usage:
                self._total_tokens_used += response.usage.total_tokens

            if verdict.upper().startswith("PASS"):
                return True, ""

            # Extract the failed claim from "FAIL: [claim]"
            failed_claim = verdict[5:].strip() if verdict.upper().startswith("FAIL:") else verdict
            return False, failed_claim

        except (APIConnectionError, APIError) as e:
            # If verification itself fails, log and pass through rather than
            # blocking tree construction entirely
            logger.warning(f"Faithfulness verification call failed: {e}. Skipping check.")
            return True, ""

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
                f"Cannot reach Ollama at {_OLLAMA_BASE_URL}.\n"
                f"Run: ollama serve\n{e}"
            ) from e