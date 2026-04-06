import logging
from typing import List

from ddgs import DDGS

logger = logging.getLogger(__name__)


class WebSearcher:

    def __init__(self, n_results: int = 3, timeout: int = 10):
        self.n_results = n_results
        self.timeout = timeout

    def search(self, query: str) -> List[str]:
        """
        Returns a list of text snippets from web search.
        Each snippet is: "Title. Body snippet. Source: URL"
        Returns [] on failure, so the pipeline degrades gracefully
        if web search is unavailable.
        """
        try:
            with DDGS(timeout=self.timeout) as ddgs:
                results = ddgs.text(
                    query,
                    max_results=self.n_results,
                    safesearch="moderate",
                )

            snippets = []
            for r in results:
                snippet = f"{r.get('title', '')}. {r.get('body', '')} (Source: {r.get('href', '')})"
                snippets.append(snippet.strip())

            logger.debug(f"WebSearcher: {len(snippets)} results for '{query[:60]}'")
            return snippets
        except Exception as e:
            logger.warning(f"Web search failed for '{query[:60]}': {e}")
            return []
