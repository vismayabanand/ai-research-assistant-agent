# src/summarizer.py
"""Generate structured (JSON) summaries for research papers using Gemini 1.5.

This module:
  • Loads the Google API key from the project‑root .env
  • Calls the Gemini 1.5 model to summarise a paper into introduction / methods / conclusion
  • Preserves identity fields (title, url, authors) so later stages—insight extraction,
    reading‑plan, RAG—don’t lose them.
  • Provides a helper to parse JSON even if the model wraps it in prose or fences.

Functions exported:
    summarize_paper(paper: dict)  -> dict
    summarize_papers(papers)      -> list[dict] | dict
"""

from __future__ import annotations

import os
import json
import re
from pathlib import Path
from typing import Dict, Any, List

from dotenv import load_dotenv
import google.generativeai as genai

# ── Environment ------------------------------------------------------------
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)

GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]  # raises KeyError if missing

# Configure Gemini once for the whole module
genai.configure(api_key=GOOGLE_API_KEY)

# ── Model / generation settings -------------------------------------------
MODEL_NAME = "gemini-1.5-flash-latest"  # switch to "gemini-1.5-pro-latest" if desired
MAX_TOKENS = 256

# ── Helpers ----------------------------------------------------------------
_json_block = re.compile(r"\{[\s\S]*?\}")  # non‑greedy match of first {...}

def _extract_json(raw: str) -> Dict[str, Any]:
    """Extract first JSON object from LLM response (ignores extra prose or fences)."""
    m = _json_block.search(raw)
    if not m:
        raise ValueError("LLM response contained no JSON object")
    return json.loads(m.group(0))

# ── Core API ----------------------------------------------------------------

def summarize_paper(paper: Dict[str, Any]) -> Dict[str, Any]:
    """Return a structured JSON summary for a single paper.

    The returned dict always contains:
      title, url, authors, introduction, methods, conclusion
    """

    prompt = f"""
You are an academic assistant.

Return ONLY valid JSON with keys: title, url, authors, introduction, methods, conclusion.

Title: {paper['title']}
Authors: {', '.join(paper['authors'])}
URL: {paper['url']}
Abstract: {paper['summary']}
"""

    model = genai.GenerativeModel(MODEL_NAME)
    response = model.generate_content(
        prompt,
        generation_config={"max_output_tokens": MAX_TOKENS}
    )

    data = _extract_json(response.text)

    # Safety net: ensure identity fields are present
    data.setdefault("title", paper["title"])
    data.setdefault("url", paper["url"])
    data.setdefault("authors", paper["authors"])
    return data


def summarize_papers(papers: List[Dict[str, Any]] | Dict[str, Any]):
    """Map `summarize_paper` over a list or single dict for convenience."""
    if isinstance(papers, dict):
        return summarize_paper(papers)
    return [summarize_paper(p) for p in papers]


__all__ = [
    "summarize_paper",
    "summarize_papers",
    "MODEL_NAME",
]
