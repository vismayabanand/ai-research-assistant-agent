import os, json, re, google.generativeai as genai
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any, List

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

MODEL_NAME = "gemini-1.5-flash-latest"          # or gemini-1.5-pro-latest

_json_block = re.compile(r"\{.*?}", re.DOTALL)

def _extract_json(raw: str) -> Dict[str, Any]:
    m = _json_block.search(raw)
    if not m:
        raise ValueError("LLM response contained no JSON block")
    return json.loads(m.group(0))

def extract_insights(summary: Dict[str, Any]) -> Dict[str, Any]:
    prompt = f"""
You are an academic assistant.

Given this paper summary (JSON):
{json.dumps(summary, indent=2)}

Extract three *lists* with keys exactly:
  • contributions
  • gaps
  • comparisons

Return **ONLY** a valid JSON object — no markdown, no code block, no extra text.
"""
    resp = genai.GenerativeModel(MODEL_NAME).generate_content(prompt)
    new_data = _extract_json(resp.text)
    summary.update(new_data)
    return summary

def extract_insights_batch(summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [extract_insights(s) for s in summaries]