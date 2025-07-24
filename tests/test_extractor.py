import json
import os
import pytest
from src.extractor import extract_insights

class DummyResp:
    choices = [type("c", (), {"message": type("m", (), {
        "content": json.dumps({
            "contributions": ["C1"],
            "gaps": ["G1"],
            "comparisons": ["Comp1"]
        })
    })})]

def test_extract_insights(monkeypatch):
    # Mock GenAI call
    import google.generativeai as genai
    monkeypatch.setenv("GOOGLE_API_KEY", "test")
    monkeypatch.setattr(genai.ChatCompletion, "create", lambda **kw: DummyResp())

    summary = {"introduction": "I", "methods": "M", "conclusion": "C"}
    insights = extract_insights(summary)

    assert set(insights.keys()) == {"contributions", "gaps", "comparisons"}
    assert isinstance(insights["contributions"], list)
    assert insights["contributions"] == ["C1"]
