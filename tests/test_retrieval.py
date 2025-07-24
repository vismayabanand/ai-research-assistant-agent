# tests/test_retrieval.py
from src.retrieval import fetch_arxiv, fetch_semantic_scholar  # Adjust import based on your project structure 
def test_arxiv_basic():
    papers = fetch_arxiv("quantum computing", max_results=2)
    assert isinstance(papers, list) and len(papers) == 2
    assert all('title' in p for p in papers)

# tests/test_summarizer.py
import os
from src.summarizer import summarize_papers

def test_summarize_structure(monkeypatch):
    # Monkeypatch GENAI to return a fixed JSON string
    class DummyResp:
        choices = [type("c", (), {"message": type("m", (), {"content": '{"introduction":"Intro","methods":"M","conclusion":"C"}'})})]
    import google.generativeai as genai
    monkeypatch.setattr(genai.ChatCompletion, "create", lambda **kw: DummyResp())

    paper = {'title': 'T','authors': ['A'], 'summary': 'S','url':'U'}
    out = summarize_papers(paper)
    assert set(out.keys()) == {"introduction","methods","conclusion"}
