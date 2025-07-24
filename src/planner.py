# src/planner.py
import re
from typing import List, Dict, Any
import google.generativeai as genai

def sort_papers_by_insight(insights_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def score(item: Dict[str, Any]):
        contribs = item.get("contributions", []) or []
        gaps     = item.get("gaps", []) or []
        return (-len(contribs), len(gaps), item.get("title", ""))

    return sorted(insights_list, key=score)

#NEW LLM-POWERED PLANNER
def plan_reading_with_llm(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    print("  Generating an intelligent reading plan with an LLM...")
    prompt_papers = [f"Title: {p.get('title', 'Untitled')}\nSummary: {p.get('summary', '')}" for p in papers]
    prompt_context = "\n\n---\n\n".join(prompt_papers)
    prompt = f"""
You are an academic advisor. Based on the following research paper titles and summaries, create a logical reading plan for a student new to the topic.

Order the papers starting with foundational or survey papers, then move to more specific applications or advanced topics.

Return ONLY a numbered list of the paper titles in the new, logical order. Do not add any commentary or explanation.

Here are the papers:
{prompt_context}
"""
    try:
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        response = model.generate_content(prompt)
        ordered_titles = [line.strip() for line in re.findall(r'^\d+\.\s*(.*)', response.text, re.MULTILINE)]
        if not ordered_titles:
            print("    LLM planner did not return a valid list. Returning original order.")
            return papers
        paper_map = {p.get('title'): p for p in papers}
        # Reorder the original list of papers based on the LLM's suggested title order
        ordered_papers = [paper_map[title] for title in ordered_titles if title in paper_map]
        missing_papers = [p for p in papers if p.get('title') not in ordered_titles]
        return ordered_papers + missing_papers

    except Exception as e:
        print(f"    Error during LLM planning: {e}. Returning original order.")
        return papers