import os
from typing import List, Dict, Any, TypedDict
from langgraph.graph import StateGraph, END

from .retrieval import fetch_arxiv, fetch_semantic_scholar, PDFLoaderTool
from .planner import plan_reading_with_llm
from .rag_qa import build_rag
from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests
import time

# Define the State that our agent will use.
class AgentState(TypedDict):
    query: str
    source: str
    papers: List[Dict[str, Any]]
    processed_papers: List[Dict[str, Any]]
    rag_collection: Any
    reading_plan: List[Dict[str, Any]]

# Define the Nodes.
def fetch_papers_node(state: AgentState) -> Dict[str, Any]:
    """Fetches the initial list of papers."""
    print("--- 1. FETCHING PAPERS ---")
    query = state["query"]
    source = state["source"]
    print(f"Searching {source.capitalize()} for '{query}'...")
    if source == "arxiv":
        papers = fetch_arxiv(query)
    else:
        papers = fetch_semantic_scholar(query)
    
    if papers:
        print(f"Found {len(papers)} papers.")
        
    return {"papers": papers}

def process_pdfs_node(state: AgentState) -> Dict[str, Any]:
    """Downloads PDFs, extracts text, and chunks it."""
    print("\n--- 2. PROCESSING FULL TEXT ---")
    papers = state["papers"]
    PDF_DIR = "./temp_pdfs"
    os.makedirs(PDF_DIR, exist_ok=True)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    processed_papers = []
    for paper in papers:
        try:
            pdf_url = paper['url'].replace('/abs/', '/pdf/') + '.pdf' if 'arxiv.org' in paper['url'] else paper['url']
            pdf_filename = f"{paper['url'].split('/')[-1]}.pdf"
            pdf_path = os.path.join(PDF_DIR, pdf_filename)
            response = requests.get(pdf_url)
            response.raise_for_status()
            with open(pdf_path, 'wb') as f: f.write(response.content)
            loader = PDFLoaderTool(file_path=pdf_path)
            pages = loader._run(query="")
            full_text = "\n".join(pages)
            if full_text:
                chunks = text_splitter.split_text(full_text)
                paper_meta = {'title': paper.get('title', ''), 'authors': paper.get('authors', []), 'url': paper.get('url', ''), 'summary': paper.get('summary', ''), 'chunks': chunks}
                processed_papers.append(paper_meta)
            time.sleep(1)
        except Exception as e:
            print(f"  Failed to process paper {paper.get('title', 'Untitled')}: {e}")
    return {"processed_papers": processed_papers}


def plan_reading_node(state: AgentState) -> Dict[str, Any]:
    """Generates the reading plan using the new LLM planner."""
    print("\n--- 3. CREATING READING PLAN ---")
    plan = plan_reading_with_llm(state["processed_papers"])
    return {"reading_plan": plan}

def build_rag_node(state: AgentState) -> Dict[str, Any]:
    """Builds the RAG database from the processed paper chunks."""
    print("\n--- 4. BUILDING RAG DATABASE ---")
    all_chunks = []
    all_metadatas = []
    for paper in state["processed_papers"]:
        for chunk in paper["chunks"]:
            all_chunks.append(chunk)
            all_metadatas.append({'title': paper.get('title', ''), 'authors': ", ".join(paper.get('authors', [])), 'url': paper.get('url', '')})
    if not all_chunks:
        return {"rag_collection": None}
    collection = build_rag(documents=all_chunks, metadatas=all_metadatas)
    print(f"Database built with {len(all_chunks)} text chunks.")
    return {"rag_collection": collection}


def decide_to_process(state: AgentState) -> str:
    """Determines whether to process PDFs or end."""
    if not state.get("papers"):
        print("No papers found. Ending workflow.")
        return "end"
    else:
        return "process"

def decide_to_plan(state: AgentState) -> str:
    """Determines whether to create a plan or end."""
    if not state.get("processed_papers"):
        print("Could not process any papers. Ending workflow.")
        return "end"
    else:
        return "plan"

workflow = StateGraph(AgentState)

workflow.add_node("fetch", fetch_papers_node)
workflow.add_node("process", process_pdfs_node)
workflow.add_node("plan", plan_reading_node)
workflow.add_node("build_rag", build_rag_node)

workflow.set_entry_point("fetch")

workflow.add_conditional_edges(
    "fetch",
    decide_to_process,
    {
        "process": "process",
        "end": END,
    },
)
workflow.add_conditional_edges(
    "process",
    decide_to_plan,
    {
        "plan": "plan",
        "end": END,
    },
)
workflow.add_edge("plan", "build_rag")
workflow.add_edge("build_rag", END)

app = workflow.compile()