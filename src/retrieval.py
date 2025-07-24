# src/retrieval.py
import os
import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain.tools import BaseTool
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

# **MODIFIED CLASS DEFINITION**
class PDFLoaderTool(BaseTool):
    name: str = "pdf_loader"
    description: str = "Load and split text from an uploaded PDF file."
    
    # Declare file_path as a class attribute, which is the correct Pydantic way.
    file_path: str

    # The custom __init__ method has been removed.
    # Pydantic now handles initialization automatically.

    def _run(self, query: str) -> List[str]:
        # self.file_path will now exist correctly.
        loader = PyPDFLoader(self.file_path)
        docs = loader.load_and_split()
        return [page.page_content for page in docs]

class ArxivTool(BaseTool):
    name: str = "arxiv_search"
    description: str = "Fetch papers from ArXiv for a given query"

    def _run(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        base = os.getenv("ARXIV_API_URL", "https://export.arxiv.org/api/query")
        params = {'search_query': f'all:{query}', 'start': 0, 'max_results': max_results}
        resp = requests.get(base, params=params)
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        papers = []
        for entry in root.findall('atom:entry', ns):
            papers.append({
                'title': entry.find('atom:title', ns).text.strip(),
                'authors': [a.find('atom:name', ns).text for a in entry.findall('atom:author', ns)],
                'summary': entry.find('atom:summary', ns).text.strip(),
                'url': entry.find('atom:id', ns).text
            })
        return papers

class SemanticScholarTool(BaseTool):
    name: str = "semantic_scholar_search"
    description: str = "Fetch papers from Semantic Scholar for a given query"

    def _run(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        headers = {'x-api-key': api_key} if api_key else {}
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {'query': query, 'limit': max_results, 'fields': 'title,authors,abstract,url'}
        resp = requests.get(url, params=params, headers=headers)
        resp.raise_for_status()
        data = resp.json().get('data', [])
        return [
            {'title': p['title'],
             'authors': [a['name'] for a in p['authors']],
             'summary': p.get('abstract', ''),
             'url': p['url']}
            for p in data
        ]


def fetch_arxiv(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    return ArxivTool()._run(query, max_results)


def fetch_semantic_scholar(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    return SemanticScholarTool()._run(query, max_results)