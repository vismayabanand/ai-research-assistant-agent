# AI Research Assistant Agent

An intelligent assistant powered by Large Language Models (LLMs) to help students and researchers explore academic topics efficiently. The agent retrieves relevant papers, summarizes them, extracts key insights, and plans a logical reading path.

---

## 🔍 Project Overview

This project is part of CS180 Summer 2025 Lab and focuses on building an AI agent using modern agentic frameworks like LangChain and LangGraph. The goal is to create a tool that assists in literature review, research planning, and interactive document exploration.

---

## 🧠 Key Features

- **Topic-based Paper Retrieval**: Uses ArXiv/Semantic Scholar APIs to fetch relevant academic papers
- **PDF Parsing & Summarization**: Extracts structured summaries from uploaded or retrieved PDFs
- **Key Insight Extraction**: Identifies core contributions, challenges, and comparisons
- **Reading Path Planning**: Suggests logical reading order based on content structure and topic overlap
- **Agentic RAG (optional)**: Supports interactive question-answering over the retrieved corpus

---

## ⚙️ Tech Stack

- **LangChain**: LLM orchestration and tool integration
- **LangGraph**: Agent state management and planning
- **OpenAI GPT**: LLM backend for summarization and reasoning
- **ArXiv/Semantic Scholar APIs**: External paper sources
- **ChromaDB / FAISS**: Vector database for RAG

---

## 🧩 Architecture Preview

```
User Query
   ↓
Search Agent (ArXiv API)
   ↓
PDF Loader & Summarizer
   ↓
Insight Extractor
   ↓
Reading Planner
   ↓
RAG-based Q&A (optional)
```

---

## 🚧 Project Milestones

- **Week 1**: Idea proposal, repo setup
- **Week 2**: Architecture design + paper retrieval module
- **Week 3**: Summarization + insight extraction
- **Week 4**: Planner + optional RAG
- **Week 5**: Final integration + documentation

---

## 📁 Repository Structure

```
.
├── README.md
├── LICENSE.txt
├── docs/
│   └── architecture_diagram.png
├── src/
│   ├── main_agent.py
│   ├── paper_retriever.py
│   ├── summarizer.py
│   └── planner.py
└── data/
    └── sample_papers/
```

---

## 📜 License

This code is shared for academic demonstration purposes only.  
Copying or reuse without explicit permission is not allowed.

---

## 🤝 Contributions

This is an individual course project for CS180 and is not open to external contributions.

---

## 👤 Author

Vismaya Anand B  
University of California, Riverside  
Summer 2025 – CS180 Lab
