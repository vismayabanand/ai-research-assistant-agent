#  AI Research Assistant Agent

An intelligent agent that automates the academic research workflow. Give it a topic, and it finds, reads, and prioritizes relevant papers, then builds a knowledge base you can chat with.

## Overview

Starting research on a new academic topic can be overwhelming. This project tackles that problem by providing an AI agent that automates the entire process. It uses Large Language Models and a stateful workflow to transform a simple topic query into a structured, queryable library of academic papers.

[cite_start]This project was built to fulfill the requirements of the CS180 Project Proposal[cite: 1].

##  Features

* **Multi-Source Paper Retrieval**: Fetches academic papers from both **ArXiv** and **Semantic Scholar** based on a user's query.
* **Full-Text PDF Processing**: Instead of just using abstracts, the agent downloads and extracts the **full text** from each paper's PDF for deep analysis and accurate answers.
* **LLM-Powered Reading Plan**: Uses the **Google Gemini LLM** to analyze all processed papers and generate a logical reading plan, ordering papers from foundational to advanced.
* **Stateful Agentic Workflow**: The core backend is built with **LangGraph**, orchestrating the entire process—fetching, processing, and planning—as a robust, stateful graph. This ensures the workflow is reliable and can handle steps where no papers are found or processed.
* **Interactive RAG Q&A**: Builds a vector knowledge base using **ChromaDB**. It then allows you to ask specific questions and get answers grounded directly in the content of the source documents using a Retrieval-Augmented Generation (RAG) pipeline.
* **Simple Web UI**: An interactive user interface built with **Streamlit** allows for easy interaction with the agent without needing to use the command line.

## 🏛️ Architecture & Workflow

The project's workflow is managed by a LangGraph state machine. This graph ensures that each step is executed in the correct order and that the state (like the list of papers and processed text) is passed between them reliably.

<img width="1024" height="1536" alt="image" src="https://github.com/user-attachments/assets/f25cc237-98df-4d6d-9d9f-18b9b93e3752" />

##  Tech Stack

* **Orchestration**: LangGraph
* **LLM & Embeddings**: Google Gemini
* **Vector Database**: ChromaDB
* **UI**: Streamlit
* **Core Libraries**: LangChain, Pydantic, Requests

## ⚙️ Setup and Installation

Follow these steps to set up the project locally.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/vismayabanand/ai-research-assistant-agent.git](https://github.com/vismayabanand/ai-research-assistant-agent.git)
    cd ai-research-assistant-agent
    ```

2.  **Create a virtual environment and activate it:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install the project in editable mode:**
    (This command uses the `pyproject.toml` file to install all required dependencies.)
    ```bash
    pip install -e .
    ```

4.  **Set up your environment variables:**
    Create a file named `.env` in the root of the project and add your Google API key.
    ```env
    # .env
    GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY_HERE"
    ```

##  How to Run

1.  Ensure your virtual environment is activated.
2.  Run the Streamlit application from the project root directory:
    ```bash
    streamlit run ui.py
    ```
3.  A new tab should open in your browser with the application running.

---
##  Project Documentation

This section provides a detailed breakdown of the project's architecture, API, and source code modules, as required for the final project submission.

### How to Run the Project

This application can be run in two different modes:

1.  **Monolithic UI (Recommended for Demo)**: A self-contained Streamlit application that runs the entire workflow in one process.
    ```bash
    # Run this single command from the project root
    streamlit run ui.py
    ```

2.  **Separate Backend & Frontend**: For a more traditional client-server architecture, you can run the FastAPI backend and the Streamlit frontend separately.
    * **Terminal 1 (Backend API):**
        ```bash
        uvicorn api_server:app --reload
        ```
    * **Terminal 2 (Frontend UI):**
        ```bash
        # Note: You would need the API-based version of ui.py for this mode
        streamlit run ui.py
        ```

---
### API Documentation

The research assistant's logic can be exposed via a FastAPI server. The following details how to interact with the API endpoints when running in the client-server mode.

* **Authentication**: There are currently no authentication requirements to use the API.

---

#### Endpoint: `POST /start-research`

This endpoint initiates a new research session. It runs the full LangGraph workflow, which can take several minutes as it fetches, downloads, and processes multiple PDF files.

* **Request Body**:
    ```json
    {
      "query": "your research topic",
      "source": "arxiv"
    }
    ```
    * `query` (string, required): The research topic you want to investigate.
    * `source` (string, optional): The database to search. Can be `"arxiv"` or `"semantic"`. Defaults to `"arxiv"`.

* **Success Response (200 OK)**:
    ```json
    {
      "session_id": "a-unique-session-id",
      "reading_plan": [
        {
          "title": "Paper Title 1",
          "authors": ["Author A", "Author B"],
          "url": "[http://example.com/paper1](http://example.com/paper1)"
        }
      ]
    }
    ```

* **Example Call**:
    ```bash
    curl -X POST "[http://127.0.0.1:8000/start-research](http://127.0.0.1:8000/start-research)" \
    -H "Content-Type: application/json" \
    -d '{"query": "adversarial machine learning"}'
    ```

---

#### Endpoint: `POST /ask-question`

This endpoint allows you to ask a question within an active research session. You must provide the `session_id` returned from a successful `/start-research` call.

* **Request Body**:
    ```json
    {
      "session_id": "the-session-id-from-the-previous-call",
      "question": "your question about the papers"
    }
    ```
    * `session_id` (string, required): The unique ID for an active session.
    * `question` (string, required): The question you want to ask.

* **Success Response (200 OK)**:
    ```json
    {
      "answer": "The agent's answer based on the documents."
    }
    ```

* **Example Call**:
    ```bash
    curl -X POST "[http://127.0.0.1:8000/ask-question](http://127.0.0.1:8000/ask-question)" \
    -H "Content-Type: application/json" \
    -d '{"session_id": "a-unique-session-id", "question": "What is the main contribution of the first paper?"}'
    ```

---
### Code Documentation

This section provides an overview of the key Python modules.

* **`ui.py`**
    * **Purpose**: The main entry point for the application. It creates an interactive user interface using **Streamlit**, allowing users to run the research agent without using the command line.
* **`api_server.py`**
    * **Purpose**: An optional **FastAPI** server that exposes the agent's functionality via RESTful endpoints, allowing it to be used as a separate backend service.
* **`src/agent.py`**
    * **Purpose**: The "brain" of the application. It defines and orchestrates the entire multi-step research workflow using **LangGraph**.
    * **Key Components**: `AgentState` (manages data flow), `fetch_papers_node`, `process_pdfs_node` (workflow steps), and conditional logic functions (`decide_to_process`).
* **`src/retrieval.py`**
    * **Purpose**: Handles fetching academic papers and processing PDF files.
    * **Key Components**: `ArxivTool` and `SemanticScholarTool` for API queries; `PDFLoaderTool` for extracting text from PDFs using **LangChain**.
* **`src/planner.py`**
    * **Purpose**: Contains the logic for creating an intelligent reading plan.
    * **Key Components**: `plan_reading_with_llm` uses the **Gemini LLM** to analyze paper summaries and suggest a logical reading order.
* **`src/rag_qa.py`**
    * **Purpose**: Manages the Retrieval-Augmented Generation (RAG) pipeline for Q&A.
    * **Key Components**: `build_rag` creates a searchable vector database with **ChromaDB**; `answer_query` retrieves relevant context and uses the **Gemini LLM** to synthesize an answer.
* **`src/summarizer.py` & `src/extractor.py`**
    * **Purpose**: Contain functions for more granular, abstract-based content analysis. These were part of the initial project design but are not used in the final LangGraph agent, which processes full PDFs directly.
