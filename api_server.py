# api_server.py
import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

# Import the LangGraph app and the Q&A function from your project
from src.agent import app as research_agent_app
from src.rag_qa import answer_query

# Initialize the FastAPI app
app = FastAPI(
    title="AI Research Assistant API",
    description="An API for finding, processing, and querying research papers.",
    version="1.0.0",
)

# --- In-Memory Storage for Session Data ---
# In a production system, you would use a proper database like Redis or a persistent DB.
# For this project, a simple dictionary is sufficient to hold session state.
session_data: Dict[str, Any] = {}

# --- Pydantic Models for Request & Response Data ---
# These models define the expected data shapes for our API endpoints.

class ResearchRequest(BaseModel):
    query: str
    source: str = "arxiv"

class ResearchResponse(BaseModel):
    session_id: str
    reading_plan: List[Dict[str, Any]]

class QARequest(BaseModel):
    session_id: str
    question: str

class QAResponse(BaseModel):
    answer: str


# --- API Endpoints ---

@app.post("/start-research", response_model=ResearchResponse)
def start_research(request: ResearchRequest):
    """
    Starts a new research session by running the LangGraph agent.
    This process can take a while as it fetches, downloads, and processes PDFs.
    """
    # Generate a unique ID for this new session
    session_id = str(uuid.uuid4())
    
    print(f"Starting new research session: {session_id}")
    
    # Define the initial state for the LangGraph agent
    initial_state = {"query": request.query, "source": request.source}
    
    # Invoke the agent to run the full workflow
    final_state = research_agent_app.invoke(initial_state)
    
    rag_collection = final_state.get("rag_collection")
    reading_plan = final_state.get("reading_plan")
    
    if not rag_collection or not reading_plan:
        raise HTTPException(status_code=500, detail="Agent workflow failed to produce results.")
        
    # Store the results in our "database"
    session_data[session_id] = {
        "rag_collection": rag_collection,
        "reading_plan": reading_plan,
    }
    
    return {"session_id": session_id, "reading_plan": reading_plan}


@app.post("/ask-question", response_model=QAResponse)
def ask_question(request: QARequest):
    """
    Asks a question within an existing research session.
    """
    # Retrieve the session data using the provided session_id
    session = session_data.get(request.session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
        
    rag_collection = session["rag_collection"]
    
    print(f"Answering question for session {request.session_id}: '{request.question}'")
    
    # Use the RAG collection from the session to answer the question
    answer = answer_query(rag_collection, request.question)
    
    return {"answer": answer}