# ui.py
import streamlit as st
import time

# Import your agent's functions directly
from src.agent import app as research_agent_app
from src.rag_qa import answer_query

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 AI Research Assistant")

# --- Session State Initialization ---
# This is to store variables across user interactions
if "rag_collection" not in st.session_state:
    st.session_state.rag_collection = None
if "reading_plan" not in st.session_state:
    st.session_state.reading_plan = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Main UI Logic ---

# Section 1: Start a new research session
st.header("1. Start a New Research Session")

topic_input = st.text_input("Enter a research topic:", placeholder="e.g., machine learning")
source_option = st.selectbox("Select a source:", ("arxiv", "semantic"))

if st.button("Start Research"):
    if topic_input:
        with st.spinner("Finding, downloading, and processing papers... This may take a few minutes."):
            try:
                # Define the initial state for the LangGraph agent
                initial_state = {"query": topic_input, "source": source_option}
                
                # Directly invoke the agent's graph
                final_state = research_agent_app.invoke(initial_state)

                # Store the results directly in the session state
                st.session_state.rag_collection = final_state.get("rag_collection")
                st.session_state.reading_plan = final_state.get("reading_plan")
                st.session_state.messages = [] # Clear previous messages
                
                if not st.session_state.rag_collection or not st.session_state.reading_plan:
                    st.error("The agent workflow failed to produce results. Please try another topic.")
                else:
                    st.success("Research session started successfully!")

            except Exception as e:
                st.error(f"An error occurred during the research process: {e}")
    else:
        st.warning("Please enter a research topic.")


# Section 2: Display the Reading Plan and Q&A
if st.session_state.reading_plan:
    st.divider()
    st.header("2. AI-Generated Reading Plan")
    
    # Display the reading plan
    for idx, paper in enumerate(st.session_state.reading_plan, 1):
        st.markdown(f"**{idx}. {paper.get('title', 'Untitled')}**")
        with st.expander("Details"):
            authors = paper.get('authors')
            # Handle both list and string author formats
            author_str = ", ".join(authors) if isinstance(authors, list) else authors
            st.markdown(f"**Authors:** {author_str or 'N/A'}")
            st.markdown(f"**URL:** {paper.get('url', '#')}")

    st.divider()
    st.header("3. Ask Questions About the Papers")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input for user's question
    if prompt := st.chat_input("What would you like to ask?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Thinking..."):
                try:
                    # Directly call the answer_query function
                    answer = answer_query(st.session_state.rag_collection, prompt)
                    message_placeholder.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                except Exception as e:
                    error_message = f"An error occurred: {e}"
                    message_placeholder.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})