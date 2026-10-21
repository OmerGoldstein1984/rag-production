"""
Streamlit UI for the RAG Academy system.
Provides a simple interface for asking questions and viewing answers with citations.
"""

import streamlit as st
import requests
import json
import time
from typing import Dict, Any, List

# Configuration
API_BASE_URL = "http://localhost:8000"

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "feedback_given" not in st.session_state:
        st.session_state.feedback_given = set()

def call_api_ask(query: str, top_k: int = 5, use_reranker: bool = True,
                 use_hybrid: bool = True, use_rrf: bool = False,
                 current_version_only: bool = True,
                 access_level: int = 1) -> Dict[str, Any]:
    """Call the /ask endpoint of the RAG API."""
    headers = {
        "Content-Type": "application/json",
        "X-Access-Level": str(access_level)
    }

    payload = {
        "query": query,
        "top_k": top_k,
        "use_reranker": use_reranker,
        "use_hybrid": use_hybrid,
        "use_rrf": use_rrf,
        "current_version_only": current_version_only
    }

    try:
        response = requests.post(
            f"{API_BASE_URL}/ask",
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling API: {str(e)}")
        return {"error": str(e)}

def call_api_search(query: str, top_k: int = 5,
                    use_hybrid: bool = True, use_rrf: bool = False,
                    current_version_only: bool = True,
                    access_level: int = 1) -> Dict[str, Any]:
    """Call the /search endpoint of the RAG API."""
    headers = {
        "Content-Type": "application/json",
        "X-Access-Level": str(access_level)
    }

    payload = {
        "query": query,
        "top_k": top_k,
        "use_hybrid": use_hybrid,
        "use_rrf": use_rrf,
        "current_version_only": current_version_only
    }

    try:
        response = requests.post(
            f"{API_BASE_URL}/search",
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling search API: {str(e)}")
        return {"error": str(e)}

def call_api_health() -> Dict[str, Any]:
    """Call the /health endpoint of the RAG API."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling health API: {str(e)}")
        return {"error": str(e)}

def submit_feedback(feedback_id: str, rating: int, comment: str = None):
    """Submit feedback for a response."""
    headers = {"Content-Type": "application/json"}
    payload = {
        "feedback_id": feedback_id,
        "rating": rating,
        "comment": comment
    }

    try:
        response = requests.post(
            f"{API_BASE_URL}/feedback",
            headers=headers,
            json=payload,
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error submitting feedback: {str(e)}")
        return {"error": str(e)}

def display_sources(sources: List[Dict[str, Any]]):
    """Display source documents in an expandable format."""
    if not sources:
        st.info("No sources found.")
        return

    for i, source in enumerate(sources):
        with st.expander(f"Source {i+1}: {source['source']}"):
            st.write("**Content Snippet:**")
            st.write(source["content_snippet"])

            if st.checkbox(f"Show full metadata for source {i+1}", key=f"meta_{i}"):
                st.json(source["metadata"])

def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="RAG Academy - Telecom Knowledge Base",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    initialize_session_state()

    # Header
    st.title("📚 RAG Academy Telecom Knowledge Base")
    st.markdown("*Field Development Engineer Interview Preparation Assistant*")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        # Access level selector
        access_level_options = {
            "Guest (Level 0)": 0,
            "Employee (Level 1)": 1,
            "Manager (Level 2)": 2,
            "Admin (Level 3)": 3
        }

        selected_access = st.selectbox(
            "Access Level",
            options=list(access_level_options.keys()),
            index=1  # Default to Employee
        )
        access_level = access_level_options[selected_access]

        # Advanced options
        st.subheader("🔧 Advanced Options")
        top_k = st.slider("Number of sources to retrieve", 1, 10, 5)
        use_reranker = st.checkbox("Use reranker", value=True)
        use_hybrid = st.checkbox("Use hybrid retrieval (BM25 + dense)", value=True)
        use_rrf = st.checkbox("Use Reciprocal Rank Fusion", value=False)
        current_version_only = st.checkbox("Current version only", value=True)

        # API Health Check
        st.subheader("🏥 System Status")
        if st.button("Check System Health"):
            with st.spinner("Checking system health..."):
                health_data = call_api_health()
                if "error" not in health_data:
                    st.success("✅ System is healthy")
                    st.json(health_data)
                else:
                    st.error("❌ System unhealthy")

    # Main content area
    tab1, tab2, tab3 = st.tabs(["❓ Ask Questions", "🔍 Search Documents", "📊 Analytics"])

    with tab1:
        st.header("Ask a Question")

        # Question input
        query = st.text_area(
            "Enter your telecom domain question:",
            placeholder="e.g., What is Acme Corp's policy on remote work?",
            height=100
        )

        col1, col2 = st.columns([1, 4])
        with col1:
            ask_button = st.button("🚀 Ask Question", type="primary", disabled=not query.strip())

        if ask_button and query.strip():
            with st.spinner("Processing your question..."):
                start_time = time.time()
                result = call_api_ask(
                    query=query.strip(),
                    top_k=top_k,
                    use_reranker=use_reranker,
                    use_hybrid=use_hybrid,
                    use_rrf=use_rrf,
                    current_version_only=current_version_only,
                    access_level=access_level
                )
                end_time = time.time()

                if "error" not in result:
                    # Display answer
                    st.success("✅ Answer generated!")
                    st.write("### Answer")
                    st.write(result["answer"])

                    # Display metadata
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Latency", f"{result['latency_ms']}ms")
                    with col2:
                        st.metric("Sources Found", len(result["sources"]))
                    with col3:
                        st.metric("Feedback ID", result["feedback_id"][:8] + "...")

                    # Display sources
                    st.write("### Sources & Citations")
                    display_sources(result["sources"])

                    # Feedback section
                    st.write("### Feedback")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("👍 Helpful", key=f"up_{result['feedback_id']}"):
                            feedback_result = submit_feedback(
                                feedback_id=result["feedback_id"],
                                rating=1,
                                comment="User found the answer helpful"
                            )
                            if "error" not in feedback_result:
                                st.success("Thank you for your feedback!")
                            else:
                                st.error("Failed to submit feedback")

                    with col2:
                        if st.button("👍 Not Helpful", key=f"down_{result['feedback_id']}"):
                            feedback_result = submit_feedback(
                                feedback_id=result["feedback_id"],
                                rating=0,
                                comment="User found the answer not helpful"
                            )
                            if "error" not in feedback_result:
                                st.success("Thank you for your feedback!")
                            else:
                                st.error("Failed to submit feedback")

                    # Add to conversation history
                    st.session_state.conversation_history.append({
                        "timestamp": time.time(),
                        "query": query,
                        "answer": result["answer"],
                        "feedback_id": result["feedback_id"]
                    })

                else:
                    st.error(f"❌ Error: {result['error']}")

    with tab2:
        st.header("Search Documents")

        # Search input
        search_query = st.text_area(
            "Enter search terms:",
            placeholder="e.g., remote work policy or PTO accrual",
            height=100
        )

        col1, col2 = st.columns([1, 4])
        with col1:
            search_button = st.button("🔍 Search", type="primary", disabled=not search_query.strip())

        if search_button and search_query.strip():
            with st.spinner("Searching documents..."):
                start_time = time.time()
                result = call_api_search(
                    query=search_query.strip(),
                    top_k=top_k,
                    use_hybrid=use_hybrid,
                    use_rrf=use_rrf,
                    current_version_only=current_version_only,
                    access_level=access_level
                )
                end_time = time.time()

                if "error" not in result:
                    st.success(f"✅ Found {len(result['results'])} results!")
                    st.write(f"**Query:** {result['query']}")
                    st.write(f"**Latency:** {result['latency_ms']}ms")

                    # Display results
                    for i, res in enumerate(result["results"]):
                        with st.expander(f"Result {i+1} - Score: {res['score']:.3f}"):
                            st.write("**Content:**")
                            st.write(res["content"][:500] + ("..." if len(res["content"]) > 500 else ""))
                            st.write(f"**Source:** {res['source']}")

                            if st.checkbox(f"Show metadata for result {i+1}", key=f"search_meta_{i}"):
                                st.json(res["metadata"])
                else:
                    st.error(f"❌ Error: {result['error']}")

    with tab3:
        st.header("Analytics & Conversation History")

        # Conversation history
        if st.session_state.conversation_history:
            st.subheader("💬 Conversation History")
            for i, convo in enumerate(reversed(st.session_state.conversation_history[-5:])):  # Show last 5
                with st.expander(f"Q: {convo['query'][:50]}... ({time.strftime('%H:%M:%S', time.localtime(convo['timestamp']))})"):
                    st.write("**Question:**")
                    st.write(convo["query"])
                    st.write("**Answer:**")
                    st.write(convo["answer"])
                    st.write(f"**Feedback ID:** {convo['feedback_id']}")
        else:
            st.info("No conversation history yet. Start asking questions!")

        # System stats placeholder
        st.subheader("📈 System Statistics")
        if st.button("Refresh Statistics"):
            with st.spinner("Fetching system statistics..."):
                health_data = call_api_health()
                if "error" not in health_data:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Vector Store Documents",
                                health_data.get("vector_store_stats", {}).get("total_documents", "N/A"))
                        st.metric("Cache Hit Rate",
                                f"{health_data.get('cache_stats', {}).get('hit_rate', 0):.2%}")
                    with col2:
                        st.metric("Vector Store Chunks",
                                health_data.get("vector_store_stats", {}).get("total_chunks", "N/A"))
                        st.metric("Cache Size",
                                f"{health_data.get('cache_stats', {}).get('size', 0)}/{health_data.get('cache_stats', {}).get('max_size', 0)}")
                else:
                    st.error("Failed to fetch system statistics")

if __name__ == "__main__":
    main()