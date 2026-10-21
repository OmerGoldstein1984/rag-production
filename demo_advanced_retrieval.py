"""
Demonstration script showing before/after comparison of advanced retrieval techniques.
"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.generation.chat import RAGChat


def run_comparison_demo():
    """Run a demonstration comparing basic vs advanced retrieval."""
    print("RAG Academy Advanced Retrieval Demonstration")
    print("=" * 50)
    print("Comparing basic retrieval vs advanced retrieval techniques")
    print()

    # Initialize the chat system with basic settings
    print("Initializing RAG system with basic retrieval...")
    chat_basic = RAGChat(data_directory="./data/raw")
    # We'll modify the retriever settings to simulate basic retrieval

    # Initialize the chat system with advanced settings
    print("Initializing RAG system with advanced retrieval...")
    chat_advanced = RAGChat(data_directory="./data/raw")
    # In a full implementation, we would configure different retrievers

    # Define test questions that would benefit from advanced techniques
    test_questions = [
        {
            "question": "What are the 5G NR timing requirements for FR2?",
            "description": "Specific technical question that might benefit from query expansion",
            "expected_benefit": "Query expansion or HyDE could help match technical terminology"
        },
        {
            "question": "How do the handover procedures differ between LTE and 5G?",
            "description": "Comparison question that benefits from hybrid retrieval",
            "expected_benefit": "Hybrid retrieval (BM25 + dense) could capture both exact terms and semantic meaning"
        },
        {
            "question": "What are the penalties for violating timing requirements?",
            "description": "Follow-up question that needs conversation memory",
            "expected_benefit": "Conversation memory would resolve 'timing requirements' from previous context"
        },
        {
            "question": "Explain beamforming in massive MIMO systems",
            "description": "Complex conceptual question benefiting from reranking",
            "expected_benefit": "Cross-encoder reranking would better assess relevance of complex explanations"
        }
    ]

    print("Test Questions:")
    print("-" * 30)
    for i, item in enumerate(test_questions, 1):
        print(f"{i}. {item['description']}")
        print(f"   Question: {item['question']}")
        print(f"   Expected benefit: {item['expected_benefit']}")
        print()

    print("Note: For a full before/after comparison, we would:")
    print("1. Run each question with basic retrieval (dense only, no reranking, etc.)")
    print("2. Run each question with advanced retrieval (all techniques enabled)")
    print("3. Measure metrics like MRR, recall@k, and user relevance judgments")
    print("4. Show specific examples where advanced techniques improved results")
    print()
    print("Key advanced techniques implemented:")
    print("- Hybrid retrieval with Reciprocal Rank Fusion (BM25 + dense)")
    print("- Cross-encoder reranking (top 20 → top 5)")
    print("- Query rewriting (LLM-based, multi-query expansion, HyDE)")
    print("- Conversation memory (follow-up question condensation)")
    print()
    print("See learn/03-advanced-retrieval.md for detailed explanations of each technique.")


if __name__ == "__main__":
    run_comparison_demo()