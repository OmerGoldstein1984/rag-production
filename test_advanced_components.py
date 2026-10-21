"""
Test script to verify that advanced retrieval components can be imported.
"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all advanced components can be imported."""
    print("Testing imports of advanced retrieval components...")

    try:
        # Test hybrid retrieval
        from src.retrieval.hybrid import HybridRetrieverRRF, create_hybrid_retriever_rrf
        print("✓ hybrid.py imported successfully")
    except Exception as e:
        print(f"✗ Failed to import hybrid.py: {e}")
        return False

    try:
        # Test query rewriter
        from src.retrieval.query_rewriter import QueryRewriter, create_query_rewriter
        print("✓ query_rewriter.py imported successfully")
    except Exception as e:
        print(f"✗ Failed to import query_rewriter.py: {e}")
        return False

    try:
        # Test conversation memory
        from src.retrieval.conversation_memory import ConversationMemory, create_conversation_memory
        print("✓ conversation_memory.py imported successfully")
    except Exception as e:
        print(f"✗ Failed to import conversation_memory.py: {e}")
        return False

    try:
        # Test that existing components still work
        from src.retrieval.retriever import Retriever, create_retriever
        from src.retrieval.reranker import Reranker, create_reranker
        from src.retrieval.sparse_retriever import BM25Retriever, create_bm25_retriever
        print("✓ Existing retrieval components still import successfully")
    except Exception as e:
        print(f"✗ Failed to import existing components: {e}")
        return False

    print("\nAll imports successful! Advanced retrieval components are ready to use.")
    return True

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)