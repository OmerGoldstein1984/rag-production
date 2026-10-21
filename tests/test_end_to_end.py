"""
End-to-end test for the RAG system.
Tests the complete flow from document ingestion to query answering.
"""
import pytest
import sys
import os
import tempfile
import shutil
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.ingestion.loaders import load_documents_from_directory
from src.ingestion.cleaner import clean_documents
from src.ingestion.chunker import chunk_documents
from src.embedding.embedder import embed_documents
from src.store.vector_store import create_vector_store, add_documents_to_store
from src.retrieval.retriever import create_retriever
from src.retrieval.reranker import create_reranker
from src.generation.answer import create_answer_generator

def test_end_to_end_rag_pipeline():
    """Test the complete RAG pipeline with real documents."""
    # Create a temporary directory with test documents
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test documents
        test_doc_content = """
        Acme Corp Remote Work Policy

        Employees are eligible for remote work after completing a 6-month introductory period.
        Remote work is allowed up to 3 days per week with manager approval.
        Core collaboration hours are 10:00 AM to 3:00 PM local time.
        The company provides laptop, monitor, and necessary peripherals for remote work.
        Internet reimbursement is $50 per month for service costs.
        Multi-factor authentication (MFA) and company VPN are required for remote logins.
        """

        test_file = Path(temp_dir) / "remote_work_policy.md"
        test_file.write_text(test_doc_content.strip())

        try:
            # Step 1: Load documents
            raw_docs = load_documents_from_directory(temp_dir)
            assert len(raw_docs) > 0, "Should load at least one document"

            # Step 2: Clean documents
            cleaned_docs = clean_documents(raw_docs)
            assert len(cleaned_docs) > 0, "Should have cleaned documents"

            # Step 3: Chunk documents
            chunked_docs = chunk_documents(
                cleaned_docs,
                chunk_size=200,
                chunk_overlap=20,
                strategy="recursive"
            )
            assert len(chunked_docs) > 0, "Should have created chunks"

            # Step 4: Embed documents
            embedded_docs = embed_documents(chunked_docs)
            assert len(embedded_docs) == len(chunked_docs), "Should have same number of embedded docs"

            # Step 5: Create vector store and add documents
            vector_store = create_vector_store()
            add_documents_to_store(vector_store, embedded_docs)

            # Verify documents were stored
            stats = vector_store.get_stats()
            assert stats.get("total_documents", 0) > 0, "Should have stored documents"

            # Step 6: Create retriever and reranker
            retriever = create_retriever(vector_store, None)  # None for embedder in test
            reranker = create_reranker()
            answer_generator = create_answer_generator()

            # Verify components were created
            assert retriever is not None, "Retriever should be created"
            assert reranker is not None, "Reranker should be created"
            assert answer_generator is not None, "Answer generator should be created"

            # Step 7: Test retrieval
            query = "What is the remote work policy?"
            retrieved_docs_with_scores = retriever.retrieve(query, k=3)
            assert len(retrieved_docs_with_scores) > 0, "Should retrieve documents"

            retrieved_docs = [doc for doc, _ in retrieved_docs_with_scores]
            assert len(retrieved_docs) > 0, "Should have retrieved document objects"

            # Step 8: Test reranking
            if reranker is not None:
                scores = [score for _, score in retrieved_docs_with_scores]
                reranked_docs_with_scores = reranker.rerank(
                    query,
                    list(zip(retrieved_docs, scores)),
                    k=3
                )
                assert len(reranked_docs_with_scores) <= 3, "Should rerank to requested k"

            # Step 9: Test answer generation
            final_docs = [doc for doc, _ in retrieved_docs_with_scores[:2]]  # Use top 2
            answer_result = answer_generator.generate_answer_with_context(
                query, final_docs
            )

            assert "answer" in answer_result, "Should generate an answer"
            assert isinstance(answer_result["answer"], str), "Answer should be a string"
            assert len(answer_result["answer"]) > 0, "Answer should not be empty"

            # Verify answer contains relevant information
            answer_lower = answer_result["answer"].lower()
            # Should mention something about remote work policy
            assert any(term in answer_lower for term in ["remote", "work", "policy", "days"]), \
                "Answer should contain relevant terms"

        except Exception as e:
            # Re-raise the exception to fail the test
            pytest.fail(f"End-to-end test failed with exception: {str(e)}")

def test_end_to_end_with_known_question():
    """Test with a question similar to those in the golden set."""
    # Use the existing data/raw directory
    data_dir = Path(__file__).parent.parent / "data" / "raw"

    if not data_dir.exists():
        pytest.skip("Data directory not found - skipping end-to-end test with real data")

    try:
        # Initialize pipeline with existing data
        from src.ingestion.loaders import load_documents_from_directory
        from src.ingestion.cleaner import clean_documents
        from src.ingestion.chunker import chunk_documents
        from src.embedding.embedder import embed_documents
        from src.store.vector_store import create_vector_store, add_documents_to_store
        from src.retrieval.retriever import create_retriever
        from src.generation.answer import create_answer_generator

        # Load and process documents
        raw_docs = load_documents_from_directory(str(data_dir))
        assert len(raw_docs) > 0, "Should load documents from data/raw"

        cleaned_docs = clean_documents(raw_docs)
        chunked_docs = chunk_documents(cleaned_docs, chunk_size=500, chunk_overlap=50)
        embedded_docs = embed_documents(chunked_docs)

        vector_store = create_vector_store()
        add_documents_to_store(vector_store, embedded_docs)

        retriever = create_retriever(vector_store, None)
        answer_generator = create_answer_generator()

        # Test a known question from the golden set
        query = "What is Acme Corp's policy on remote work?"
        retrieved_docs_with_scores = retriever.retrieve(query, k=5)

        assert len(retrieved_docs_with_scores) > 0, "Should retrieve documents for known question"

        # Generate answer
        final_docs = [doc for doc, _ in retrieved_docs_with_scores[:3]]
        answer_result = answer_generator.generate_answer_with_context(query, final_docs)

        assert "answer" in answer_result
        assert len(answer_result["answer"]) > 0

        # The answer should contain expected terms from the golden set
        answer_lower = answer_result["answer"].lower()
        expected_terms = ["remote work", "eligible employees", "3 days per week", "manager approval"]

        # At least some of the expected terms should be present
        found_terms = [term for term in expected_terms if term in answer_lower]
        assert len(found_terms) > 0, f"Should find at least some expected terms. Answer: {answer_result['answer']}"

    except Exception as e:
        pytest.fail(f"End-to-end test with known question failed: {str(e)}")

if __name__ == "__main__":
    pytest.main([__file__])