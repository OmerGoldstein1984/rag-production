"""
CLI chat interface for the RAG system.
Usage: python -m src.generation.chat "How many remote days are allowed?"
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List, Dict, Any

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.ingestion.loaders import load_documents_from_directory
from src.ingestion.cleaner import clean_documents
from src.ingestion.chunker import chunk_documents
from src.embedding.embedder import embed_documents
from src.store.vector_store import create_vector_store, add_documents_to_store
from src.retrieval.retriever import create_retriever, create_hybrid_retriever
from src.retrieval.reranker import create_reranker, create_metadata_reranker
from src.retrieval.sparse_retriever import create_bm25_retriever
from src.retrieval.hybrid import create_hybrid_retriever_rrf
from src.generation.answer import create_answer_generator, GenerationResult
from src.generation.prompt import create_grounded_prompt, format_context
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RAGChat:
    """CLI chat interface for RAG system."""

    def __init__(self, data_directory: str = "./data/raw", use_advanced_retrieval: bool = False):
        """
        Initialize the RAG chat system.

        Args:
            data_directory: Directory containing source documents
            use_advanced_retrieval: Whether to use advanced retrieval techniques (RRF hybrid, etc.)
        """
        self.data_directory = data_directory
        self.use_advanced_retrieval = use_advanced_retrieval
        self.vector_store = None
        self.embedder = None
        self.retriever = None
        self.hybrid_retriever = None
        self.answer_generator = None
        self._initialize_pipeline()

    def _initialize_pipeline(self):
        """Initialize the RAG pipeline components."""
        logger.info("Initializing RAG pipeline...")

        # Load and process documents
        logger.info(f"Loading documents from {self.data_directory}")
        raw_docs = load_documents_from_directory(self.data_directory)
        logger.info(f"Loaded {len(raw_docs)} raw documents")

        cleaned_docs = clean_documents(raw_docs)
        logger.info("Documents cleaned")

        chunked_docs = chunk_documents(cleaned_docs, chunk_size=500, chunk_overlap=50)
        logger.info(f"Created {len(chunked_docs)} chunks")

        embedded_docs = embed_documents(chunked_docs)
        logger.info("Documents embedded")

        # Create vector store and add documents
        self.vector_store = create_vector_store()
        add_documents_to_store(self.vector_store, embedded_docs)
        logger.info("Vector store populated")

        # Initialize embedder (needed for retriever)
        from src.embedding.embedder import Embedder
        self.embedder = Embedder()

        # Create retriever
        self.retriever = create_retriever(self.vector_store, self.embedder)

        # Create sparse retriever for hybrid search
        sparse_retriever = create_bm25_retriever()
        sparse_retriever.add_documents(embedded_docs)

        # Create hybrid retriever (basic or advanced based on flag)
        if self.use_advanced_retrieval:
            # Use advanced hybrid with RRF
            self.hybrid_retriever = create_hybrid_retriever_rrf(
                self.retriever, sparse_retriever,
                k=60, weight_dense=0.7, weight_sparse=0.3
            )
            logger.info("Using advanced hybrid retriever with RRF")
        else:
            # Use basic hybrid retriever
            self.hybrid_retriever = create_hybrid_retriever(self.retriever, sparse_retriever)
            logger.info("Using basic hybrid retriever")

        # Create reranker
        reranker = create_reranker()

        # Create answer generator
        self.answer_generator = create_answer_generator()

        logger.info("RAG pipeline initialized successfully")

    def query(self,
              question: str,
              use_hybrid: bool = True,
              use_reranker: bool = True,
              top_k: int = 5,
              current_version_only: bool = True) -> GenerationResult:
        """
        Process a query through the RAG pipeline.

        Args:
            question: User question
            use_hybrid: Whether to use hybrid search
            use_reranker: Whether to use reranking
            top_k: Number of results to retrieve
            current_version_only: Whether to filter to current document versions only

        Returns:
            GenerationResult with answer and metadata
        """
        logger.info(f"Processing query: '{question}'")

        # Retrieve relevant documents
        if use_hybrid:
            retrieved_results = self.hybrid_retriever.retrieve(
                question,
                k=top_k * 2,  # Get extra for reranking
                current_version_only=current_version_only
            )
        else:
            if current_version_only:
                retrieved_results = self.retriever.retrieve_with_version_filter(
                    question,
                    k=top_k * 2
                )
            else:
                retrieved_results = self.retriever.retrieve(
                    question,
                    k=top_k * 2
                )

        retrieved_docs = [doc for doc, _ in retrieved_results]
        retrieval_scores = [score for _, score in retrieved_results]

        logger.info(f"Retrieved {len(retrieved_docs)} documents")

        # Rerank if requested
        if use_reranker and retrieved_docs:
            from src.generation.prompt import format_context
            # For reranking, we need to pass the query and documents with scores
            reranked_results = self.retriever.reranker.rerank(
                question,
                list(zip(retrieved_docs, retrieval_scores)),
                k=top_k
            ) if hasattr(self.retriever, 'reranker') and self.retriever.reranker else \
                 [(doc, score) for doc, score in zip(retrieved_docs[:top_k], retrieval_scores[:top_k])]

            final_docs = [doc for doc, _ in reranked_results]
            final_scores = [score for _, score in reranked_results]
        else:
            # Use top k from initial retrieval
            final_docs = retrieved_docs[:top_k]
            final_scores = retrieval_scores[:top_k]

        logger.info(f"Using top {len(final_docs)} documents after reranking")

        # Generate answer
        result = self.answer_generator.generate_answer_with_context(
            question, final_docs)

        logger.info(f"Generated answer in {result.latency_ms:.2f}ms")
        return result

    def chat_loop(self):
        """Run an interactive chat loop."""
        print("RAG Academy Chat System")
        print("Type your questions about company policies. Type 'quit' to exit.")
        print("-" * 50)

        while True:
            try:
                question = input("\nQuestion: ").strip()

                if question.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break

                if not question:
                    continue

                print("\nProcessing...")
                result = self.query(question)

                print(f"\nAnswer: {result.answer}")
                print(f"\nSources ({len(result.sources)}):")
                for i, source in enumerate(result.sources, 1):
                    print(f"  {i}. {source['source_filename']}")

                print(f"\nLatency: {result.latency_ms:.2f}ms")
                print(f"Tokens: {result.token_counts.get('estimated_total_tokens', 0)} "
                      f"(context: {result.token_counts.get('estimated_tokens', 0)}, "
                      f"prompt: {result.token_counts.get('prompt_tokens', 0)})")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                print(f"Error: {e}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="RAG Academy Chat System")
    parser.add_argument("question", nargs="?", help="Question to ask the RAG system")
    parser.add_argument("--data-dir", type=str, default="./data/raw",
                       help="Directory containing source documents (default: ./data/raw)")
    parser.add_argument("--no-hybrid", action="store_true",
                       help="Use dense retrieval only (no hybrid search)")
    parser.add_argument("--no-reranker", action="store_true",
                       help="Disable reranking")
    parser.add_argument("--use-advanced-retrieval", action="store_true",
                       help="Use advanced retrieval techniques (RRF hybrid, etc.)")
    parser.add_argument("--top-k", type=int, default=5,
                       help="Number of results to retrieve (default: 5)")
    parser.add_argument("--all-versions", action="store_true",
                       help="Include all document versions (not just current)")
    parser.add_argument("--interactive", "-i", action="store_true",
                       help="Run in interactive chat mode")
    parser.add_argument("--log-level",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO",
                       help="Logging level (default: INFO)")

    args = parser.parse_args()

    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Initialize chat system
    chat = RAGChat(data_directory=args.data_dir, use_advanced_retrieval=args.use_advanced_retrieval)

    if args.interactive or not args.question:
        # Interactive mode
        chat.chat_loop()
    else:
        # Single question mode
        result = chat.query(
            question=args.question,
            use_hybrid=not args.no_hybrid,
            use_reranker=not args.no_reranker,
            top_k=args.top_k,
            current_version_only=not args.all_versions
        )

        print(f"Answer: {result.answer}")
        print(f"Sources: {[s['source_filename'] for s in result.sources]}")
        print(f"Latency: {result.latency_ms:.2f}ms")
        print(f"Tokens: {result.token_counts.get('estimated_total_tokens', 0)}")


if __name__ == "__main__":
    main()