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
    """
    CLI chat interface for RAG system.

    WHAT: Encapsulates the full RAG pipeline (ingest -> embed -> store ->
    retrieve -> rerank -> generate) and exposes a query() method plus an
    interactive chat_loop() for a command-line chat experience.
    WHY: It is the end-to-end demo/entry point showing how all RAG components
    compose: load docs, clean, chunk, embed, index in a vector store, retrieve
    (dense/sparse/hybrid), rerank, and generate a grounded answer.
    INPUTS (__init__): data_directory: str (default "./data/raw"); use_advanced_retrieval:
        bool (default False).
    OUTPUTS (query): GenerationResult — answer, sources, latency, token counts,
        metadata.
    SIDE EFFECTS: __init__ runs the full ingestion+indexing pipeline (reads
    disk, embeds, writes to vector store); chat_loop() reads stdin and writes
    stdout; logging configured at import.
    FAILURE MODES: __init__ raises if documents can't be loaded/embedded or the
    vector store can't be created; query() logs errors and re-raises inside
    chat_loop's try/except (so one bad question doesn't kill the loop).
    ALTERNATIVE: A FastAPI endpoint (src/api/main.py) serves the same pipeline
    over HTTP instead of a CLI.
    # GAP: __init__ re-ingests and re-indexes on every RAGChat construction — no
    persistence/reuse of an existing index, so repeated CLI invocations duplicate
    work.
    """

    def __init__(self, data_directory: str = "./data/raw", use_advanced_retrieval: bool = False):
        """
        Initialize the RAG chat system.

        WHAT: Store config and run _initialize_pipeline() to ingest, embed, and
        index documents plus build retrievers and the answer generator.
        WHY: Constructing RAGChat performs the expensive one-time setup so each
        query() call is fast.
        INPUTS: data_directory: str — e.g. "./data/raw" (folder of source docs);
            use_advanced_retrieval: bool — whether to use RRF hybrid retrieval.
        OUTPUTS: None (instance).
        SIDE EFFECTS: Runs full ingestion pipeline (disk I/O, embedding,
        vector-store writes); sets self.vector_store, self.retriever, etc.
        FAILURE MODES: Raises if load_documents_from_directory, embed_documents,
        or create_vector_store fail.
        ALTERNATIVE: Accept a pre-built vector store/retriever instead of
        rebuilding from scratch each time.
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
        """
        Initialize the RAG pipeline components.

        WHAT: Execute load -> clean -> chunk -> embed -> store, then build dense
        + sparse + hybrid retrievers, a reranker, and the answer generator.
        WHY: This is the composition root that wires all RAG stages together
        before any query can be served.
        INPUTS: None (uses self.data_directory, self.use_advanced_retrieval).
        OUTPUTS: None (sets instance attributes).
        SIDE EFFECTS: Reads documents from disk; calls embedding (may download
        model); writes to vector store; logs INFO at each stage.
        FAILURE MODES: Raises if any stage (load/clean/chunk/embed/store) fails.
        ALTERNATIVE: Lazily initialize components on first query instead of at
        construction.
        """
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

        WHAT: Retrieve relevant docs (dense/sparse/hybrid), optionally rerank,
        then generate a grounded answer via the answer generator.
        WHY: This is the core request path — turns a user question into a
        sourced answer with latency and token metadata.
        INPUTS: question: str — e.g. "How many remote days are allowed?";
            use_hybrid: bool (default True); use_reranker: bool (default True);
            top_k: int (default 5); current_version_only: bool (default True).
        OUTPUTS: GenerationResult — answer string, sources list, latency_ms,
            token_counts, metadata.
        SIDE EFFECTs: Logs INFO per stage; retrieves from vector store; calls
        the LLM (network) via answer_generator.
        FAILURE MODES: Catches nothing here — exceptions propagate to chat_loop
        (which catches and prints) or to the CLI caller. If no docs retrieved,
        the answer generator still runs with empty context and returns an
        insufficiency message.
        ALTERNATIVE: Skip reranking/hybrid for lower latency at the cost of
        relevance.
        # GAP: The reranker branch reads self.retriever.reranker, but create_reranker()
        result is stored in a local variable `reranker` that is never assigned to
        self — so hasattr(self.retriever, 'reranker') may be False and the
        reranker is effectively unused (dead local).
        # INTERVIEW: Say it out loud — "Retrieve 2x top_k, rerank down to top_k,
        then ground the answer in only those chunks — so the LLM never sees
        unranked noise."
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
        """
        Run an interactive chat loop.

        WHAT: Print a banner, then repeatedly read questions from stdin, run
        query(), and print the answer/sources/latency until the user quits.
        WHY: Provides a human-interactive way to exercise the RAG pipeline
        without writing code or hitting the HTTP API.
        INPUTS: None (reads stdin via input()).
        OUTPUTS: None (prints to stdout).
        SIDE EFFECTs: Blocks on stdin; prints to stdout; catches KeyboardInterrupt
        and generic exceptions per iteration so the loop survives bad input.
        FAILURE MODES: A KeyboardInterrupt/EOF exits gracefully; any other
        exception is caught, logged, and printed — the loop continues.
        ALTERNATIVE: A Streamlit UI (ui/streamlit_app.py) provides the same
        interaction over a browser.
        """
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
    """
    Main CLI entry point.

    WHAT: Parse CLI args, build a RAGChat, and either run one question or the
    interactive loop based on flags.
    WHY: Standard `python -m src.generation.chat` entry point for manual testing
    and demos.
    INPUTS: None (reads sys.argv via argparse).
    OUTPUTS: None (prints results to stdout).
    SIDE EFFECTs: Configures logging level; constructs RAGChat (runs ingestion);
    prints answer/sources/latency/tokens.
    FAILURE MODES: Raises if RAGChat construction fails (bad data dir, embedding
    failure, vector store failure).
    ALTERNATIVE: Import RAGChat and call query() programmatically instead of the
    CLI.
    """
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
