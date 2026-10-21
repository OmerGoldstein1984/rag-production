"""
CLI for running the document ingestion pipeline.
Usage: python -m src.ingestion.run --strategy recursive --chunk-size 500 --overlap 50
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
from src.ingestion.chunker import chunk_documents, ChunkingStrategy
from src.embedding.embedder import embed_documents
from src.store.vector_store import create_vector_store, add_documents_to_store
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_ingestion_pipeline(data_directory: str = "./data/raw",
                          strategy: str = "recursive",
                          chunk_size: int = 500,
                          overlap: int = 50,
                          persist_directory: str = "./data/chroma_db") -> None:
    """
    Run the complete ingestion pipeline.

    Args:
        data_directory: Directory containing source documents
        strategy: Chunking strategy ("fixed_size", "recursive", "structure_aware")
        chunk_size: Target size of each chunk
        overlap: Number of characters to overlap between chunks
        persist_directory: Directory to persist the vector store
    """
    logger.info("Starting ingestion pipeline...")

    # Step 1: Load documents
    logger.info(f"Loading documents from {data_directory}")
    documents = load_documents_from_directory(data_directory)
    logger.info(f"Loaded {len(documents)} documents")

    if len(documents) == 0:
        logger.warning("No documents found to process")
        return

    # Step 2: Clean documents
    logger.info("Cleaning documents")
    cleaned_documents = clean_documents(documents)
    logger.info("Document cleaning completed")

    # Step 3: Chunk documents
    logger.info(f"Chunking documents using {strategy} strategy (chunk_size={chunk_size}, overlap={overlap})")
    chunked_documents = chunk_documents(
        cleaned_documents,
        strategy=strategy,
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    logger.info(f"Created {len(chunked_documents)} chunks from {len(cleaned_documents)} documents")

    # Step 4: Embed documents
    logger.info("Embedding chunks")
    embedded_documents = embed_documents(chunked_documents)
    logger.info(f"Embedded {len(embedded_documents)} chunks")

    # Step 5: Store in vector database
    logger.info(f"Storing embeddings in vector store at {persist_directory}")
    vector_store = create_vector_store(persist_directory=persist_directory)
    add_documents_to_store(vector_store, embedded_documents)

    # Print statistics
    stats = vector_store.get_stats()
    logger.info(f"Vector store stats: {stats}")

    logger.info("Ingestion pipeline completed successfully!")


def compare_chunking_strategies(data_directory: str = "./data/raw",
                               chunk_size: int = 500,
                               overlap: int = 50) -> None:
    """
    Compare all three chunking strategies and print a comparison table.

    Args:
        data_directory: Directory containing source documents
        chunk_size: Target size of each chunk
        overlap: Number of characters to overlap between chunks
    """
    logger.info("Comparing chunking strategies...")

    # Load and clean documents once
    documents = load_documents_from_directory(data_directory)
    cleaned_documents = clean_documents(documents)

    if len(cleaned_documents) == 0:
        logger.warning("No documents found to process")
        return

    # Test each strategy
    strategies = ["fixed_size", "recursive", "structure_aware"]
    results = []

    for strategy in strategies:
        logger.info(f"Testing {strategy} strategy...")
        chunked_docs = chunk_documents(
            cleaned_documents,
            strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=overlap
        )

        # Calculate statistics
        chunk_sizes = [len(doc["content"]) for doc in chunked_docs]
        avg_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
        min_size = min(chunk_sizes) if chunk_sizes else 0
        max_size = max(chunk_sizes) if chunk_sizes else 0

        # Find an example of a "bad split" (chunk that ends in the middle of a word or sentence)
        bad_split_example = None
        for doc in chunked_docs[:5]:  # Check first 5 chunks
            content = doc["content"]
            if content and not content.endswith(('.', '!', '?', '\n', ' ')) and len(content) > 10:
                # Check if it ends mid-word (last character is letter and second-to-last is not space)
                if len(content) >= 2 and content[-1].isalpha() and content[-2] != ' ':
                    bad_split_example = content[:50] + "..." if len(content) > 50 else content
                    break

        results.append({
            "strategy": strategy,
            "num_chunks": len(chunked_docs),
            "avg_size": round(avg_size, 1),
            "min_size": min_size,
            "max_size": max_size,
            "bad_split_example": bad_split_example or "No bad splits found in first 5 chunks"
        })

    # Print comparison table
    print("\n" + "="*80)
    print("CHUNKING STRATEGY COMPARISON")
    print("="*80)
    print(f"{'Strategy':<15} {'Chunks':<8} {'Avg Size':<10} {'Min Size':<10} {'Max Size':<10} {'Bad Split Example'}")
    print("-"*80)

    for result in results:
        strategy = result["strategy"]
        num_chunks = result["num_chunks"]
        avg_size = result["avg_size"]
        min_size = result["min_size"]
        max_size = result["max_size"]
        bad_example = result["bad_split_example"]

        # Truncate bad example for display
        if len(bad_example) > 30:
            bad_example = bad_example[:27] + "..."

        print(f"{strategy:<15} {num_chunks:<8} {avg_size:<10} {min_size:<10} {max_size:<10} {bad_example}")

    print("="*80)
    print(f"Notes: All strategies used chunk_size={chunk_size}, overlap={overlap}")
    print("Bad split example shows a chunk that doesn't end at a natural boundary")
    print("="*80)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Run the RAG Academy ingestion pipeline")
    parser.add_argument("--strategy",
                       choices=["fixed_size", "recursive", "structure_aware"],
                       default="recursive",
                       help="Chunking strategy to use (default: recursive)")
    parser.add_argument("--chunk-size",
                       type=int,
                       default=500,
                       help="Target size of each chunk in characters (default: 500)")
    parser.add_argument("--overlap",
                       type=int,
                       default=50,
                       help="Number of characters to overlap between chunks (default: 50)")
    parser.add_argument("--data-dir",
                       type=str,
                       default="./data/raw",
                       help="Directory containing source documents (default: ./data/raw)")
    parser.add_argument("--persist-dir",
                       type=str,
                       default="./data/chroma_db",
                       help="Directory to persist vector store (default: ./data/chroma_db)")
    parser.add_argument("--compare",
                       action="store_true",
                       help="Compare all chunking strategies and exit")
    parser.add_argument("--log-level",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO",
                       help="Logging level (default: INFO)")

    args = parser.parse_args()

    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    if args.compare:
        compare_chunking_strategies(
            data_directory=args.data_dir,
            chunk_size=args.chunk_size,
            overlap=args.overlap
        )
    else:
        run_ingestion_pipeline(
            data_directory=args.data_dir,
            strategy=args.strategy,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            persist_directory=args.persist_dir
        )


if __name__ == "__main__":
    main()