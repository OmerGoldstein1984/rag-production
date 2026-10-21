"""
Demonstration script showing how to use the evaluation pipeline.
"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def print_demo_info():
    """Print information about the evaluation demonstration."""
    print("RAG Academy Evaluation Demonstration")
    print("=" * 40)
    print()
    print("This demo shows how to use the evaluation pipeline to:")
    print("1. Evaluate a RAG system with different configurations")
    print("2. Run ablation studies to understand component contributions")
    print("3. Generate results for stakeholder presentations")
    print()
    print("Key features of the evaluation pipeline:")
    print("- Configurable retrieval (dense, hybrid, RRF, reranking)")
    print("- Comprehensive metrics (hit-rate@K, recall@K, MRR, nDCG)")
    print("- LLM-as-judge for generation quality assessment")
    print("- Ablation study capabilities")
    print("- Results export to JSON and Markdown")
    print()
    print("To run a basic evaluation:")
    print("  python -m src.evaluation.run_eval --data-dir data/raw")
    print()
    print("To run an ablation study:")
    print("  python -m src.evaluation.run_eval --run-ablation")
    print()
    print("To run a configuration comparison:")
    print("  python -m src.evaluation.run_eval --run-comparison")
    print()
    print("See learn/04-evaluation.md for detailed explanations of")
    print("evaluation methodologies, metrics, and best practices.")

if __name__ == "__main__":
    print_demo_info()