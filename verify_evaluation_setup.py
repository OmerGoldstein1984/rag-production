"""
Verification script to confirm that all evaluation components are properly set up.
This script checks imports and basic structure without running the full evaluation.
"""
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def verify_imports():
    """Verify that all evaluation components can be imported."""
    print("Verifying evaluation component imports...")

    try:
        # Test metrics
        from src.evaluation.metrics import RetrievalMetrics, GenerationMetrics, RAGEvaluator
        print("✓ metrics.py imported successfully")
    except Exception as e:
        print(f"✗ Failed to import metrics.py: {e}")
        return False

    try:
        # Test run_eval
        from src.evaluation.run_eval import RAGEvaluationPipeline
        print("✓ run_eval.py imported successfully")
    except Exception as e:
        print(f"✗ Failed to import run_eval.py: {e}")
        return False

    try:
        # Test golden set loading
        import json
        golden_set_path = project_root / "src" / "evaluation" / "golden_set.json"
        with open(golden_set_path, 'r') as f:
            golden_set = json.load(f)
        question_count = len(golden_set.get('questions', []))
        print(f"✓ golden_set.json loaded successfully with {question_count} questions")
        if question_count != 30:
            print(f"⚠ Warning: Expected 30 questions, got {question_count}")
    except Exception as e:
        print(f"✗ Failed to load golden_set.json: {e}")
        return False

    try:
        # Test that key methods exist
        pipeline = RAGEvaluationPipeline(
            chunk_strategy="recursive",
            chunk_size=500,
            chunk_overlap=50,
            use_hybrid=True,
            use_reranker=True,
            use_rrf=True,
            hybrid_alpha=0.7,
            prompt_version="grounded_v1"
        )
        print("✓ RAGEvaluationPipeline can be instantiated")
    except Exception as e:
        print(f"✗ Failed to instantiate RAGEvaluationPipeline: {e}")
        return False

    print("\nAll verification checks passed! The evaluation system is ready to run.")
    print("\nTo run the ablation study when the service is available:")
    print("  python -m src.evaluation.run_eval --run-ablation --data-dir data/raw")
    return True

if __name__ == "__main__":
    success = verify_imports()
    sys.exit(0 if success else 1)