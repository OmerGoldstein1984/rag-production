"""
Test script to verify that the evaluation components work correctly.
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_evaluation_imports():
    """Test that all evaluation components can be imported."""
    print("Testing imports of evaluation components...")

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
        print(f"✓ golden_set.json loaded successfully with {len(golden_set.get('questions', []))} questions")
    except Exception as e:
        print(f"✗ Failed to load golden_set.json: {e}")
        return False

    print("\nAll evaluation imports successful! Components are ready to use.")
    return True

if __name__ == "__main__":
    success = test_evaluation_imports()
    sys.exit(0 if success else 1)