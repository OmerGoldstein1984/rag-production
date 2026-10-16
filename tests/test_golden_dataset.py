"""
Test file to verify golden dataset integrity.
"""

import sys
import os

# Add the project root to the path so we can import the evals module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from evals.golden_dataset import get_golden_dataset, get_questions_only, get_question_by_index

def test_golden_dataset_structure():
    """Test that the golden dataset has the proper structure."""
    dataset = get_golden_dataset()

    # Test that we have questions
    assert len(dataset) > 0, "Golden dataset should not be empty"

    # Test that each entry has the required fields
    for i, item in enumerate(dataset):
        assert "question" in item, f"Item {i} missing 'question' field"
        assert "expected_answer" in item, f"Item {i} missing 'expected_answer' field"
        assert "expected_sources" in item, f"Item {i} missing 'expected_sources' field"
        assert "question_type" in item, f"Item {i} missing 'question_type' field"

        # Test field types
        assert isinstance(item["question"], str), f"Item {i} 'question' should be string"
        assert isinstance(item["expected_answer"], str), f"Item {i} 'expected_answer' should be string"
        assert isinstance(item["expected_sources"], list), f"Item {i} 'expected_sources' should be list"
        assert isinstance(item["question_type"], str), f"Item {i} 'question_type' should be string"

        # Test that expected_sources is not empty
        assert len(item["expected_sources"]) > 0, f"Item {i} 'expected_sources' should not be empty"

        # Test that question_type is one of the expected types
        assert item["question_type"] in ["factual", "benefits", "technical", "comparison", "requirements"], \
            f"Item {i} 'question_type' should be one of: factual, benefits, technical, comparison, requirements"

def test_get_questions_only():
    """Test the get_questions_only function."""
    questions = get_questions_only()
    dataset = get_golden_dataset()

    # Should return same number of questions as dataset
    assert len(questions) == len(dataset), "get_questions_only should return same count as dataset"

    # Each question should match the dataset
    for i, question in enumerate(questions):
        assert question == dataset[i]["question"], f"Question {i} should match dataset"

def test_get_question_by_index():
    """Test the get_question_by_index function."""
    dataset = get_golden_dataset()

    # Test valid indices
    for i in range(len(dataset)):
        item = get_question_by_index(i)
        assert item == dataset[i], f"get_question_by_index({i}) should return dataset[{i}]"

    # Test invalid indices
    try:
        get_question_by_index(-1)
        assert False, "get_question_by_index(-1) should raise IndexError"
    except IndexError:
        pass  # Expected

    try:
        get_question_by_index(len(dataset))
        assert False, "get_question_by_index(len(dataset)) should raise IndexError"
    except IndexError:
        pass  # Expected

    try:
        get_question_by_index(len(dataset) + 10)
        assert False, "get_question_by_index(len(dataset)+10) should raise IndexError"
    except IndexError:
        pass  # Expected

def test_question_count():
    """Test that we have a reasonable number of questions."""
    dataset = get_golden_dataset()
    # We expect around 28 questions based on our implementation
    assert len(dataset) >= 20, f"Expected at least 20 questions, got {len(dataset)}"
    assert len(dataset) <= 40, f"Expected at most 40 questions, got {len(dataset)}"

if __name__ == "__main__":
    test_golden_dataset_structure()
    test_get_questions_only()
    test_get_question_by_index()
    test_question_count()
    print("All tests passed!")