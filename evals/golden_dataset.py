"""
Golden dataset for evaluating the RAG system on telecom documentation.
Contains question-answer pairs with expected sources and answers.
"""

from typing import List, Dict, Any

# Golden dataset for telecom documentation (eSIM, roaming specs)
GOLDEN_DATASET = [
    {
        "question": "What is an eSIM and how does it differ from a traditional SIM card?",
        "expected_answer": "An eSIM (embedded SIM) is a programmable SIM card that is embedded directly into a device's hardware, unlike traditional SIM cards which are physical, removable cards. eSIMs can be reprogrammed remotely to switch carriers without physically changing the SIM.",
        "expected_sources": ["esim-spec.pdf", "gsma-remote-sim-provisioning.pdf"],
        "question_type": "factual"
    },
    {
        "question": "What are the key benefits of eSIM technology for consumers?",
        "expected_answer": "Key benefits of eSIM for consumers include: ability to switch carriers without changing physical SIM cards, support for multiple profiles on a single device, easier international roaming by downloading local carrier profiles, and reduced risk of SIM loss or damage.",
        "expected_sources": ["esim-consumer-benefits.pdf", "gsma-esim-overview.pdf"],
        "question_type": "benefits"
    },
    {
        "question": "How does SIM-based authentication work in LTE networks for roaming scenarios?",
        "expected_answer": "In LTE networks, SIM-based authentication uses the Authentication and Key Agreement (AKA) protocol. The SIM stores the subscriber key (Ki) and performs the authentication challenge-response with the home network's AuC (Authentication Center), even when roaming, to verify the subscriber's identity and establish a secure connection.",
        "expected_sources": ["lte-security-spec.pdf", "3gpp-ts-33-401.pdf"],
        "question_type": "technical"
    },
    {
        "question": "What is the difference between steered and non-steered roaming?",
        "expected_answer": "In steered roaming, the home network actively guides the subscriber to preferred partner networks through SIM toolkit commands or network signaling. In non-steered roaming, the device selects available networks based on signal strength or other criteria without active guidance from the home network.",
        "expected_sources": ["roaming-steering-spec.pdf", "gsma-roaming-parameters.pdf"],
        "question_type": "comparison"
    },
    {
        "question": "What are the mandatory GSMA requirements for eSIM remote provisioning?",
        "expected_answer": "Mandatory GSMA requirements for eSIM remote provisioning include: secure communication channels using TLS, mutual authentication between devices and SM-DP+ (Subscription Manager Data Preparation), use of ES6 and ES7 interfaces for profile download and management, and compliance with GSMA Remote SIM Provisioning specifications (SGP.01, SGP.02, SGP.11, SGP.21, SGP.22).",
        "expected_sources": ["gsma-rsp-specification.pdf", "sgp-01-architecture.pdf"],
        "question_type": "requirements"
    }
]

def get_golden_dataset() -> List[Dict[str, Any]]:
    """Return the golden dataset for evaluation."""
    return GOLDEN_DATASET

def get_questions_only() -> List[str]:
    """Return just the questions from the golden dataset."""
    return [item["question"] for item in GOLDEN_DATASET]

def get_question_by_index(index: int) -> Dict[str, Any]:
    """Get a specific question-answer pair by index."""
    if 0 <= index < len(GOLDEN_DATASET):
        return GOLDEN_DATASET[index]
    raise IndexError(f"Index {index} out of range for dataset of size {len(GOLDEN_DATASET)}")