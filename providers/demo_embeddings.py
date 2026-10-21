#!/usr/bin/env python3
"""
Demo script to demonstrate embedding concepts and cosine similarity.
This script creates embeddings for 3 sentences and calculates their pairwise cosine similarities.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from providers.embeddings import create_embedding_provider
import numpy as np

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

def main():
    print("=== Embedding Demo ===\n")

    # Initialize the embedding provider (using mock for demo)
    provider = create_embedding_provider("mock", dimension=384)
    print(f"Using embedding provider: {provider.get_model_info()['provider']}")
    print(f"Embedding dimension: {provider.get_model_info()['dimension']}\n")

    # Three example sentences
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "A fast brown canine leaps above a sleepy canine.",
        "Machine learning is a subset of artificial intelligence."
    ]

    print("Sentences:")
    for i, sentence in enumerate(sentences, 1):
        print(f"{i}. {sentence}")
    print()

    # Get embeddings for each sentence
    embeddings = provider.embed_batch(sentences)

    print("Embeddings generated (showing first 5 dimensions):")
    for i, (sentence, embedding) in enumerate(zip(sentences, embeddings)):
        print(f"{i+1}. [{sentence[:30]}...] -> [{', '.join(f'{x:.3f}' for x in embedding[:5])}, ...]")
    print()

    # Calculate pairwise cosine similarities
    print("Pairwise Cosine Similarities:")
    print("(Values range from -1 to 1, where 1 means identical direction, 0 means orthogonal, -1 means opposite)")
    print()

    for i in range(len(sentences)):
        for j in range(i+1, len(sentences)):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            print(f"Between sentence {i+1} and {j+1}: {sim:.4f}")
            print(f"  '{sentences[i]}'")
            print(f"  '{sentences[j]}'")
            print()

    # Interpretation
    print("Interpretation:")
    sim_1_2 = cosine_similarity(embeddings[0], embeddings[1])  # Similar sentences
    sim_1_3 = cosine_similarity(embeddings[0], embeddings[2])  # Dissimilar sentences
    sim_2_3 = cosine_similarity(embeddings[1], embeddings[2])  # Dissimilar sentences

    print(f"- Sentences 1 & 2 (similar meaning): {sim_1_2:.4f} (high similarity expected)")
    print(f"- Sentences 1 & 3 (different topics): {sim_1_3:.4f} (lower similarity expected)")
    print(f"- Sentences 2 & 3 (different topics): {sim_2_3:.4f} (lower similarity expected)")
    print()

    print("Demo complete! This demonstrates how:")
    print("1. Semantically similar sentences get higher cosine similarity scores")
    print("2. Dissimilar sentences get lower cosine similarity scores")
    print("3. Embeddings capture meaning beyond exact word matching")

if __name__ == "__main__":
    main()