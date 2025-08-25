#!/usr/bin/env python3
"""
LLM Embedding Inference Script

This script loads a trained SimCSE model and generates embeddings.
Usage:
    Relevant text:
        uv run modeling/llm_embedding/inference.py "Aritificial Intelligence is the future. -- John Doe" "Deep learning is AGI. -- Jane Smith"
    Irrelevant text:
        uv run modeling/llm_embedding/inference.py "Today is a sunny day. -- John Doe" "Deep learning is AGI. -- Jane Smith"
"""

import torch
import argparse
from pathlib import Path

# Import the SimCSEModel from shared model file
try:
    from .model import SimCSEModel
except ImportError:
    # Fallback for direct script execution
    import sys
    import os

    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_dir)
    from model import SimCSEModel


def load_model(model_path, model_name="bert-base-uncased"):
    """
    Load a trained SimCSE model.

    Args:
        model_path: Path to the saved model checkpoint
        model_name: Base model name (e.g., "bert-base-uncased")

    Returns:
        Loaded SimCSE model
    """
    model = SimCSEModel(model_name)

    if Path(model_path).exists():
        print(f"✅ Loading trained SimCSE model from {model_path}")
        try:
            # Load the trained model state dict
            state_dict = torch.load(model_path, map_location="cpu")
            model.load_state_dict(state_dict)
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"⚠️  Error loading model: {e}")
    else:
        print(f"⚠️  Model not found at {model_path}")

    return model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate embeddings using trained SimCSE model"
    )
    parser.add_argument("text", nargs="+", help="Text(s) to embed")
    parser.add_argument(
        "--model", default="models/simcse_model.pt", help="Path to trained SimCSE model"
    )
    parser.add_argument(
        "--model-name", default="bert-base-uncased", help="Base model name"
    )
    return parser.parse_args()


def generate_embeddings(args, device):
    """Generate embeddings and show results based on input type."""
    texts = args.text

    print("🤖 Loading model: {}".format(args.model_name))
    model = load_model(args.model, args.model_name).to(device)
    model.eval()

    # Generate embeddings
    print("🔍 Generating embeddings...")
    embeddings = model.encode(texts, device)

    # Print results
    print("\n📊 Results:")
    print("   Embedding shape: {}".format(embeddings.shape))
    print("   Embedding dimension: {}".format(embeddings.shape[1]))

    # Handle different input scenarios
    if len(texts) == 1:
        # Single text - just show embedding info
        text = texts[0]
        emb = embeddings[0]
        text_preview = text[:100] + ("..." if len(text) > 100 else "")
        print(f"\n📝 Text: {text_preview}")
        print("   Embedding norm: {:.4f}".format(emb.norm().item()))
        print("   First 10 values: {}".format(emb[:10].cpu().numpy()))

    elif len(texts) == 2:
        # Two texts - show individual embeddings and similarity
        print("\n📝 Individual Embeddings:")
        for i, (text, emb) in enumerate(zip(texts, embeddings)):
            text_preview = text[:100] + ("..." if len(text) > 100 else "")
            print(f"\n   Text {i+1}: {text_preview}")
            print("   Embedding norm: {:.4f}".format(emb.norm().item()))

        # Calculate similarity
        similarity = torch.dot(embeddings[0], embeddings[1]).item()
        print(f"\n🔗 Similarity between texts: {similarity:.4f}")
        print(f"   (Cosine similarity: {similarity:.4f})")

    else:
        # Multiple texts - show sample embeddings and similarity matrix
        print("\n📝 Sample Embeddings:")
        for i, (text, emb) in enumerate(zip(texts[:3], embeddings[:3])):
            text_preview = text[:100] + ("..." if len(text) > 100 else "")
            print(f"\n   Text {i+1}: {text_preview}")
            print("   Embedding norm: {:.4f}".format(emb.norm().item()))

        # Show similarity matrix
        print("\n🔗 Computing similarity matrix...")
        similarity_matrix = torch.mm(embeddings, embeddings.t())

        print("\n📊 Similarity Matrix:")
        print("=" * 80)
        for i, text1 in enumerate(texts):
            for j, text2 in enumerate(texts):
                sim = similarity_matrix[i, j].item()
                print(f"{sim:.3f}", end="\t")
            print(f" | {text1[:30]}...")

        # Find most similar pairs
        print("\n🔗 Most Similar Pairs:")
        print("=" * 60)
        similarities = []
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                sim = similarity_matrix[i, j].item()
                similarities.append((sim, i, j))

        # Sort by similarity (descending)
        similarities.sort(reverse=True)

        for sim, i, j in similarities[:5]:  # Show top 5
            print(f"Similarity {sim:.3f}:")
            print(f"  '{texts[i]}'")
            print(f"  '{texts[j]}'")
            print()

    return embeddings


def main():
    args = parse_args()

    print("📝 Processing {} text(s)".format(len(args.text)))

    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 Using CUDA GPU")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple Silicon GPU
        print("🍎 Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("💻 Using CPU")

    generate_embeddings(args, device)


if __name__ == "__main__":
    main()
