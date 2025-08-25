#!/usr/bin/env python3
"""
SimCSE Inference Script

Generates embeddings using trained SimCSE model.
Usage:
    uv run modeling/llm_embedding/inference.py "Text 1" "Text 2"
"""

import torch
import argparse
from pathlib import Path

try:
    from .model import SimCSEModel
    from .config import DEVICE
except ImportError:
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_dir)
    from model import SimCSEModel
    from config import DEVICE


def load_model(model_path, model_name="bert-base-uncased"):
    """Load a trained SimCSE model."""
    model = SimCSEModel(model_name)

    if Path(model_path).exists():
        print(f"✅ Loading trained SimCSE model from {model_path}")
        try:
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


def generate_embeddings(args):
    """Generate embeddings and show results."""
    texts = args.text

    print("🤖 Loading model: {}".format(args.model_name))
    model = load_model(args.model, args.model_name).to(DEVICE)
    model.eval()

    print("🔍 Generating embeddings...")
    embeddings = model.encode(texts, DEVICE)

    print("\n📊 Results:")
    print("   Embedding shape: {}".format(embeddings.shape))
    print("   Embedding dimension: {}".format(embeddings.shape[1]))

    if len(texts) == 1:
        text = texts[0]
        emb = embeddings[0]
        text_preview = text[:100] + ("..." if len(text) > 100 else "")
        print(f"\n📝 Text: {text_preview}")
        print("   Embedding norm: {:.4f}".format(emb.norm().item()))
        print("   First 10 values: {}".format(emb[:10].cpu().numpy()))

    elif len(texts) == 2:
        print("\n📝 Individual Embeddings:")
        for i, (text, emb) in enumerate(zip(texts, embeddings)):
            text_preview = text[:100] + ("..." if len(text) > 100 else "")
            print(f"\n   Text {i+1}: {text_preview}")
            print("   Embedding norm: {:.4f}".format(emb.norm().item()))

        similarity = torch.dot(embeddings[0], embeddings[1]).item()
        print(f"\n🔗 Similarity between texts: {similarity:.4f}")

    else:
        try:
            from .utils import evaluate_similarity_matrix
        except ImportError:
            from utils import evaluate_similarity_matrix
        evaluate_similarity_matrix(embeddings, texts)

    return embeddings


def main():
    args = parse_args()
    print("📝 Processing {} text(s)".format(len(args.text)))
    print(f"📱 Using device: {DEVICE}")
    generate_embeddings(args)


if __name__ == "__main__":
    main()
