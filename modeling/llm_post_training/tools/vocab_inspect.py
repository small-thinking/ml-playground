#!/usr/bin/env python3
"""
Vocabulary Inspector for HuggingFace Models

Inspect model vocabularies and find similar tokens.

Usage:
    python vocab_inspect.py --model-path meta-llama/Llama-3.2-3B \\
        --query "dog" --top-k 5
    python vocab_inspect.py --model-path meta-llama/Llama-3.2-3B --tokenize "Hello, world!"
    python vocab_inspect.py --model-path distilbert-base-uncased --tokenize "The quick brown fox"
"""

import argparse
from typing import List, Dict, Tuple, Optional
import numpy as np

import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


class VocabularyInspector:
    """Inspect model vocabularies and find similar tokens."""

    def __init__(self, model_path: str, device: str = "auto"):
        """Initialize vocabulary inspector."""
        self.model_path = model_path
        self.device = self._get_device(device)
        self.tokenizer = None
        self.model = None
        self.vocab = None
        self.token_embeddings = None

    def _get_device(self, device: str) -> str:
        """Determine the best device to use."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def load_model(self, load_embeddings: bool = True) -> None:
        """
        Load the tokenizer and optionally the model for embeddings.

        Args:
            load_embeddings: Whether to load the model for embedding
            computation
        """
        print(f"Loading model from: {self.model_path}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )

        # Load vocabulary
        self.vocab = self.tokenizer.get_vocab()
        print(f"Loaded vocabulary with {len(self.vocab)} tokens")

        if load_embeddings:
            try:
                # Load model for embeddings
                self.model = AutoModel.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                    torch_dtype=(
                        torch.float16 if self.device == "cuda" else torch.float32
                    ),
                ).to(self.device)
                self.model.eval()
                print(f"Loaded model on device: {self.device}")
            except Exception as e:
                print(f"Warning: Could not load model for embeddings: {e}")
                print("Will use token-based similarity instead")
                self.model = None

    def get_token_embeddings(self, tokens: List[str]) -> np.ndarray:
        """
        Get embeddings for a list of tokens.

        Args:
            tokens: List of tokens to embed

        Returns:
            Array of token embeddings
        """
        if self.model is None:
            raise ValueError(
                "Model not loaded. Call load_model(load_embeddings=True) " "first."
            )

        # Tokenize and get embeddings
        inputs = self.tokenizer(
            tokens, return_tensors="pt", padding=True, truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use the last hidden state and take the first token
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        return embeddings

    def find_similar_tokens_embedding(
        self, query_token: str, top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Find similar tokens using model embeddings.

        Args:
            query_token: Token to find similarities for
            top_k: Number of similar tokens to return

        Returns:
            List of (token, similarity_score) tuples
        """
        if self.model is None:
            raise ValueError("Model not loaded for embeddings")

        # Get all vocabulary tokens
        vocab_tokens = list(self.vocab.keys())

        # Get embeddings for query and all vocab tokens
        query_embedding = self.get_token_embeddings([query_token])[0]
        vocab_embeddings = self.get_token_embeddings(vocab_tokens)

        # Compute cosine similarities
        similarities = cosine_similarity([query_embedding], vocab_embeddings)[0]

        # Get top-k similar tokens
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            token = vocab_tokens[idx]
            similarity = similarities[idx]
            results.append((token, float(similarity)))

        return results

    def find_similar_tokens_text(
        self, query_token: str, top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Find similar tokens using text-based similarity (TF-IDF).

        Args:
            query_token: Token to find similarities for
            top_k: Number of similar tokens to return

        Returns:
            List of (token, similarity_score) tuples
        """
        # Get all vocabulary tokens
        vocab_tokens = list(self.vocab.keys())

        # Create documents for TF-IDF (each token as a document)
        documents = [f"{token} {token}" for token in vocab_tokens]

        # Fit TF-IDF vectorizer
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3))
        tfidf_matrix = vectorizer.fit_transform(documents)

        # Transform query token
        query_vector = vectorizer.transform([f"{query_token} {query_token}"])

        # Compute similarities
        similarities = cosine_similarity(query_vector, tfidf_matrix)[0]

        # Get top-k similar tokens
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            token = vocab_tokens[idx]
            similarity = similarities[idx]
            results.append((token, float(similarity)))

        return results

    def find_similar_tokens(
        self, query_token: str, top_k: int = 10, method: str = "auto"
    ) -> List[Tuple[str, float]]:
        """
        Find similar tokens using the specified method.

        Args:
            query_token: Token to find similarities for
            top_k: Number of similar tokens to return
            method: Method to use ('embedding', 'text', or 'auto')

        Returns:
            List of (token, similarity_score) tuples
        """
        if method == "auto":
            # Try embedding method first, fallback to text method
            if self.model is not None:
                try:
                    return self.find_similar_tokens_embedding(query_token, top_k)
                except Exception as e:
                    print(f"Embedding method failed: {e}")
                    print("Falling back to text-based similarity")
                    return self.find_similar_tokens_text(query_token, top_k)
            else:
                return self.find_similar_tokens_text(query_token, top_k)
        elif method == "embedding":
            return self.find_similar_tokens_embedding(query_token, top_k)
        elif method == "text":
            return self.find_similar_tokens_text(query_token, top_k)
        else:
            raise ValueError(f"Unknown method: {method}")

    def list_tokens(
        self, limit: Optional[int] = None, pattern: Optional[str] = None
    ) -> List[str]:
        """
        List tokens in the vocabulary.

        Args:
            limit: Maximum number of tokens to return
            pattern: Optional pattern to filter tokens (substring
                match)

        Returns:
            List of tokens
        """
        tokens = list(self.vocab.keys())

        if pattern:
            tokens = [token for token in tokens if pattern.lower() in token.lower()]

        if limit:
            tokens = tokens[:limit]

        return tokens

    def get_token_info(self, token: str) -> Dict:
        """
        Get information about a specific token.

        Args:
            token: Token to get info for

        Returns:
            Dictionary with token information
        """
        if token not in self.vocab:
            return {"error": f"Token '{token}' not found in vocabulary"}

        token_id = self.vocab[token]

        info = {
            "token": token,
            "token_id": token_id,
            "length": len(token),
            "is_special": token.startswith("<") and token.endswith(">"),
            "is_punctuation": token in ".,!?;:()[]{}",
            "is_digit": token.isdigit(),
            "is_alpha": token.isalpha(),
        }

        return info

    def tokenize_query(self, query: str) -> List[Dict]:
        """
        Tokenize an input query and return detailed information about
        each token.

        Args:
            query: Input text to tokenize

        Returns:
            List of dictionaries containing token information
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call load_model() first.")

        # Tokenize the input
        tokens = self.tokenizer.tokenize(query)

        # Get token IDs
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # Get detailed info for each token
        token_info_list = []
        for i, (token, token_id) in enumerate(zip(tokens, token_ids)):
            # Get basic token info
            basic_info = self.get_token_info(token)

            # Add additional tokenization info
            token_info = {
                "position": i,
                "token": token,
                "token_id": token_id,
                "length": len(token),
                "is_special": token.startswith("<") and token.endswith(">"),
                "is_punctuation": token in ".,!?;:()[]{}",
                "is_digit": token.isdigit(),
                "is_alpha": token.isalpha(),
                "is_whitespace": token.isspace(),
                "is_subword": token.startswith("##") or token.startswith("▁"),
                "is_unknown": token == self.tokenizer.unk_token,
                "is_padding": token == self.tokenizer.pad_token,
                "is_separator": token
                in [
                    self.tokenizer.sep_token,
                    self.tokenizer.eos_token,
                    self.tokenizer.bos_token,
                ],
            }

            # Add any errors from basic info
            if "error" in basic_info:
                token_info["error"] = basic_info["error"]

            token_info_list.append(token_info)

        return token_info_list


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Vocabulary Inspector for HuggingFace Models"
    )
    parser.add_argument("--model-path", required=True, help="Path to HuggingFace model")
    parser.add_argument("--query", help="Token to find similar tokens for")
    parser.add_argument(
        "--top-k", type=int, default=10, help="Number of similar tokens to return"
    )
    parser.add_argument(
        "--method",
        choices=["auto", "embedding", "text"],
        default="auto",
        help="Method for finding similar tokens",
    )
    parser.add_argument(
        "--list-tokens", action="store_true", help="List tokens in vocabulary"
    )
    parser.add_argument("--limit", type=int, help="Limit number of tokens to list")
    parser.add_argument("--pattern", help="Pattern to filter tokens when listing")
    parser.add_argument("--tokenize", help="Text to tokenize and analyze")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use",
    )
    parser.add_argument(
        "--no-embeddings", action="store_true", help="Don't load model for embeddings"
    )

    args = parser.parse_args()

    # Initialize inspector
    tool = VocabularyInspector(args.model_path, device=args.device)
    tool.load_model(load_embeddings=not args.no_embeddings)

    # Handle different operations
    if args.tokenize:
        print(f"\nTokenizing: '{args.tokenize}'")
        print("=" * 60)

        try:
            token_info_list = tool.tokenize_query(args.tokenize)

            # Print header
            print(
                f"{'Pos':<4} {'Token':<20} {'ID':<8} {'Len':<4} "
                f"{'Type':<12} {'Flags'}"
            )
            print("-" * 60)
            for token_info in token_info_list:
                # Determine token type
                token_type = "unknown"
                if token_info["is_special"]:
                    token_type = "special"
                elif token_info["is_punctuation"]:
                    token_type = "punctuation"
                elif token_info["is_digit"]:
                    token_type = "digit"
                elif token_info["is_alpha"]:
                    token_type = "alpha"
                elif token_info["is_whitespace"]:
                    token_type = "whitespace"
                elif token_info["is_subword"]:
                    token_type = "subword"

                # Build flags string
                flags = []
                if token_info["is_unknown"]:
                    flags.append("UNK")
                if token_info["is_padding"]:
                    flags.append("PAD")
                if token_info["is_separator"]:
                    flags.append("SEP")
                if token_info["is_subword"]:
                    flags.append("SUB")

                flags_str = ",".join(flags) if flags else ""

                # Format token for display (handle special characters)
                display_token = repr(token_info["token"])[1:-1]

                print(
                    f"{token_info['position']:<4} {display_token:<20} "
                    f"{token_info['token_id']:<8} {token_info['length']:<4} "
                    f"{token_type:<12} {flags_str}"
                )

            print(f"\nTotal tokens: {len(token_info_list)}")

        except Exception as e:
            print(f"Error tokenizing: {e}")

    elif args.query:
        print(f"\nSimilar to '{args.query}':")

        # Show query token info compactly
        query_info = tool.get_token_info(args.query)
        if "error" not in query_info:
            print(
                f"Query: {args.query} "
                f"(id:{query_info['token_id']}, "
                f"len:{query_info['length']}, "
                f"alpha:{query_info['is_alpha']})"
            )

        similar_tokens = tool.find_similar_tokens(
            args.query, top_k=args.top_k, method=args.method
        )

        for i, (token, score) in enumerate(similar_tokens, 1):
            token_info = tool.get_token_info(token)
            if "error" not in token_info:
                print(
                    f"{i:2d}. {token:<15} {score:.3f} "
                    f"(id:{token_info['token_id']}, "
                    f"len:{token_info['length']}, "
                    f"alpha:{token_info['is_alpha']})"
                )
            else:
                print(f"{i:2d}. {token:<15} {score:.3f} (error)")

    elif args.list_tokens:
        print("\nTokens:")
        tokens = tool.list_tokens(limit=args.limit, pattern=args.pattern)
        for i, token in enumerate(tokens, 1):
            print(f"{i:3d}. {token}")
        print(f"({len(tokens)} tokens)")

    else:
        print("No operation specified. Use --tokenize, --query, or " "--list-tokens")


if __name__ == "__main__":
    main()
