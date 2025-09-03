#!/usr/bin/env python3
"""
SimCSE Model Architecture

This module contains the SimCSE model implementation that can be shared
between training and inference scripts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


class SimCSEModel(nn.Module):
    """
    BERT-based embedding model using SimCSE approach.

    This model generates sentence embeddings by applying dropout twice
    to the same input to create positive pairs, then uses contrastive
    learning to learn meaningful representations.
    """

    def __init__(self, model_name: str = "bert-base-uncased", max_length: int = 64):
        """
        Initialize the SimCSE model.

        Args:
            model_name: Name of the pre-trained BERT model to use
            max_length: Maximum sequence length for tokenization
        """
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self.bert = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(
        self,
        anchor_input_ids: torch.Tensor = None,
        anchor_attention_mask: torch.Tensor = None,
        positive_input_ids: torch.Tensor = None,
        positive_attention_mask: torch.Tensor = None,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
    ):
        """
        Forward pass generating embeddings for anchor-positive pairs or single inputs.

        Args:
            anchor_input_ids: Token IDs for anchor sentences
            anchor_attention_mask: Attention mask for anchor sentences
            positive_input_ids: Token IDs for positive sentences
            positive_attention_mask: Attention mask for positive sentences
            input_ids: Token IDs (for backward compatibility)
            attention_mask: Attention mask (for backward compatibility)

        Returns:
            Tuple of two normalized embeddings (emb1, emb2)
        """
        # Handle both anchor-positive pairs and single inputs
        if anchor_input_ids is not None and positive_input_ids is not None:
            # Anchor-positive pair mode (all-nli dataset)
            anchor_out = self.bert(
                input_ids=anchor_input_ids, attention_mask=anchor_attention_mask
            ).last_hidden_state
            positive_out = self.bert(
                input_ids=positive_input_ids, attention_mask=positive_attention_mask
            ).last_hidden_state

            # Mean pooling over sequence dimension
            anchor_mask = anchor_attention_mask.unsqueeze(-1).float()
            positive_mask = positive_attention_mask.unsqueeze(-1).float()

            emb1 = (anchor_out * anchor_mask).sum(1) / anchor_mask.sum(1)
            emb2 = (positive_out * positive_mask).sum(1) / positive_mask.sum(1)
        else:
            # Original SimCSE mode (single input with dropout)
            assert (
                input_ids is not None and attention_mask is not None
            ), "Must provide either anchor/positive pairs or input_ids/attention_mask"

            # Generate two views using dropout for SimCSE positive pairs
            # Each forward pass will have different dropout masks
            out1 = self.bert(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state
            out2 = self.bert(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state

            # Mean pooling over sequence dimension
            mask = attention_mask.unsqueeze(-1).float()
            emb1 = (out1 * mask).sum(1) / mask.sum(1)
            emb2 = (out2 * mask).sum(1) / mask.sum(1)

        # L2 normalization
        emb1 = F.normalize(emb1, p=2, dim=1)
        emb2 = F.normalize(emb2, p=2, dim=1)

        return emb1, emb2

    def single_forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Single forward pass for inference (no duplicate computation).

        Args:
            input_ids: Token IDs from tokenizer
            attention_mask: Attention mask for padding

        Returns:
            Single normalized embedding tensor
        """
        # Single forward pass (more efficient for inference)
        out = self.bert(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state

        # Mean pooling over sequence dimension
        mask = attention_mask.unsqueeze(-1).float()
        emb = (out * mask).sum(1) / mask.sum(1)

        # L2 normalization
        emb = F.normalize(emb, p=2, dim=1)

        return emb

    def encode(self, texts: list, device: torch.device = None) -> torch.Tensor:
        """
        Encode a list of texts into embeddings.

        Args:
            texts: List of text strings to encode
            device: Device to run inference on

        Returns:
            Tensor of shape [batch_size, embedding_dim]
        """
        self.eval()
        if device is None:
            device = next(self.parameters()).device

        with torch.no_grad():
            # Tokenize inputs
            encodings = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(device)

            # Use single forward pass for efficiency - filter parameters
            model_inputs = {
                k: v
                for k, v in encodings.items()
                if k in ["input_ids", "attention_mask"]
            }
            return self.single_forward(**model_inputs)
