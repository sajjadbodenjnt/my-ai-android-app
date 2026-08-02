"""
architecture.py

Lightweight custom causal Transformer core (small-scale) with clinical Guardrail layer.
This is intentionally small for CI/test purposes. Do NOT use directly for clinical decision-making.
"""

from typing import Optional, List
import re
import torch
import torch.nn as nn
import math


class ClinicalGuardrail(Exception):
    pass


class GuardrailLayer:
    """Implements conservative, rule-based checks to reduce risky medical outputs."""
    def __init__(self):
        self.forbidden_phrases = [
            "you have",
            "definitive diagnosis",
            "guaranteed cure",
            "must do",
        ]

    def check(self, text: str) -> List[str]:
        hits = [p for p in self.forbidden_phrases if p in text.lower()]
        return hits

    def enforce(self, text: str, evidence: Optional[List[str]] = None) -> str:
        hits = self.check(text)
        if hits:
            raise ClinicalGuardrail(f"Guardrail blocked output containing: {hits}")
        if evidence is None or len(evidence) == 0:
            # If text contains recommendation words, require evidence
            if any(w in text.lower() for w in ["recommend", "prescribe", "treat", "should"]):
                return ("I cannot provide clinical recommendations without referencing validated clinical sources. "
                        "Please consult a qualified health professional.")
        return text


class SmallCausalTransformer(nn.Module):
    """A tiny causal Transformer implementation for development and testing.

    This model is intentionally small (~few MB) and should be used for fast local tests only.
    """
    def __init__(self, vocab_size: int = 50257, d_model: int = 128, n_head: int = 4, n_layer: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(1024, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=d_model*4, dropout=dropout)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.pos_embedding(positions)
        # TransformerEncoder expects shape (S, N, E)
        x = x.permute(1, 0, 2)
        for layer in self.layers:
            x = layer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits


if __name__ == "__main__":
    # quick smoke test
    model = SmallCausalTransformer()
    x = torch.zeros((1, 8), dtype=torch.long)
    logits = model(x)
    print("Logits shape:", logits.shape)
