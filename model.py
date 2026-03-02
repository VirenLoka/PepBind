"""
Transformer encoder for peptide activity classification.

Architecture
────────────
1.  **Chunking**:  The 1280-d mean-pooled ESM2 embedding is reshaped into
    `num_chunks` pseudo-tokens of dimension `chunk_dim` (e.g. 16 × 80).
2.  **Linear projection** from `chunk_dim` → `d_model`.
3.  A learnable **[CLS] token** is optionally prepended (pool="cls").
4.  **Positional embeddings** are added (learnable, not sinusoidal).
5.  `n_layers` of standard **non-causal (bidirectional) Transformer encoder**
    blocks, each consisting of multi-head self-attention + feed-forward.
6.  **Pooling**: the [CLS] representation or mean-pool of all tokens.
7.  A two-layer **MLP classification head** → `num_classes` logits.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


# ── Building blocks ───────────────────────────────────────────────────────────


class MultiHeadSelfAttention(nn.Module):
    """Standard scaled dot-product multi-head self-attention (no causal mask)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = math.sqrt(self.head_dim)

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len, d_model)

        Returns
        -------
        (batch, seq_len, d_model)
        """
        B, S, D = x.shape

        # ── Q, K, V ──────────────────────────────────────────────────────────
        qkv = self.qkv_proj(x)                          # (B, S, 3·D)
        qkv = qkv.reshape(B, S, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)                # (3, B, H, S, head_dim)
        q, k, v = qkv.unbind(dim=0)                     # each (B, H, S, head_dim)

        # ── Attention (no causal mask → full bidirectional) ──────────────────
        attn_weights = (q @ k.transpose(-2, -1)) / self.scale   # (B, H, S, S)
        attn_weights = attn_weights.softmax(dim=-1)
        attn_weights = self.attn_drop(attn_weights)

        out = attn_weights @ v                           # (B, H, S, head_dim)
        out = out.transpose(1, 2).reshape(B, S, D)      # (B, S, D)
        return self.proj_drop(self.out_proj(out))


class FeedForward(nn.Module):
    """Position-wise feed-forward with GELU activation."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerEncoderBlock(nn.Module):
    """Pre-norm Transformer encoder block (more stable training than post-norm)."""

    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff    = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


# ── Main model ────────────────────────────────────────────────────────────────


class PeptideTransformer(nn.Module):
    """
    Transformer encoder that classifies peptides from chunked ESM2 embeddings.

    Parameters are read from a flat namespace or OmegaConf DictConfig under the
    `model` key.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        m = cfg.model

        self.num_chunks = m.num_chunks
        self.chunk_dim  = m.chunk_dim
        self.d_model    = m.d_model
        self.pool_mode  = m.pool            # "cls" or "mean"

        # ── Input projection ──────────────────────────────────────────────────
        self.input_proj = nn.Sequential(
            nn.Linear(m.chunk_dim, m.d_model),
            nn.LayerNorm(m.d_model),
            nn.Dropout(m.dropout),
        )

        # ── [CLS] token (optional) ───────────────────────────────────────────
        total_tokens = m.num_chunks
        if self.pool_mode == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, m.d_model) * 0.02)
            total_tokens += 1

        # ── Positional embeddings ─────────────────────────────────────────────
        self.pos_embed = nn.Parameter(torch.randn(1, total_tokens, m.d_model) * 0.02)
        self.embed_drop = nn.Dropout(m.dropout)

        # ── Transformer encoder stack ─────────────────────────────────────────
        self.encoder = nn.ModuleList([
            TransformerEncoderBlock(m.d_model, m.n_heads, m.d_ff, m.dropout)
            for _ in range(m.n_layers)
        ])
        self.final_norm = nn.LayerNorm(m.d_model)

        # ── Classification head ───────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(m.d_model, m.d_model),
            nn.GELU(),
            nn.Dropout(m.dropout),
            nn.Linear(m.d_model, m.num_classes),
        )

        self._init_weights()

    # ── Weight initialisation ─────────────────────────────────────────────────

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, esm_embed_dim)   — e.g. (B, 1280)

        Returns
        -------
        logits : (batch, num_classes)
        """
        B = x.size(0)

        # ── Chunk into pseudo-tokens ──────────────────────────────────────────
        tokens = x.view(B, self.num_chunks, self.chunk_dim)   # (B, 16, 80)
        tokens = self.input_proj(tokens)                       # (B, 16, d_model)

        # ── Prepend [CLS] if needed ───────────────────────────────────────────
        if self.pool_mode == "cls":
            cls = self.cls_token.expand(B, -1, -1)            # (B, 1, d_model)
            tokens = torch.cat([cls, tokens], dim=1)           # (B, 17, d_model)

        # ── Add positional embeddings ─────────────────────────────────────────
        tokens = self.embed_drop(tokens + self.pos_embed)

        # ── Transformer encoder (non-causal / bidirectional) ──────────────────
        for block in self.encoder:
            tokens = block(tokens)

        tokens = self.final_norm(tokens)

        # ── Pool ──────────────────────────────────────────────────────────────
        if self.pool_mode == "cls":
            pooled = tokens[:, 0]                              # [CLS] representation
        else:
            pooled = tokens.mean(dim=1)                        # mean over all tokens

        # ── Classify ──────────────────────────────────────────────────────────
        return self.classifier(pooled)


# ── Convenience ───────────────────────────────────────────────────────────────

def build_model(cfg: DictConfig) -> PeptideTransformer:
    """Instantiate the model and print a parameter summary."""
    model = PeptideTransformer(cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] PeptideTransformer — {n_params:,} trainable parameters")
    return model


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    from omegaconf import OmegaConf

    cfg = OmegaConf.load("configs/train_config.yaml")
    model = build_model(cfg)

    dummy = torch.randn(4, cfg.model.esm_embed_dim)   # batch of 4
    logits = model(dummy)
    print(f"Input : {dummy.shape}")
    print(f"Output: {logits.shape}")      # expect (4, 2)
    print(f"Logits: {logits}")