"""asena base model architecture — decoder-only with RoPE/RMSNorm/SwiGLU/GQA (spec §4.3).

AGENT-EDITABLE (Tier 2). See factory/guards.py:FORBIDDEN_IMPORTS for the
patterns enforced pre-GPU. Do not introduce any of them here.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AsenaConfig:
    vocab_size: int
    n_layers: int
    n_embd: int
    n_head: int
    n_kv_heads: int
    mlp_ratio: float
    rope_theta: float
    tie_embeddings: bool
    init_std: float
    max_seq_len: int


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * norm.to(x.dtype)) * self.scale


def _rope_cache(head_dim: int, max_seq_len: int, theta: float, device, dtype):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(pos, inv_freq)  # [T, head_dim/2]
    cos = freqs.cos().to(dtype)
    sin = freqs.sin().to(dtype)
    return cos, sin


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, T, D]; cos/sin: [T, D/2]
    x1, x2 = x.chunk(2, dim=-1)
    cos = cos[None, None, : x.size(-2), :]
    sin = sin[None, None, : x.size(-2), :]
    rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated


class Attention(nn.Module):
    def __init__(self, cfg: AsenaConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        assert cfg.n_head % cfg.n_kv_heads == 0
        self.n_head = cfg.n_head
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.n_embd // cfg.n_head
        self.q_proj = nn.Linear(cfg.n_embd, cfg.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.n_embd, cfg.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.n_embd, cfg.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        rep = self.n_head // self.n_kv_heads
        if rep > 1:
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim)
        return self.o_proj(y)


class SwiGLU(nn.Module):
    def __init__(self, cfg: AsenaConfig):
        super().__init__()
        hidden = int(cfg.n_embd * cfg.mlp_ratio)
        self.gate = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.up = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: AsenaConfig):
        super().__init__()
        self.ln1 = RMSNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.ln2 = RMSNorm(cfg.n_embd)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        x = x + self.mlp(self.ln2(x))
        return x


class AsenaModel(nn.Module):
    def __init__(self, cfg: AsenaConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = RMSNorm(cfg.n_embd)
        if cfg.tie_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() >= 2:
                nn.init.normal_(p, mean=0.0, std=self.cfg.init_std)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        assert T <= self.cfg.max_seq_len
        x = self.embed(idx)
        head_dim = self.cfg.n_embd // self.cfg.n_head
        cos, sin = _rope_cache(head_dim, T, self.cfg.rope_theta, idx.device, x.dtype)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        x = self.ln_f(x)
        if self.lm_head is None:
            return F.linear(x, self.embed.weight)
        return self.lm_head(x)
