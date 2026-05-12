"""Smoke-prompt evaluator: fixed prompts with deterministic pass/fail rules (spec §5.1).

Generation is greedy (temperature=0, top_k=1) for reproducibility. Rules are pure
regex + token-count checks — no LLM-as-judge.

IMMUTABLE (Tier 1).
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import yaml
import torch
from tokenizers import Tokenizer

from train.arch import AsenaConfig, AsenaModel

_TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)


@dataclass
class SmokePromptResult:
    prompt_id: str
    generation: str
    passed: bool
    reason: str


@lru_cache(maxsize=4)
def _load_blacklist(path: str) -> frozenset[str]:
    return frozenset(
        line.strip().lower()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    )


def _check_rules(generation: str, rules: dict, blacklist_path: str) -> SmokePromptResult:
    tokens = [t.lower() for t in _TOKEN_RE.findall(generation)]
    n = len(tokens)
    min_tokens = rules.get("min_tokens", 0)
    if n < min_tokens:
        return SmokePromptResult("", generation, False, f"min_tokens: {n} < {min_tokens}")
    if rules.get("no_modern_loanwords", False):
        bl = _load_blacklist(blacklist_path)
        bad = [t for t in tokens if t in bl]
        if bad:
            return SmokePromptResult("", generation, False, f"loanword: {bad[0]}")
    if rules.get("no_repetition_5gram", False):
        if len(tokens) >= 10:
            five_grams = [" ".join(tokens[i:i+5]) for i in range(len(tokens) - 4)]
            if len(five_grams) - len(set(five_grams)) >= 3:
                return SmokePromptResult("", generation, False, "5gram repetition")
    return SmokePromptResult("", generation, True, "ok")


def _load_model(checkpoint_path: Path, device: torch.device) -> tuple[AsenaModel, int]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]; mcfg = cfg["model"]
    model_cfg = AsenaConfig(
        vocab_size=cfg["vocab_size"], n_layers=mcfg["n_layers"], n_embd=mcfg["n_embd"],
        n_head=mcfg["n_head"], n_kv_heads=mcfg["n_kv_heads"],
        mlp_ratio=mcfg["mlp_ratio"], rope_theta=float(mcfg["rope_theta"]),
        tie_embeddings=mcfg["tie_embeddings"], init_std=mcfg["init_std"],
        max_seq_len=mcfg["max_seq_len"],
    )
    model = AsenaModel(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg["vocab_size"]


@torch.inference_mode()
def _greedy_generate(model, tok: Tokenizer, prompt: str, max_new_tokens: int, device) -> str:
    ids = tok.encode(prompt).ids
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    for _ in range(max_new_tokens):
        logits = model(x[:, -model.cfg.max_seq_len:])
        next_id = int(logits[0, -1].argmax())
        x = torch.cat([x, torch.tensor([[next_id]], device=device, dtype=torch.long)], dim=1)
    return tok.decode(x[0].tolist())


def evaluate_smoke_prompts(
    checkpoint_path: Path,
    tokenizer_path: Path,
    prompts_path: Path,
    blacklist_path: Path,
    device: str = "cuda",
) -> tuple[float, list[SmokePromptResult]]:
    """Return (fail_rate, list-of-results)."""
    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    model, _ = _load_model(checkpoint_path, dev)
    tok = Tokenizer.from_file(str(tokenizer_path))
    cfg = yaml.safe_load(Path(prompts_path).read_text())
    results: list[SmokePromptResult] = []
    for entry in cfg["prompts"]:
        gen = _greedy_generate(model, tok, entry["prompt"], entry["max_new_tokens"], dev)
        r = _check_rules(gen, entry["rules"], str(blacklist_path))
        results.append(SmokePromptResult(entry["id"], gen, r.passed, r.reason))
    if not results:
        return 1.0, []
    fail_rate = sum(1 for r in results if not r.passed) / len(results)
    return fail_rate, results
