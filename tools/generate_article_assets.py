"""Generate diagrams + tables for the Fuzuli v0.1 article.

Reads experiments.sqlite and data/raw/*.parquet; writes PNG diagrams and a
machine-readable summary.json into docs/article-fuzuli-v0.1/assets/.

Usage:
    .venv/bin/python tools/generate_article_assets.py
"""
from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

ASSETS = Path("docs/article-fuzuli-v0.1/assets")
ASSETS.mkdir(parents=True, exist_ok=True)

# Common style — neutral, publication-friendly.
plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 160,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "legend.frameon": False,
})

# Era / genre color palettes — subdued, not too saturated.
ERA_COLORS = {
    "classical":    "#5b8db8",   # blue-gray
    "tanzimat":     "#c9a877",   # ochre
    "late_ottoman": "#a76b6b",   # muted red
}
OUTCOME_COLORS = {
    "accept":       "#5e8b5e",   # green
    "reject_eval":  "#a76b6b",   # red
    "reject_smoke": "#c9a877",   # ochre
    "error":        "#888888",
}


# ---------------------------------------------------------------------------
# Corpus diagrams
# ---------------------------------------------------------------------------

def load_corpus() -> tuple[list[dict], list[dict]]:
    anadolu = pq.read_table("data/raw/anadolu.parquet").to_pylist()
    evliya = pq.read_table("data/raw/evliya.parquet").to_pylist()
    return anadolu + evliya, anadolu + evliya


def fig_corpus_era(rows: list[dict]) -> None:
    """Pie chart of era distribution (page count)."""
    counts = Counter(r["era"] for r in rows)
    eras = ["classical", "tanzimat", "late_ottoman"]
    sizes = [counts.get(e, 0) for e in eras]
    colors = [ERA_COLORS[e] for e in eras]
    labels = [f"{e}\n{counts.get(e, 0):,} pages" for e in eras]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    wedges, _texts, _autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct="%1.1f%%",
        startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        textprops={"fontsize": 10},
    )
    ax.set_title("Corpus composition by era (post-cleaning, 24,253 pages)")
    fig.tight_layout()
    fig.savefig(ASSETS / "corpus_era.png", bbox_inches="tight")
    plt.close(fig)


def fig_corpus_genre(rows: list[dict]) -> None:
    """Bar chart of genre distribution (page count)."""
    counts = Counter(r["genre"] for r in rows)
    order = ["newspaper", "literary", "other", "poetry", "religious", "official", "legal"]
    genres = [g for g in order if counts.get(g, 0) > 0]
    sizes = [counts[g] for g in genres]

    fig, ax = plt.subplots(figsize=(7, 3.6))
    bars = ax.bar(genres, sizes, color="#5b8db8", edgecolor="white")
    for bar, n in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
                f"{n:,}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Pages")
    ax.set_title("Corpus composition by genre")
    ax.set_ylim(0, max(sizes) * 1.15)
    fig.tight_layout()
    fig.savefig(ASSETS / "corpus_genre.png", bbox_inches="tight")
    plt.close(fig)


def fig_top_documents(rows: list[dict], n: int = 20) -> None:
    """Horizontal bar chart of top-N documents by character count."""
    char_by_pdf = {}
    for r in rows:
        char_by_pdf[r["source_pdf"]] = char_by_pdf.get(r["source_pdf"], 0) + r["length_chars"]
    top = sorted(char_by_pdf.items(), key=lambda x: -x[1])[:n]

    # Shorten labels for readability — drop trailing .pdf and very long names.
    def short(name: str) -> str:
        s = name.replace(".pdf", "")
        if len(s) > 50:
            s = s[:47] + "…"
        return s
    labels = [short(p) for p, _ in top][::-1]
    sizes = [c / 1e6 for _, c in top][::-1]

    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.barh(labels, sizes, color="#5b8db8", edgecolor="white")
    ax.set_xlabel("Million chars (post-cleaning)")
    ax.set_title(f"Top {n} documents by character count")
    fig.tight_layout()
    fig.savefig(ASSETS / "corpus_top_documents.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Experiment diagrams (from experiments.sqlite)
# ---------------------------------------------------------------------------

def load_experiments() -> list[dict]:
    con = sqlite3.connect("experiments.sqlite")
    con.row_factory = sqlite3.Row
    return [dict(r) for r in con.execute("SELECT * FROM experiments ORDER BY id")]


def fig_score_progression(rows: list[dict]) -> None:
    """4-panel chart of metric values over experiment id, accepts highlighted."""
    # Ignore reject_smoke rows (their scores are 0,0,0,0 dummies).
    valid = [r for r in rows if r["outcome"] != "reject_smoke"]
    accepts = [r for r in rows if r["outcome"] == "accept"]

    metrics = [
        ("score_ppl_bpb", "Heldout PPL (bits/byte)", "#5b8db8"),
        ("score_lexicon", "Lexicon score (lower = more Ottoman flavor)", "#a76b6b"),
        ("score_flatness", "Modern-Turkish flatness", "#c9a877"),
        ("score_smoke", "Smoke fail rate", "#5e8b5e"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ax, (col, label, color) in zip(axes.flat, metrics):
        xs = [r["id"] for r in valid]
        ys = [r[col] for r in valid]
        # Light dots only — no connecting lines (too noisy with 300+ points).
        ax.scatter(xs, ys, c=color, s=10, alpha=0.35, edgecolors="none",
                   label="all experiments")
        # Trace the accepted-baseline trajectory as a step plot.
        ax_xs = [r["id"] for r in accepts]
        ax_ys = [r[col] for r in accepts]
        if ax_xs:
            ax.step(ax_xs, ax_ys, where="post", color="#333", lw=1.6,
                    alpha=0.75, label="baseline trajectory")
            ax.scatter(ax_xs, ax_ys, c="#5e8b5e", s=110, marker="*",
                       edgecolors="black", linewidths=0.7,
                       label="accepted", zorder=5)
        ax.set_ylabel(label, fontsize=9)
        ax.legend(loc="best", fontsize=8)
    axes[1, 0].set_xlabel("Experiment id")
    axes[1, 1].set_xlabel("Experiment id")
    fig.suptitle("All 4 evaluators across the autoresearch session — "
                 "stars mark accepted experiments", fontsize=12, y=1.00)
    fig.tight_layout()
    fig.savefig(ASSETS / "score_progression.png", bbox_inches="tight")
    plt.close(fig)


def fig_accept_rate(rows: list[dict]) -> None:
    """Rolling accept rate over experiment id."""
    window = 20
    xs, ys = [], []
    for i in range(len(rows)):
        lo = max(0, i - window + 1)
        chunk = rows[lo:i + 1]
        rate = sum(1 for r in chunk if r["outcome"] == "accept") / len(chunk)
        xs.append(rows[i]["id"])
        ys.append(rate * 100)

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(xs, ys, color="#5b8db8", lw=1.6)
    ax.fill_between(xs, ys, alpha=0.15, color="#5b8db8")
    ax.set_ylabel(f"Accept rate (rolling {window}-experiment window, %)")
    ax.set_xlabel("Experiment id")
    ax.set_ylim(0, max(35, max(ys) + 5))
    ax.set_title("Accept rate collapses after the first ~20 experiments")
    fig.tight_layout()
    fig.savefig(ASSETS / "accept_rate.png", bbox_inches="tight")
    plt.close(fig)


def _bucket_reason(reason: str | None) -> str:
    if not reason:
        return "other"
    r = reason.lower()
    if "ppl_bpb" in r:
        return "regression in heldout PPL"
    if "lexicon" in r:
        return "regression in lexicon"
    if "flatness" in r:
        return "regression in flatness"
    if "smoke" in r and ("import" in r or "exec" in r):
        return "smoke crash (bad patch)"
    if "smoke" in r:
        return "regression in smoke"
    return "other"


def fig_reject_reasons(rows: list[dict]) -> None:
    """Bar chart of why experiments were rejected (grouped)."""
    rejects = [r for r in rows if r["outcome"] != "accept"]
    counts = Counter(_bucket_reason(r["reject_reason"]) for r in rejects)
    order = [
        "regression in heldout PPL",
        "regression in lexicon",
        "regression in flatness",
        "regression in smoke",
        "smoke crash (bad patch)",
        "other",
    ]
    cats = [c for c in order if counts.get(c, 0) > 0]
    sizes = [counts[c] for c in cats]

    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    bars = ax.barh(cats, sizes, color="#a76b6b", edgecolor="white")
    for bar, n in zip(bars, sizes):
        ax.text(bar.get_width() + max(sizes) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{n}", va="center", fontsize=9)
    ax.set_xlabel("Experiment count")
    ax.set_title(f"Why kimi's {sum(sizes)} non-accepts were rejected")
    ax.invert_yaxis()
    ax.set_xlim(0, max(sizes) * 1.12)
    fig.tight_layout()
    fig.savefig(ASSETS / "reject_reasons.png", bbox_inches="tight")
    plt.close(fig)


def fig_seed_variance() -> None:
    """The seed-lottery diagram — same recipe, lex score swings 2.85 → 9.96.

    Hardcoded from the user's transcript since the ledger doesn't carry seed
    metadata. Five seeds: 42 (baseline), 123, 1, 100, 999.
    """
    seeds = [42, 123, 1, 100, 999]
    lex_scores = [2.847, 2.824, 9.962, 3.011, None]
    # 999 was in flight when kimi was interrupted — leave a "?" marker.
    fig, ax = plt.subplots(figsize=(7, 3.8))
    xs = list(range(len(seeds)))
    valid_xs = [x for x, y in zip(xs, lex_scores) if y is not None]
    valid_ys = [y for y in lex_scores if y is not None]
    bars = ax.bar(valid_xs, valid_ys,
                  color=["#5e8b5e" if y < 4 else "#a76b6b" for y in valid_ys],
                  edgecolor="white")
    for bar, y in zip(bars, valid_ys):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                f"{y:.2f}", ha="center", fontsize=9)
    ax.axhline(2.847, color="#888", lw=0.8, linestyle="--", alpha=0.7)
    ax.text(len(seeds) - 0.5, 2.847 + 0.1, "baseline (seed 42)",
            ha="right", fontsize=8, color="#666")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(s) for s in seeds])
    ax.set_xlabel("torch.manual_seed value")
    ax.set_ylabel("Lexicon score (lower = more Ottoman)")
    ax.set_title("Same recipe, different seed: lexicon swings ~3× — the data-starved signature")
    ax.set_ylim(0, 11)
    fig.tight_layout()
    fig.savefig(ASSETS / "seed_variance.png", bbox_inches="tight")
    plt.close(fig)


def fig_system_architecture() -> None:
    """Boxes-and-arrows diagram of factory ↔ kimi ↔ ledger ↔ git."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.grid(False)

    # Boxes
    def box(x, y, w, h, label, color):
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="#333",
                             linewidth=1.4, alpha=0.85)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=10, weight="bold")

    box(0.3, 5.0, 2.4, 1.4, "LLM Agent\n(researcher)", "#c9a877")
    box(3.8, 5.0, 2.4, 1.4, "Factory\n(orchestrator)", "#5b8db8")
    box(7.3, 5.0, 2.4, 1.4, "GPU\n(sprint training)", "#5e8b5e")
    box(0.3, 0.6, 2.4, 1.4, "agent/prompts/\nrun-autoresearch.md\n(driving prompt)", "#dcdcdc")
    box(3.8, 0.6, 2.4, 1.4, "experiments.sqlite\n(ledger)", "#dcdcdc")
    box(7.3, 0.6, 2.4, 1.4, "git\nbranches + main\nfor accepts", "#dcdcdc")
    box(3.8, 2.8, 2.4, 1.4, "eval/policy.py\nstrict-no-trades", "#a76b6b")

    # Arrows
    def arrow(x1, y1, x2, y2, label="", offset=(0, 0)):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.2, color="#333"))
        if label:
            ax.text((x1 + x2) / 2 + offset[0], (y1 + y2) / 2 + offset[1],
                    label, ha="center", fontsize=8, style="italic", color="#444")

    arrow(2.7, 5.7, 3.8, 5.7, "cli.py train-sprint", offset=(0, 0.18))
    arrow(6.2, 5.7, 7.3, 5.7, "subprocess", offset=(0, 0.18))
    arrow(7.3, 5.5, 6.2, 4.2, "scores", offset=(0.4, 0.2))
    arrow(5.0, 2.8, 5.0, 2.0, "accept/reject", offset=(0.7, 0.0))
    arrow(3.8, 1.3, 2.7, 1.3, "ledger tail / baseline show", offset=(0, -0.25))
    arrow(2.7, 1.3, 0.7, 4.95, "(at start of every cycle)", offset=(-1.05, 0.4))
    arrow(6.2, 1.3, 7.3, 1.3, "branch + commit\non accept",
          offset=(0, -0.3))

    ax.set_title("Autoresearch system architecture",
                 fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(ASSETS / "system_architecture.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Model architecture diagram
# ---------------------------------------------------------------------------

def fig_model_architecture() -> None:
    """Vertical block diagram of one Fuzuli transformer block + the full stack."""
    fig, (ax_block, ax_stack) = plt.subplots(1, 2, figsize=(11, 6.8),
                                              gridspec_kw={"width_ratios": [1.4, 1]})

    # ---- Per-block detail ----
    ax_block.set_xlim(0, 8)
    ax_block.set_ylim(0, 13)
    ax_block.axis("off")
    ax_block.grid(False)

    def block(ax, x, y, w, h, label, color, fontsize=9):
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="#333",
                             linewidth=1.2, alpha=0.9)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fontsize)

    def arrow_down(ax, x, y1, y2):
        ax.annotate("", xy=(x, y2), xytext=(x, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.0, color="#444"))

    def add_circle(ax, x, y, label="+"):
        circ = plt.Circle((x, y), 0.22, facecolor="white", edgecolor="#333", lw=1.2)
        ax.add_patch(circ)
        ax.text(x, y, label, ha="center", va="center", fontsize=10, weight="bold")

    cx = 4.0
    # Sequence top to bottom: input → norm1 → attn → +res → norm2 → mlp → +res → output
    block(ax_block, 1.5, 11.6, 5.0, 0.8, "input  hidden  state", "#dcdcdc")
    arrow_down(ax_block, cx, 11.6, 10.9)
    block(ax_block, 1.5, 10.0, 5.0, 0.9, "RMSNorm  (no bias)", "#e8e0c5")
    arrow_down(ax_block, cx, 10.0, 9.0)
    block(ax_block, 1.0, 7.3, 6.0, 1.7,
          "Multi-head Grouped-Query Attention\n10 Q heads · 4 KV heads · head dim 64\nRoPE positional rotation (θ = 10 000)",
          "#5b8db8", fontsize=8.5)
    arrow_down(ax_block, cx, 7.3, 6.6)
    add_circle(ax_block, cx, 6.4, "+")
    arrow_down(ax_block, 0.7, 11.6 + 0.3, 6.4)  # residual skip line
    ax_block.annotate("", xy=(cx - 0.22, 6.4), xytext=(0.7, 6.4),
                      arrowprops=dict(arrowstyle="->", lw=1.0, color="#444"))
    ax_block.text(0.4, 9.0, "residual", rotation=90, fontsize=8, color="#666")
    arrow_down(ax_block, cx, 6.18, 5.4)
    block(ax_block, 1.5, 4.5, 5.0, 0.9, "RMSNorm  (no bias)", "#e8e0c5")
    arrow_down(ax_block, cx, 4.5, 3.5)
    block(ax_block, 1.0, 1.8, 6.0, 1.7,
          "SwiGLU FFN\n3 matrices: gate · up · down\nmlp_ratio = 2.67  (hidden 1707)",
          "#a76b6b", fontsize=8.5)
    arrow_down(ax_block, cx, 1.8, 1.1)
    add_circle(ax_block, cx, 0.9, "+")
    ax_block.annotate("", xy=(cx - 0.22, 0.9), xytext=(0.7, 0.9),
                      arrowprops=dict(arrowstyle="->", lw=1.0, color="#444"))
    ax_block.plot([0.7, 0.7], [6.4, 0.9], color="#444", lw=1.0)
    ax_block.text(0.4, 3.0, "residual", rotation=90, fontsize=8, color="#666")
    arrow_down(ax_block, cx, 0.65, 0.0)
    ax_block.text(cx, -0.3, "→ next block", ha="center", fontsize=8, style="italic",
                  color="#666")
    ax_block.set_title("One decoder block (12× stacked)", fontsize=11, weight="bold")

    # ---- Full stack ----
    ax_stack.set_xlim(0, 6)
    ax_stack.set_ylim(0, 13)
    ax_stack.axis("off")
    ax_stack.grid(False)

    block(ax_stack, 1, 11.5, 4, 1.1,
          "Token embedding\nvocab 24 000 × 640", "#dcdcdc")
    arrow_down(ax_stack, 3, 11.5, 10.6)
    # Stacked block representation
    for i in range(12):
        y = 4.4 + i * 0.5
        block(ax_stack, 1, y, 4, 0.45, f"decoder block ×  {i + 1}",
              "#5b8db8" if i % 2 == 0 else "#7da4c4", fontsize=7.5)
    arrow_down(ax_stack, 3, 4.35, 3.5)
    block(ax_stack, 1, 2.6, 4, 0.9, "Final RMSNorm", "#e8e0c5")
    arrow_down(ax_stack, 3, 2.6, 1.7)
    block(ax_stack, 1, 0.6, 4, 1.1,
          "LM head (untied)\n640 → 24 000", "#a76b6b")

    ax_stack.set_title("Fuzuli decoder stack\nn_layers = 12, n_embd = 640",
                       fontsize=11, weight="bold")

    fig.suptitle("Architecture: standard post-2023 dense decoder transformer",
                 fontsize=12, weight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(ASSETS / "model_architecture.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Tokenizer compression diagram
# ---------------------------------------------------------------------------

def fig_tokenizer_compression() -> None:
    """Bar chart: chars/token for custom 24k vs cl100k vs o200k on the same Ottoman corpus."""
    tokenizers = ["Fuzuli\ncustom 24k BPE", "OpenAI\ncl100k_base", "OpenAI\no200k_base"]
    chars_per_tok = [3.614, 2.074, 2.380]
    colors = ["#5e8b5e", "#5b8db8", "#5b8db8"]

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    bars = ax.bar(tokenizers, chars_per_tok, color=colors, edgecolor="white", width=0.55)
    for bar, v in zip(bars, chars_per_tok):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{v:.2f}", ha="center", fontsize=10, weight="bold")
    ax.set_ylabel("Characters per token (higher = better compression)")
    ax.set_title("Tokenization efficiency on the Fuzuli training corpus\n"
                 "(28.8 M chars of post-cleaning Latinized Ottoman)",
                 fontsize=11)
    ax.set_ylim(0, 4.4)
    fig.text(0.5, -0.02,
             "Custom Ottoman BPE produces 1.74× fewer tokens than cl100k for the same text — "
             "an Ottoman-trained vocabulary captures Ottoman morphology that general-purpose "
             "tokenizers fragment.",
             ha="center", fontsize=8.5, style="italic", wrap=True)
    fig.tight_layout()
    fig.savefig(ASSETS / "tokenizer_compression.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Scaling landscape: where Fuzuli sits relative to Chinchilla optimum
# ---------------------------------------------------------------------------

def fig_scaling_landscape() -> None:
    """Scatter of params vs training tokens, with Chinchilla-20 line."""
    # name, params (M), unique training tokens (M)
    # All counts are unique-token data-scale (the Chinchilla axis), not seen-with-repetition.
    points = [
        ("GPT-2 (small)",       117,    8_000),     # ~8B tokens, OWT
        ("GPT-3 (175B)",        175_000, 300_000),  # 300B tokens
        ("LLaMA-2 7B",          7_000,  2_000_000), # 2T tokens
        ("Chinchilla 70B",      70_000, 1_400_000), # 1.4T tokens
        ("BLOOM 560M",          560,    341_000),   # 341B
        ("BERTurk (cased)",     110,    3_500),     # ~35GB raw text → ~3.5B tokens (rough)
        ("TinyStories-33M",      33,    480),       # ~480M synthetic tokens
        ("Fuzuli v0.1",          84,    8),         # 8M unique BPE tokens
    ]
    names = [p[0] for p in points]
    params = np.array([p[1] for p in points])
    tokens = np.array([p[2] for p in points])

    fig, ax = plt.subplots(figsize=(9, 5.5))
    # Chinchilla optimum line: tokens = 20 * params
    p_line = np.logspace(2, 6, 100)
    ax.plot(p_line, 20 * p_line, color="#888", lw=1.0, linestyle="--",
            label="Chinchilla optimum (≈ 20 tok/param)")
    ax.fill_between(p_line, 20 * p_line, 100 * 20 * p_line,
                    color="#888", alpha=0.06,
                    label="overtrained (LLaMA territory)")
    ax.fill_between(p_line, 0.01 * 20 * p_line, 20 * p_line,
                    color="#a76b6b", alpha=0.06,
                    label="undertrained (low-resource regime)")

    # Per-point label offsets (xytext multipliers) to avoid collisions.
    label_offsets = {
        "GPT-2 (small)":       (1.20, 0.85),
        "GPT-3 (175B)":        (0.45, 0.75),
        "LLaMA-2 7B":          (0.35, 1.20),
        "Chinchilla 70B":      (1.20, 1.15),
        "BLOOM 560M":          (1.20, 1.10),
        "BERTurk (cased)":     (1.30, 1.30),
        "TinyStories-33M":     (1.30, 0.65),
        "Fuzuli v0.1":         (1.40, 1.55),
    }
    for n, p, t in zip(names, params, tokens):
        is_us = n == "Fuzuli v0.1"
        ax.scatter(p, t,
                   s=240 if is_us else 80,
                   c="#5e8b5e" if is_us else "#5b8db8",
                   marker="*" if is_us else "o",
                   edgecolors="black", linewidths=1.2 if is_us else 0.5,
                   zorder=5)
        ox, oy = label_offsets.get(n, (1.15, 1.05))
        ax.annotate(n, (p, t), xytext=(p * ox, t * oy), fontsize=9,
                    weight="bold" if is_us else "normal",
                    color="#5e8b5e" if is_us else "#333")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Model parameters (millions)")
    ax.set_ylabel("Unique training tokens (millions, log scale)")
    ax.set_title("Where Fuzuli v0.1 sits in the scaling landscape — "
                 "deep in the low-resource regime, but in good company",
                 fontsize=11)
    ax.set_xlim(20, 1e6)
    ax.set_ylim(2, 1e7)
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    fig.savefig(ASSETS / "scaling_landscape.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Roadmap chart
# ---------------------------------------------------------------------------

def fig_roadmap() -> None:
    """Corpus-tokens vs target-params for v0.1 → v0.5 → v1.0."""
    versions = ["v0.1\n(now)", "v0.5\n(M2)", "v1.0\n(M3)"]
    tokens = [8, 30, 50]      # M unique BPE tokens
    params = [84, 120, 150]   # M

    fig, ax1 = plt.subplots(figsize=(7.5, 4))
    x = np.arange(len(versions))
    w = 0.36
    bars1 = ax1.bar(x - w / 2, tokens, w, color="#5b8db8", label="Unique BPE tokens (M)")
    ax1.set_ylabel("Unique BPE tokens (millions)", color="#5b8db8")
    ax1.tick_params(axis="y", colors="#5b8db8")
    ax1.set_ylim(0, max(tokens) * 1.3)
    for bar, v in zip(bars1, tokens):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                 f"{v}M", ha="center", fontsize=9, color="#3d6586")

    ax2 = ax1.twinx()
    ax2.grid(False)
    bars2 = ax2.bar(x + w / 2, params, w, color="#a76b6b", label="Model params (M)")
    ax2.set_ylabel("Model parameters (millions)", color="#a76b6b")
    ax2.tick_params(axis="y", colors="#a76b6b")
    ax2.set_ylim(0, max(params) * 1.3)
    for bar, v in zip(bars2, params):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                 f"{v}M", ha="center", fontsize=9, color="#7d4f4f")
    ax2.spines["top"].set_visible(False)

    ax1.set_xticks(x)
    ax1.set_xticklabels(versions)
    ax1.set_title("Roadmap: corpus tokens and model size scale together")
    fig.tight_layout()
    fig.savefig(ASSETS / "roadmap.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary JSON for the article writer
# ---------------------------------------------------------------------------

def write_summary(corpus_rows: list[dict], experiments: list[dict]) -> None:
    accepts = [r for r in experiments if r["outcome"] == "accept"]
    rej_eval = [r for r in experiments if r["outcome"] == "reject_eval"]
    rej_smoke = [r for r in experiments if r["outcome"] == "reject_smoke"]
    summary: dict[str, Any] = {
        "corpus": {
            "documents": len(set(r["source_pdf"] for r in corpus_rows)),
            "pages": len(corpus_rows),
            "chars": sum(r["length_chars"] for r in corpus_rows),
            "era_breakdown": dict(Counter(r["era"] for r in corpus_rows)),
            "genre_breakdown": dict(Counter(r["genre"] for r in corpus_rows)),
        },
        "experiments": {
            "total": len(experiments),
            "accepts": len(accepts),
            "reject_eval": len(rej_eval),
            "reject_smoke": len(rej_smoke),
            "accept_ids": [r["id"] for r in accepts],
            "first_baseline": {
                k: experiments[0][k] for k in
                ("id", "score_ppl_bpb", "score_lexicon", "score_flatness", "score_smoke")
            },
            "current_baseline": {
                k: accepts[-1][k] for k in
                ("id", "score_ppl_bpb", "score_lexicon", "score_flatness", "score_smoke")
            } if accepts else None,
        },
    }
    (ASSETS / "summary.json").write_text(json.dumps(summary, indent=2, default=str))


def main() -> None:
    print(f"Writing assets to {ASSETS}/")
    corpus_rows, _ = load_corpus()
    experiments = load_experiments()

    fig_corpus_era(corpus_rows)
    print("  ✓ corpus_era.png")
    fig_corpus_genre(corpus_rows)
    print("  ✓ corpus_genre.png")
    fig_top_documents(corpus_rows, n=20)
    print("  ✓ corpus_top_documents.png")
    fig_score_progression(experiments)
    print("  ✓ score_progression.png")
    fig_accept_rate(experiments)
    print("  ✓ accept_rate.png")
    fig_reject_reasons(experiments)
    print("  ✓ reject_reasons.png")
    fig_seed_variance()
    print("  ✓ seed_variance.png")
    fig_system_architecture()
    print("  ✓ system_architecture.png")
    fig_model_architecture()
    print("  ✓ model_architecture.png")
    fig_tokenizer_compression()
    print("  ✓ tokenizer_compression.png")
    fig_scaling_landscape()
    print("  ✓ scaling_landscape.png")
    fig_roadmap()
    print("  ✓ roadmap.png")
    write_summary(corpus_rows, experiments)
    print("  ✓ summary.json")
    print(f"\nDone. {len(list(ASSETS.glob('*.png')))} PNGs + summary in {ASSETS}/")


if __name__ == "__main__":
    main()
