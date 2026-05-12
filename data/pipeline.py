"""Orchestrates Stages 1-4 over a directory of raw parquet files (spec §3.2)."""
from __future__ import annotations
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

from data.schema import validate_raw_parquet
from data.stages import (
    normalize, apply_cleaning_rules, dedup_minhash,
    split_train_heldout, load_cleaning_rules,
)


def run_prepare_data(
    raw_dir: Path,
    out_dir: Path,
    rules_path: Path,
    heldout_pct: int = 2,
) -> dict:
    """Run Stages 1-4 over every parquet in raw_dir; write to out_dir/{train,heldout}/.

    Returns a summary dict: {"in_rows", "after_stage2", "after_dedup", "train", "heldout"}.
    """
    rules = load_cleaning_rules(rules_path)
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    (out_dir / "train").mkdir(parents=True, exist_ok=True)
    (out_dir / "heldout").mkdir(parents=True, exist_ok=True)

    # Stage 1+2: per-file.
    in_rows, after_stage2 = 0, []
    for path in sorted(raw_dir.glob("*.parquet")):
        table = pq.read_table(path)
        validate_raw_parquet(table)
        for row in table.to_pylist():
            in_rows += 1
            text = normalize(row["text"])
            cleaned = apply_cleaning_rules(text, rules)
            if cleaned is None:
                continue
            new_row = dict(row)
            new_row["text"] = cleaned
            new_row["length_chars"] = len(cleaned)
            after_stage2.append(new_row)

    # Stage 3: global dedup.
    kept_indices = dedup_minhash([r["text"] for r in after_stage2], threshold=0.85)
    after_dedup = [after_stage2[i] for i in kept_indices]

    # Stage 4: per-document split.
    train_rows, heldout_rows = split_train_heldout(after_dedup, heldout_pct=heldout_pct)

    if train_rows:
        pq.write_table(pa.Table.from_pylist(train_rows), out_dir / "train" / "part-00000.parquet")
    if heldout_rows:
        pq.write_table(pa.Table.from_pylist(heldout_rows), out_dir / "heldout" / "part-00000.parquet")

    return {
        "in_rows": in_rows,
        "after_stage2": len(after_stage2),
        "after_dedup": len(after_dedup),
        "train": len(train_rows),
        "heldout": len(heldout_rows),
    }
