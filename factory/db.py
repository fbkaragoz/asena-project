"""SQLite experiment ledger (spec §6.2)."""
from __future__ import annotations
import sqlite3
from dataclasses import dataclass, fields
from pathlib import Path


SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    started_utc     TEXT NOT NULL,
    finished_utc    TEXT,
    git_sha_before  TEXT NOT NULL,
    git_sha_after   TEXT,
    branch_name     TEXT NOT NULL,
    scope           TEXT,
    hypothesis      TEXT,
    diff            TEXT NOT NULL,
    outcome         TEXT NOT NULL,
    reject_reason   TEXT,
    delta_ppl_bpb   REAL, delta_lexicon  REAL, delta_flatness REAL, delta_smoke REAL,
    score_ppl_bpb   REAL, score_lexicon  REAL, score_flatness REAL, score_smoke REAL,
    train_tokens    INTEGER,
    train_steps     INTEGER,
    train_seconds   REAL,
    peak_vram_mb    INTEGER
);
CREATE INDEX IF NOT EXISTS idx_outcome ON experiments(outcome);
CREATE INDEX IF NOT EXISTS idx_scope   ON experiments(scope);

CREATE TABLE IF NOT EXISTS baselines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id),
    git_sha TEXT NOT NULL,
    set_utc TEXT NOT NULL,
    score_ppl_bpb REAL NOT NULL,
    score_lexicon REAL NOT NULL,
    score_flatness REAL NOT NULL,
    score_smoke REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS freeze_locks (
    component TEXT PRIMARY KEY,
    sha256 TEXT NOT NULL,
    frozen_utc TEXT NOT NULL,
    frozen_by TEXT NOT NULL
);
"""


@dataclass
class ExperimentRow:
    started_utc: str
    finished_utc: str | None
    git_sha_before: str
    git_sha_after: str | None
    branch_name: str
    scope: str | None
    hypothesis: str
    diff: str
    outcome: str
    reject_reason: str | None
    delta_ppl_bpb: float; delta_lexicon: float; delta_flatness: float; delta_smoke: float
    score_ppl_bpb: float; score_lexicon: float; score_flatness: float; score_smoke: float
    train_tokens: int; train_steps: int; train_seconds: float; peak_vram_mb: int


class Ledger:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def insert(self, row: ExperimentRow) -> int:
        cols = [f.name for f in fields(row)]
        placeholders = ",".join("?" * len(cols))
        values = [getattr(row, c) for c in cols]
        cur = self._conn.execute(
            f"INSERT INTO experiments ({','.join(cols)}) VALUES ({placeholders})", values
        )
        self._conn.commit()
        return cur.lastrowid

    def list_experiments(self, limit: int = 100) -> list[dict]:
        cur = self._conn.execute(
            "SELECT * FROM experiments ORDER BY id DESC LIMIT ?", (limit,)
        )
        return [dict(r) for r in cur.fetchall()]

    def query(self, scope: str | None = None, outcome: str | None = None,
              limit: int = 50) -> list[dict]:
        sql = "SELECT * FROM experiments WHERE 1=1"
        params: list = []
        if scope is not None:
            sql += " AND scope = ?"; params.append(scope)
        if outcome is not None:
            sql += " AND outcome = ?"; params.append(outcome)
        sql += " ORDER BY id DESC LIMIT ?"; params.append(limit)
        cur = self._conn.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]

    def set_baseline(self, experiment_id: int, git_sha: str, scores: dict[str, float]) -> None:
        from datetime import datetime, timezone
        self._conn.execute(
            "INSERT INTO baselines (experiment_id, git_sha, set_utc, "
            "score_ppl_bpb, score_lexicon, score_flatness, score_smoke) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (experiment_id, git_sha, datetime.now(timezone.utc).isoformat(),
             scores["score_ppl_bpb"], scores["score_lexicon"],
             scores["score_flatness"], scores["score_smoke"]),
        )
        self._conn.commit()

    def get_baseline(self) -> dict | None:
        cur = self._conn.execute(
            "SELECT * FROM baselines ORDER BY id DESC LIMIT 1"
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def close(self) -> None:
        self._conn.close()

    def __del__(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
