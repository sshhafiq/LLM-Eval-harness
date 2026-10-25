"""SQLite-backed run registry with baseline comparison and regression detection.

Every experiment run can be recorded with its aggregate metric scores; later,
:meth:`RunRegistry.compare` checks a new run against a baseline and flags any
metric whose mean dropped by more than a threshold.  This is the primitive
behind "did the model get worse?" CI gates.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Union


_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    model TEXT NOT NULL,
    dataset TEXT NOT NULL,
    created_at TEXT NOT NULL,
    config TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS scores (
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    metric TEXT NOT NULL,
    mean REAL NOT NULL,
    ci_low REAL,
    ci_high REAL,
    PRIMARY KEY (run_id, metric)
);
CREATE INDEX IF NOT EXISTS idx_scores_run ON scores(run_id);
"""


@dataclass
class RunRecord:
    """One recorded experiment run with its aggregate scores."""

    id: int
    name: str
    model: str
    dataset: str
    created_at: str
    config: Dict[str, object] = field(default_factory=dict)
    metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # metrics: name -> {"mean": ..., "ci_low": ..., "ci_high": ...}


@dataclass
class Regression:
    """A metric that regressed between a baseline run and a current run."""

    metric: str
    baseline_mean: float
    current_mean: float
    drop: float  # baseline_mean - current_mean, always > threshold here
    threshold: float
    relative: bool

    @property
    def drop_pct(self) -> float:
        if self.baseline_mean == 0:
            return 0.0
        return 100.0 * self.drop / abs(self.baseline_mean)


class RunRegistry:
    """Persistent registry of experiment runs in a SQLite database."""

    def __init__(self, db_path: Union[str, Path]) -> None:
        self.db_path = Path(db_path)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.execute("PRAGMA foreign_keys = ON")

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "RunRegistry":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def record_run(
        self,
        *,
        name: str,
        model: str,
        dataset: str,
        aggregate: Dict[str, float],
        ci: Optional[Dict[str, tuple[float, float]]] = None,
        config: Optional[Dict[str, object]] = None,
    ) -> int:
        """Persist a run and its aggregate metric scores; return the run id."""
        ci = ci or {}
        created_at = datetime.now(timezone.utc).isoformat()
        cur = self._conn.execute(
            "INSERT INTO runs (name, model, dataset, created_at, config)"
            " VALUES (?, ?, ?, ?, ?)",
            (name, model, dataset, created_at, json.dumps(config or {})),
        )
        run_id = cur.lastrowid
        assert run_id is not None
        for metric, mean in aggregate.items():
            low, high = ci.get(metric, (None, None))
            self._conn.execute(
                "INSERT INTO scores (run_id, metric, mean, ci_low, ci_high)"
                " VALUES (?, ?, ?, ?, ?)",
                (run_id, metric, mean, low, high),
            )
        self._conn.commit()
        return run_id

    def get_run(self, run_id: int) -> RunRecord:
        """Fetch a run by id; raises KeyError if it does not exist."""
        row = self._conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"no run with id {run_id}")
        scores = {
            r["metric"]: {"mean": r["mean"], "ci_low": r["ci_low"], "ci_high": r["ci_high"]}
            for r in self._conn.execute(
                "SELECT metric, mean, ci_low, ci_high FROM scores WHERE run_id = ?",
                (run_id,),
            )
        }
        return RunRecord(
            id=row["id"],
            name=row["name"],
            model=row["model"],
            dataset=row["dataset"],
            created_at=row["created_at"],
            config=json.loads(row["config"]),
            metrics=scores,
        )

    def list_runs(self, limit: int = 50) -> List[RunRecord]:
        """Most recent runs first, up to ``limit``."""
        rows = self._conn.execute(
            "SELECT id FROM runs ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [self.get_run(r["id"]) for r in rows]

    def find_run(self, name: str) -> RunRecord:
        """Fetch the most recent run with the given name; KeyError if none."""
        row = self._conn.execute(
            "SELECT id FROM runs WHERE name = ? ORDER BY id DESC LIMIT 1", (name,)
        ).fetchone()
        if row is None:
            raise KeyError(f"no run named {name!r}")
        return self.get_run(row["id"])

    def compare(
        self,
        current_id: int,
        baseline_id: int,
        threshold: float = 0.02,
        relative: bool = False,
    ) -> List[Regression]:
        """Flag metrics where ``current`` regressed vs ``baseline``.

        A metric regresses when ``baseline_mean - current_mean`` exceeds
        ``threshold``.  With ``relative=True`` the drop is measured as a
        fraction of ``|baseline_mean|`` instead of an absolute difference.
        Metrics present in only one of the runs are ignored.
        """
        current = self.get_run(current_id)
        baseline = self.get_run(baseline_id)
        regressions: List[Regression] = []
        for metric, base in baseline.metrics.items():
            if metric not in current.metrics:
                continue
            base_mean = base["mean"]
            cur_mean = current.metrics[metric]["mean"]
            if relative:
                denom = abs(base_mean) if base_mean != 0 else 1.0
                drop = (base_mean - cur_mean) / denom
            else:
                drop = base_mean - cur_mean
            if drop > threshold:
                regressions.append(
                    Regression(
                        metric=metric,
                        baseline_mean=base_mean,
                        current_mean=cur_mean,
                        drop=base_mean - cur_mean,
                        threshold=threshold,
                        relative=relative,
                    )
                )
        return regressions
