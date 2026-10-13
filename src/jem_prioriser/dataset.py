"""Build the modelling dataset from the collected SQLite database.

Rules (detailed analysis: notebooks/01_data_exploration.ipynb):
- only bugs created by humans (cleaning.human_sql);
- only fields known WHEN THE BUG IS FILED: summary, description, product, component, type,
  operating system. Excluded because they are set during or after triage (leakage):
  severity, status, resolution, keywords, number of comments;
- chronological split: train on the past, evaluate on the future.
"""

from __future__ import annotations

import sqlite3

import pandas as pd

from .cleaning import BOT_DOMAINS, CLASSES, LABEL_OF_PRIORITY, human_sql

DEFAULT_DB = "data/bugzilla.sqlite"
FEATURE_COLUMNS = ["summary", "description", "product", "component", "type", "op_sys"]
MAX_DESCRIPTION_CHARS = 4000  # some descriptions embed logs of several MB

# Chronological split boundaries (creation dates, upper bounds excluded)
TRAIN_END = "2025-01-01"
VAL_END = "2026-01-01"

_QUERY = """
SELECT b.id, b.summary, d.text AS description, b.product, b.component, b.type, b.op_sys,
       b.priority, b.creator, b.creation_time
FROM descriptions d JOIN bugs b ON b.id = d.bug_id
WHERE {where}
ORDER BY b.id
"""


def _read(db_path: str, where: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as db:
        df = pd.read_sql_query(_QUERY.format(where=where), db)
    df["description"] = df["description"].fillna("").str.slice(0, MAX_DESCRIPTION_CHARS)
    for col in ["summary", "product", "component", "type", "op_sys"]:
        df[col] = df[col].fillna("").astype(str)
    df["creation_time"] = pd.to_datetime(df["creation_time"], utc=True)
    return df


def load_labeled(db_path: str = DEFAULT_DB) -> pd.DataFrame:
    """Triaged human bugs (P1..P5) with their class: P1, P2, P3 or Low (P4+P5)."""
    df = _read(db_path, f"b.priority IN ('P1','P2','P3','P4','P5') AND {human_sql('b.creator')}")
    df["label"] = pd.Categorical(df["priority"].map(LABEL_OF_PRIORITY), categories=CLASSES, ordered=True)
    df["y"] = df["label"].cat.codes.astype("int64")  # 0 = P1 (most urgent) ... 3 = Low
    return df


def load_untriaged(db_path: str = DEFAULT_DB) -> pd.DataFrame:
    """Human bugs never triaged (priority '--'): the backlog the triage tool has to rank."""
    return _read(db_path, f"b.priority = '--' AND {human_sql('b.creator')}")


def load_bot_bugs(db_path: str = DEFAULT_DB) -> pd.DataFrame:
    """Bugs created by bots, whose texts differ strongly from human reports.
    Used as a natural out-of-distribution set to evaluate energy-based detection."""
    where = " OR ".join(f"b.creator LIKE '%{d}'" for d in BOT_DOMAINS)
    return _read(db_path, f"({where})")


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train: 2020-2024; validation: 2025; test: 2026."""
    train_end = pd.Timestamp(TRAIN_END, tz="UTC")
    val_end = pd.Timestamp(VAL_END, tz="UTC")
    t = df["creation_time"]
    return (
        df[t < train_end].reset_index(drop=True),
        df[(t >= train_end) & (t < val_end)].reset_index(drop=True),
        df[t >= val_end].reset_index(drop=True),
    )
