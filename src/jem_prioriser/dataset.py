"""Construction du dataset d'entraînement à partir de la base SQLite collectée.

Règles (analyse détaillée : notebooks/01_donnees_exploration.ipynb) :
- uniquement des bugs créés par des humains (cleaning.human_sql) ;
- uniquement les champs connus AU MOMENT DU DÉPÔT du bug : titre, description, produit,
  composant, type, système d'exploitation. Sont exclus car fixés pendant/après le triage
  (fuites) : severity, status, resolution, keywords, nombre de commentaires ;
- découpage chronologique : on entraîne sur le passé, on évalue sur le futur.
"""

from __future__ import annotations

import sqlite3

import pandas as pd

from .cleaning import BOT_DOMAINS, CLASSES, LABEL_OF_PRIORITY, human_sql

DEFAULT_DB = "data/bugzilla.sqlite"
FEATURE_COLUMNS = ["summary", "description", "product", "component", "type", "op_sys"]
MAX_DESCRIPTION_CHARS = 4000  # quelques descriptions contiennent des logs de plusieurs Mo

# Bornes du découpage chronologique (dates de création, bornes supérieures exclues)
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
    """Bugs humains triés (P1..P5) avec leur classe : P1, P2, P3 ou Basse (P4+P5)."""
    df = _read(db_path, f"b.priority IN ('P1','P2','P3','P4','P5') AND {human_sql('b.creator')}")
    df["label"] = pd.Categorical(df["priority"].map(LABEL_OF_PRIORITY), categories=CLASSES, ordered=True)
    df["y"] = df["label"].cat.codes.astype("int64")  # 0 = P1 (le plus urgent) ... 3 = Basse
    return df


def load_untriaged(db_path: str = DEFAULT_DB) -> pd.DataFrame:
    """Bugs humains jamais triés (priorité '--') : le backlog que l'outil final doit ordonner."""
    return _read(db_path, f"b.priority = '--' AND {human_sql('b.creator')}")


def load_bot_bugs(db_path: str = DEFAULT_DB) -> pd.DataFrame:
    """Bugs créés par des bots : textes très différents des rapports humains.
    Servent de jeu « hors distribution » naturel pour évaluer la détection par l'énergie."""
    where = " OR ".join(f"b.creator LIKE '%{d}'" for d in BOT_DOMAINS)
    return _read(db_path, f"({where})")


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Entraînement : 2020-2024 ; validation : 2025 ; test : 2026."""
    train_end = pd.Timestamp(TRAIN_END, tz="UTC")
    val_end = pd.Timestamp(VAL_END, tz="UTC")
    t = df["creation_time"]
    return (
        df[t < train_end].reset_index(drop=True),
        df[(t >= train_end) & (t < val_end)].reset_index(drop=True),
        df[t >= val_end].reset_index(drop=True),
    )
