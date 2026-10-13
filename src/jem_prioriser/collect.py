"""Collecte de bugs Mozilla via l'API REST de Bugzilla, stockés dans SQLite.

La collecte se fait en deux temps, chacun reprenant là où il s'était arrêté :
1. metadata     : tous les bugs créés depuis --since dans les produits choisis (1000 par requête) ;
2. descriptions : le premier commentaire (= la description du bug) pour un échantillon
                  de bugs triés (P1..P5) et non triés (--), 50 bugs par requête.

    uv run python -m jem_prioriser.collect
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import time
from pathlib import Path

import requests

from .cleaning import human_sql

API = "https://bugzilla.mozilla.org/rest"
PRODUCTS = ["Firefox", "Core", "DevTools", "Toolkit"]
FIELDS = [
    "id", "summary", "priority", "severity", "type", "product", "component", "status",
    "resolution", "creation_time", "creator", "keywords", "platform", "op_sys", "version",
]
TRIAGED = ["P1", "P2", "P3", "P4", "P5"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS bugs (
    id INTEGER PRIMARY KEY,
    summary TEXT, priority TEXT, severity TEXT, type TEXT, product TEXT, component TEXT,
    status TEXT, resolution TEXT, creation_time TEXT, creator TEXT, keywords TEXT,
    platform TEXT, op_sys TEXT, version TEXT,
    raw_json TEXT  -- réponse brute de l'API : on ne perd rien, même un champ oublié
);
CREATE TABLE IF NOT EXISTS descriptions (
    bug_id INTEGER PRIMARY KEY REFERENCES bugs(id),
    text TEXT,          -- NULL si le premier commentaire est inaccessible
    n_comments INTEGER  -- nombre de commentaires au moment de la collecte
);
"""


class Client:
    """Client HTTP poli : pause entre requêtes, nouvelles tentatives en cas d'erreur serveur."""

    def __init__(self, pause: float = 1.0, max_retries: int = 5):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "jem-prioriser (projet educatif, collecte ponctuelle)"
        self.pause = pause
        self.max_retries = max_retries

    def get(self, path: str, params: dict) -> dict:
        for attempt in range(self.max_retries):
            time.sleep(self.pause)
            try:
                r = self.session.get(f"{API}/{path}", params=params, timeout=120)
            except requests.RequestException as e:
                error = str(e)
            else:
                if r.status_code == 200:
                    return r.json()
                if r.status_code not in (429, 500, 502, 503, 504):
                    r.raise_for_status()  # erreur « définitive » (400, 401...) : inutile de réessayer
                error = f"HTTP {r.status_code}"
            wait = 5 * 2**attempt
            print(f"  ! {error}, nouvelle tentative dans {wait}s")
            time.sleep(wait)
        raise RuntimeError(f"Échec après {self.max_retries} tentatives : {path}")


def collect_metadata(client: Client, db: sqlite3.Connection, since: str, page_size: int) -> None:
    # Pagination « par curseur » : on demande les bugs d'id > dernier id stocké.
    # Plus robuste qu'un offset si des bugs sont créés pendant la collecte, et permet la reprise.
    last_id = db.execute("SELECT COALESCE(MAX(id), 0) FROM bugs").fetchone()[0]
    while True:
        params = {
            "product": PRODUCTS, "creation_time": since, "order": "bug_id", "limit": page_size,
            "include_fields": ",".join(FIELDS),
            "f1": "bug_id", "o1": "greaterthan", "v1": last_id,
        }
        bugs = client.get("bug", params)["bugs"]
        if not bugs:
            break
        db.executemany(
            f"INSERT OR REPLACE INTO bugs ({', '.join(FIELDS)}, raw_json) "
            f"VALUES ({', '.join('?' * (len(FIELDS) + 1))})",
            [
                [",".join(b[f]) if f == "keywords" else b.get(f) for f in FIELDS] + [json.dumps(b)]
                for b in bugs
            ],
        )
        db.commit()
        if max(b["id"] for b in bugs) <= last_id:
            raise RuntimeError("La pagination n'avance plus : le filtre f1/o1/v1 est ignoré par l'API ?")
        last_id = max(b["id"] for b in bugs)
        total = db.execute("SELECT COUNT(*) FROM bugs").fetchone()[0]
        print(f"  metadata : {total} bugs (dernier : #{last_id}, créé le {bugs[-1]['creation_time'][:10]})")


def choose_sample(
    db: sqlite3.Connection, n_triaged: int, n_untriaged: int, until: str, seed: int
) -> list[int]:
    """Tirage aléatoire reproductible : même graine et même date de fin -> même échantillon,
    même si la table bugs a reçu de nouveaux bugs depuis."""
    rng = random.Random(seed)
    marks = ",".join("?" * len(TRIAGED))
    triaged = [r[0] for r in db.execute(
        f"SELECT id FROM bugs WHERE priority IN ({marks}) AND creation_time < ? ORDER BY id", [*TRIAGED, until]
    )]
    untriaged = [r[0] for r in db.execute(
        "SELECT id FROM bugs WHERE priority = '--' AND creation_time < ? ORDER BY id", [until]
    )]
    sample = rng.sample(triaged, min(n_triaged, len(triaged))) + rng.sample(untriaged, min(n_untriaged, len(untriaged)))
    # 92-96 % des P4/P5 viennent de bots : le tirage aléatoire ne contient que ~800 P4/P5
    # humains. On complète donc avec TOUS les P4/P5 humains (~4 300), sans modifier le tirage.
    low_human = [r[0] for r in db.execute(
        f"SELECT id FROM bugs WHERE priority IN ('P4', 'P5') AND creation_time < ? AND {human_sql()} ORDER BY id",
        [until],
    )]
    chosen = set(sample)
    return sample + [i for i in low_human if i not in chosen]


def fetch_first_comments(client: Client, ids: list[int]) -> dict[int, tuple[str | None, int]]:
    # L'API attend l'id répété (ids=1&ids=2...), pas une liste séparée par des virgules.
    data = client.get(f"bug/{ids[0]}/comment", {"ids": ids[1:], "include_fields": "text,count"})
    out = {}
    for bug_id, content in data["bugs"].items():
        comments = content["comments"]
        first = next((c["text"] for c in comments if c["count"] == 0), None)
        out[int(bug_id)] = (first, len(comments))
    return out


def collect_descriptions(client: Client, db: sqlite3.Connection, sample: list[int], batch: int) -> None:
    done = {r[0] for r in db.execute("SELECT bug_id FROM descriptions")}
    todo = [i for i in sample if i not in done]
    print(f"  descriptions : {len(sample) - len(todo)} déjà collectées, {len(todo)} à collecter")
    for start in range(0, len(todo), batch):
        ids = todo[start : start + batch]
        try:
            result = fetch_first_comments(client, ids)
        except requests.HTTPError:
            # Un bug inaccessible fait échouer tout le lot : on repasse un par un.
            result = {}
            for i in ids:
                try:
                    result.update(fetch_first_comments(client, [i]))
                except requests.HTTPError:
                    result[i] = (None, 0)
        db.executemany(
            "INSERT OR REPLACE INTO descriptions (bug_id, text, n_comments) VALUES (?, ?, ?)",
            [(i, *result.get(i, (None, 0))) for i in ids],
        )
        db.commit()
        print(f"  descriptions : {len(done) + start + len(ids)}/{len(sample)}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default="data/bugzilla.sqlite")
    p.add_argument("--since", default="2020-01-01")
    p.add_argument("--until", default="2026-10-01", help="date de fin (exclue) pour l'échantillon")
    p.add_argument("--n-triaged", type=int, default=30000)
    p.add_argument("--n-untriaged", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--page-size", type=int, default=1000)
    p.add_argument("--batch", type=int, default=50)
    p.add_argument("--pause", type=float, default=1.0, help="secondes entre deux requêtes")
    p.add_argument("--skip-metadata", action="store_true", help="ne collecter que les descriptions")
    args = p.parse_args(argv)

    Path(args.db).parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(args.db)
    db.executescript(SCHEMA)
    client = Client(pause=args.pause)

    if not args.skip_metadata:
        print("1/2 Métadonnées")
        collect_metadata(client, db, args.since, args.page_size)
    print("2/2 Descriptions")
    collect_descriptions(client, db, choose_sample(db, args.n_triaged, args.n_untriaged, args.until, args.seed), args.batch)

    print("\nRépartition des priorités (tous les bugs collectés) :")
    for prio, n in db.execute("SELECT priority, COUNT(*) FROM bugs GROUP BY priority ORDER BY priority"):
        print(f"  {prio:4s} {n}")
    db.close()


if __name__ == "__main__":
    main()
