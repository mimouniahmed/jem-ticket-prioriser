"""Collect Mozilla bug reports from the Bugzilla REST API into SQLite.

Two resumable stages:
1. metadata     : every bug created since --since in the selected products (1,000 per request);
2. descriptions : the first comment (= the bug description) for a sample of triaged (P1..P5)
                  and untriaged (--) bugs, 50 bugs per request.

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
    raw_json TEXT  -- raw API response, so no field is ever lost
);
CREATE TABLE IF NOT EXISTS descriptions (
    bug_id INTEGER PRIMARY KEY REFERENCES bugs(id),
    text TEXT,          -- NULL if the first comment is not accessible
    n_comments INTEGER  -- number of comments at collection time
);
"""


class Client:
    """Polite HTTP client: pause between requests, retries with backoff on transient errors."""

    def __init__(self, pause: float = 1.0, max_retries: int = 5):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "jem-prioriser (personal research project, one-off collection)"
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
                    r.raise_for_status()  # permanent error (400, 401...): retrying would not help
                error = f"HTTP {r.status_code}"
            wait = 5 * 2**attempt
            print(f"  ! {error}, retrying in {wait}s")
            time.sleep(wait)
        raise RuntimeError(f"Failed after {self.max_retries} attempts: {path}")


def collect_metadata(client: Client, db: sqlite3.Connection, since: str, page_size: int) -> None:
    # Cursor pagination: request bugs with id > last stored id. Unlike offsets, it is unaffected
    # by bugs created during the collection, and it makes the collection resumable.
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
            raise RuntimeError("Pagination is not advancing: is the f1/o1/v1 filter ignored by the API?")
        last_id = max(b["id"] for b in bugs)
        total = db.execute("SELECT COUNT(*) FROM bugs").fetchone()[0]
        print(f"  metadata: {total} bugs (last: #{last_id}, created {bugs[-1]['creation_time'][:10]})")


def choose_sample(
    db: sqlite3.Connection, n_triaged: int, n_untriaged: int, until: str, seed: int
) -> list[int]:
    """Reproducible random sample: same seed and end date -> same sample,
    even if new bugs were added to the table since."""
    rng = random.Random(seed)
    marks = ",".join("?" * len(TRIAGED))
    triaged = [r[0] for r in db.execute(
        f"SELECT id FROM bugs WHERE priority IN ({marks}) AND creation_time < ? ORDER BY id", [*TRIAGED, until]
    )]
    untriaged = [r[0] for r in db.execute(
        "SELECT id FROM bugs WHERE priority = '--' AND creation_time < ? ORDER BY id", [until]
    )]
    sample = rng.sample(triaged, min(n_triaged, len(triaged))) + rng.sample(untriaged, min(n_untriaged, len(untriaged)))
    # 92-96% of P4/P5 bugs come from bots, so the random sample holds only ~800 human P4/P5.
    # It is therefore completed with ALL human P4/P5 bugs (~4,300), leaving the draw unchanged.
    low_human = [r[0] for r in db.execute(
        f"SELECT id FROM bugs WHERE priority IN ('P4', 'P5') AND creation_time < ? AND {human_sql()} ORDER BY id",
        [until],
    )]
    chosen = set(sample)
    return sample + [i for i in low_human if i not in chosen]


def fetch_first_comments(client: Client, ids: list[int]) -> dict[int, tuple[str | None, int]]:
    # The API expects repeated ids (ids=1&ids=2...), not a comma-separated list.
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
    print(f"  descriptions: {len(sample) - len(todo)} already collected, {len(todo)} to collect")
    for start in range(0, len(todo), batch):
        ids = todo[start : start + batch]
        try:
            result = fetch_first_comments(client, ids)
        except requests.HTTPError:
            # One inaccessible bug fails the whole batch: fall back to one request per bug.
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
        print(f"  descriptions: {len(done) + start + len(ids)}/{len(sample)}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default="data/bugzilla.sqlite")
    p.add_argument("--since", default="2020-01-01")
    p.add_argument("--until", default="2026-10-01", help="end date (exclusive) of the sample")
    p.add_argument("--n-triaged", type=int, default=30000)
    p.add_argument("--n-untriaged", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--page-size", type=int, default=1000)
    p.add_argument("--batch", type=int, default=50)
    p.add_argument("--pause", type=float, default=1.0, help="seconds between two requests")
    p.add_argument("--skip-metadata", action="store_true", help="only collect descriptions")
    args = p.parse_args(argv)

    Path(args.db).parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(args.db)
    db.executescript(SCHEMA)
    client = Client(pause=args.pause)

    if not args.skip_metadata:
        print("1/2 Metadata")
        collect_metadata(client, db, args.since, args.page_size)
    print("2/2 Descriptions")
    collect_descriptions(client, db, choose_sample(db, args.n_triaged, args.n_untriaged, args.until, args.seed), args.batch)

    print("\nPriority distribution (all collected bugs):")
    for prio, n in db.execute("SELECT priority, COUNT(*) FROM bugs GROUP BY priority ORDER BY priority"):
        print(f"  {prio:4s} {n}")
    db.close()


if __name__ == "__main__":
    main()
