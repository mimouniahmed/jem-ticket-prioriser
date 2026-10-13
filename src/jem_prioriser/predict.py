"""Rank a batch of bugs with the "champion" model from the MLflow Model Registry.

    uv run jem-predict                                   # untriaged Mozilla backlog (SQLite database)
    uv run jem-predict --input bugs.csv --output ranked.csv
    uv run jem-predict --model runs:/<run_id>/prioriser  # a specific run instead of the champion

The input CSV must contain: summary, description, product, component, type, op_sys.
"""

from __future__ import annotations

import argparse

import mlflow
import pandas as pd

from .dataset import DEFAULT_DB, load_untriaged
from .train import MODEL_NAME, TRACKING_URI

CHAMPION_URI = f"models:/{MODEL_NAME}@champion"


def load_bugs(path: str | None, db_path: str) -> pd.DataFrame:
    if path is None:
        return load_untriaged(db_path)
    df = pd.read_csv(path)
    for col in ["summary", "description", "product", "component", "type", "op_sys"]:
        df[col] = df[col].fillna("").astype(str) if col in df else ""
    return df


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=CHAMPION_URI)
    p.add_argument("--input", default=None, help="CSV of bugs (default: untriaged bugs from the database)")
    p.add_argument("--db", default=DEFAULT_DB)
    p.add_argument("--output", default=None)
    p.add_argument("--top", type=int, default=15)
    args = p.parse_args(argv)

    mlflow.set_tracking_uri(TRACKING_URI)
    ranked = mlflow.pyfunc.load_model(args.model).predict(load_bugs(args.input, args.db))
    if args.output:
        ranked.to_csv(args.output, index=False)
        print(f"{len(ranked)} bugs ranked -> {args.output}")

    cols = ["rank", "summary", "predicted", "urgency", "confidence", "needs_review"]
    view = ranked[cols].head(args.top).assign(summary=lambda d: d["summary"].str.slice(0, 70))
    print(view.to_string(index=False))
    print(f"\n{int(ranked['needs_review'].sum())} atypical bug(s) flagged for human review.")


if __name__ == "__main__":
    main()
