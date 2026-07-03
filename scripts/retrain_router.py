#!/usr/bin/env python3
"""
Offline router retraining script.

Loads cheap-routed request history from the DB and fits a LogisticRegression
classifier to predict escalation probability. Saves coefficients to
data_dir/router_model.pkl.

SELECTION BIAS WARNING: This model only learns the difficulty boundary within
the cheap-routed region. Requests routed directly to strong (above threshold,
tool override, or client override) have no escalation label and are excluded.
The model cannot predict outcomes for requests in the strong-direct region.
Future work: use strong model responses as pseudo-labels for unlabeled
high-difficulty requests.

Usage:
    python scripts/retrain_router.py [--db PATH] [--out PATH]
"""
from __future__ import annotations
import argparse
import json
import pickle
import sqlite3
import sys
from pathlib import Path


_FEATURE_KEYS = ["length", "tools", "code", "math", "multi_step", "depth"]


def load_from_db(db_path: Path) -> tuple[list[list[float]], list[int]]:
    """
    Load feature vectors and escalation labels for cheap-routed requests only.

    Returns (features, labels) where:
    - features: list of [length, tools, code, math, multi_step, depth] vectors
    - labels: list of int (1=escalated, 0=not escalated)

    NOTE: Only cheap-routed rows are included. Strong-direct rows (tier='strong'
    due to above_threshold, tools_forced_strong, or client_override) have no
    escalation label and are excluded. This is the selection bias documented in
    the module docstring.
    """
    con = sqlite3.connect(db_path)
    rows = con.execute(
        "SELECT layers_applied FROM requests WHERE status='ok'"
    ).fetchall()
    con.close()

    features: list[list[float]] = []
    labels: list[int] = []

    for (layers_json,) in rows:
        try:
            layers = json.loads(layers_json)
        except Exception:
            continue

        router_decision = next(
            (l for l in layers if l.get("layer") == "router" and l.get("action") == "applied"),
            None,
        )
        if router_decision is None:
            continue

        detail = router_decision.get("detail", {})
        # Only include cheap-routed requests (selection bias: strong-direct has no label)
        if detail.get("tier") != "cheap":
            continue

        feat_dict = detail.get("features", {})
        vec = [float(feat_dict.get(k, 0.0)) for k in _FEATURE_KEYS]
        escalated = int(bool(detail.get("escalated", False)))

        features.append(vec)
        labels.append(escalated)

    return features, labels


def retrain(db_path: Path, out_path: Path) -> None:
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        print("ERROR: scikit-learn not installed. Run: pip install scikit-learn", file=sys.stderr)
        sys.exit(1)

    features, labels = load_from_db(db_path)

    if len(features) < 10:
        print(f"ERROR: Only {len(features)} cheap-routed samples found. Need ≥10 to retrain.", file=sys.stderr)
        sys.exit(1)

    print(f"Training on {len(features)} samples ({sum(labels)} escalated, {len(labels)-sum(labels)} not).")
    print("Selection bias: model only covers cheap-routed region. See module docstring.")

    model = LogisticRegression(max_iter=1000)
    model.fit(features, labels)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({"model": model, "feature_keys": _FEATURE_KEYS}, f)

    print(f"Model saved to {out_path}")
    print(f"Coefficients: {dict(zip(_FEATURE_KEYS, model.coef_[0].tolist()))}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain cascade router difficulty model.")
    parser.add_argument("--db", type=Path, default=None, help="Path to tokengate.db")
    parser.add_argument("--out", type=Path, default=None, help="Output path for router_model.pkl")
    args = parser.parse_args()

    data_dir = Path("~/.rait").expanduser()
    db_path = args.db or (data_dir / "tokengate.db")
    out_path = args.out or (data_dir / "router_model.pkl")

    if not db_path.exists():
        print(f"ERROR: DB not found at {db_path}", file=sys.stderr)
        sys.exit(1)

    retrain(db_path, out_path)


if __name__ == "__main__":
    main()
