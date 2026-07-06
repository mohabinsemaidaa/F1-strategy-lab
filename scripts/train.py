"""Train the lap-time model with leave-races-out cross-validation.

Usage:
    python scripts/train.py
"""

import json
from pathlib import Path
import pandas as pd
from f1strategy.features import build_dataset, dataset_metadata
from f1strategy.model import MODEL_PATH, cross_validate, save_model, train_final

DATA = Path("data/raw_laps.parquet")
METRICS = Path("models/metrics.json")
METADATA = Path("models/metadata.json")

def main() -> None:
    raw = pd.read_parquet(DATA)
    X, y = build_dataset(raw)
    print(f"Dataset: {len(X):,} clean laps, {X['Event'].nunique()} races")

    print("Cross-validating (leave-races-out)...")
    cv = cross_validate(X, y)
    for i, mae in enumerate(cv["fold_mae"], 1):
        print(f"  fold {i}: MAE {mae:.3f}s")
    print(f"  mean MAE on unseen races: {cv['mean_mae']:.3f}s")

    print("Training final model on all data...")
    pipe = train_final(X, y)
    save_model(pipe)

    METRICS.parent.mkdir(parents=True, exist_ok=True)
    METRICS.write_text(json.dumps({"cv_mae_seconds": cv}, indent=2))
    METADATA.write_text(json.dumps(dataset_metadata(raw), indent=2))
    print(f"Saved model -> {MODEL_PATH}")
    print(f"Saved metrics -> {METRICS}, metadata -> {METADATA}")

if __name__ == "__main__":
    main()