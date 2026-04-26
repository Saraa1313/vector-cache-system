"""
Writes the training dataset to disk.

Outputs per dataset_name:
  {output_dir}/{dataset_name}.parquet      primary format (requires pandas + pyarrow)
  {output_dir}/{dataset_name}.csv          human-readable fallback
  {output_dir}/{dataset_name}_meta.json    schema + generation config
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime

from offline_training.feature_extractor import (
    FEATURE_COLS, REGRESSION_TARGET, AUX_COLS, ID_COLS, CATEGORICAL_FEATURES,
)


def write_training_dataset(
    rows: list[dict],
    output_dir: str,
    dataset_name: str,
    metadata: dict,
) -> dict[str, str]:
    """
    Persist rows + metadata.  Returns {format: path} for written files.
    """
    os.makedirs(output_dir, exist_ok=True)

    parquet_path = os.path.join(output_dir, f"{dataset_name}.parquet")
    csv_path     = os.path.join(output_dir, f"{dataset_name}.csv")
    meta_path    = os.path.join(output_dir, f"{dataset_name}_meta.json")

    paths: dict[str, str] = {}

    # ── Tabular data ─────────────────────────────────────────────────────────
    try:
        import pandas as pd  # type: ignore
        df = pd.DataFrame(rows)
        df.to_parquet(parquet_path, index=False)
        df.to_csv(csv_path, index=False)
        paths["parquet"] = parquet_path
        paths["csv"]     = csv_path
        print(f"  Wrote {len(rows):,} rows → {parquet_path}")
    except ImportError:
        # pandas/pyarrow not installed — write CSV only
        if rows:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
        paths["csv"] = csv_path
        print(f"  Wrote {len(rows):,} rows → {csv_path}  (pandas unavailable, skipped parquet)")

    # ── Metadata ──────────────────────────────────────────────────────────────
    all_cols = list(rows[0].keys()) if rows else []
    full_meta = {
        "dataset_name":       dataset_name,
        "generated_at":       datetime.utcnow().isoformat() + "Z",
        "n_rows":             len(rows),
        "task":               "regression",
        "regression_target":  REGRESSION_TARGET,
        "feature_cols":       FEATURE_COLS,
        "categorical_features": CATEGORICAL_FEATURES,
        "aux_cols":           AUX_COLS,
        "id_cols":            ID_COLS,
        "all_cols":           all_cols,
        **metadata,
    }
    with open(meta_path, "w") as f:
        json.dump(full_meta, f, indent=2)
    paths["metadata"] = meta_path
    print(f"  Wrote metadata  → {meta_path}")

    return paths
