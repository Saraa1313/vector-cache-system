"""
Load a saved XGBoost model and evaluate it on a dataset.

Usage:
    python src/training/evaluate_model.py --model models/sift1m_recall_drop_v5_xgb.ubj \
                                          --data  data/training_data/sift1m_recall_drop_v5.csv
    python src/training/evaluate_model.py --auto  # picks latest model + latest CSV
"""

import os
import sys
import glob
import json
import argparse

import numpy as np
import pandas as pd
import xgboost as xgb

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SRC)

from training.train_model import (
    FEATURE_COLS, TARGET, _latest_csv, _rmse, _threshold_metrics, _stratify_key
)

DATA_DIR  = os.path.join(os.path.dirname(_SRC), "data", "training_data")
MODEL_DIR = os.path.join(os.path.dirname(_SRC), "models")


def _latest_model(model_dir: str) -> str:
    models = sorted(glob.glob(os.path.join(model_dir, "*_xgb.ubj")))
    if not models:
        raise FileNotFoundError(f"No saved models found in {model_dir}")
    return models[-1]


def evaluate(csv_path: str, model_path: str, unsafe_threshold: float = 0.05) -> None:
    print(f"Model : {model_path}")
    print(f"Data  : {csv_path}")

    model = xgb.XGBRegressor()
    model.load_model(model_path)

    df   = pd.read_csv(csv_path)
    X    = df[FEATURE_COLS]
    y    = df[TARGET].values
    pred = model.predict(X).clip(0.0, 1.0)

    nonzero = y > 0
    print(f"\n{'='*56}")
    print(f"Rows: {len(df):,}  |  nonzero recall_drop: {nonzero.sum():,} ({100*nonzero.mean():.1f}%)")
    print(f"RMSE (all):     {_rmse(y, pred):.5f}")
    if nonzero.any():
        print(f"RMSE (nonzero): {_rmse(y[nonzero], pred[nonzero]):.5f}")

    tm = _threshold_metrics(y, pred, unsafe_threshold)
    print(f"Threshold={unsafe_threshold}  P={tm['precision']:.3f}  R={tm['recall']:.3f}  "
          f"F1={tm['f1']:.3f}  ({tm['n_true_pos']} true pos / {tm['n_pred_pos']} pred pos)")

    # ── Per-stratum breakdown ─────────────────────────────────────────────────
    print(f"\n{'─'*56}")
    print(f"{'Stratum':<45}  {'RMSE':>7}  {'F1':>6}  {'n':>6}")
    print(f"{'─'*56}")
    for col in ["mutation_type", "spatial_pattern", "query_type"]:
        if col not in df.columns:
            continue
        for val in sorted(df[col].dropna().unique()):
            mask = df[col] == val
            y_s, p_s = y[mask], pred[mask]
            rmse_s = _rmse(y_s, p_s)
            tm_s   = _threshold_metrics(y_s, p_s, unsafe_threshold)
            print(f"  {col}={val:<40}  {rmse_s:>7.5f}  {tm_s['f1']:>6.3f}  {mask.sum():>6,}")

    # ── Feature importance ────────────────────────────────────────────────────
    print(f"\n{'─'*56}")
    print("Feature importance (gain):")
    importance = dict(zip(FEATURE_COLS, model.feature_importances_))
    for feat, imp in sorted(importance.items(), key=lambda x: -x[1]):
        bar = "█" * int(imp * 40)
        print(f"  {feat:<30s}  {imp:.4f}  {bar}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved recall_drop XGBoost model")
    parser.add_argument("--model",            default=None, help="Path to .ubj model file")
    parser.add_argument("--data",             default=None, help="Path to evaluation CSV")
    parser.add_argument("--auto",             action="store_true",
                        help="Auto-pick latest model and latest CSV")
    parser.add_argument("--unsafe-threshold", type=float, default=0.05)
    args = parser.parse_args()

    model_path = _latest_model(MODEL_DIR) if (args.auto or args.model is None) else args.model
    csv_path   = _latest_csv(DATA_DIR)    if (args.auto or args.data  is None) else args.data

    evaluate(csv_path, model_path, args.unsafe_threshold)


if __name__ == "__main__":
    main()
