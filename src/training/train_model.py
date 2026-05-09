"""
Train an XGBoost model to predict recall_drop for stale partition detection.

Modes:
  --regressor (default): predicts continuous recall_drop; fetch if pred >= unsafe_threshold
  --classifier:          predicts P(recall_drop > 0); fetch if prob >= prob_threshold
                         Uses scale_pos_weight for class imbalance instead of sample weights.
                         Saves as {name}_xgb_clf.ubj.

Split strategy: stratified random 80/20 by (mutation_type × spatial_pattern × query_type).
Sample weighting: nonzero recall_drop rows upweighted to counter zero inflation (regressor only).
OOD probe: separate metrics reported on marginal_contributor and rank_targeted rows.

Usage:
    python src/training/train_model.py [--help]
    python src/training/train_model.py --data data/training_data/sift1m_recall_drop_v5.csv
    python src/training/train_model.py --auto             # picks latest version, regressor
    python src/training/train_model.py --auto --classifier
"""

import os
import sys
import json
import glob
import argparse
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SRC)

DATA_DIR = os.path.join(os.path.dirname(_SRC), "data", "training_data")
MODEL_DIR = os.path.join(os.path.dirname(_SRC), "models")

FEATURE_COLS = [
    "version_lag",
    "partition_size",
    "update_fraction",
    "delete_fraction",
    "size_reduction_fraction",
    "recon_error_stale",
    "recon_error_rel_delta",
    "normalized_probe_rank",
    "centroid_dist",
    "rel_gap_to_prev",
    "rel_gap_to_next",
    "candidate_fraction",
    "historical_hit_rate",
]
TARGET = "recall_drop"
STRAT_COLS = ["mutation_type", "spatial_pattern", "query_type"]


def _latest_csv(data_dir: str) -> str:
    csvs = glob.glob(os.path.join(data_dir, "sift1m_recall_drop_v*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No training CSVs found in {data_dir}")
    def _version(p):
        base = os.path.splitext(os.path.basename(p))[0]  # sift1m_recall_drop_v7
        try:
            return int(base.split("_v")[-1])
        except ValueError:
            return -1
    return max(csvs, key=_version)


def _stratify_key(df: pd.DataFrame) -> pd.Series:
    return df[STRAT_COLS].astype(str).agg("__".join, axis=1)


def _compute_sample_weights(y: pd.Series, nonzero_weight: float) -> np.ndarray:
    weights = np.ones(len(y), dtype=np.float32)
    weights[y > 0] = nonzero_weight
    return weights


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _threshold_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> dict:
    pred_pos = y_pred >= threshold
    true_pos = y_true >= threshold
    tp = (pred_pos & true_pos).sum()
    fp = (pred_pos & ~true_pos).sum()
    fn = (~pred_pos & true_pos).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
            "n_true_pos": int(true_pos.sum()), "n_pred_pos": int(pred_pos.sum())}


def train(
    csv_path: str,
    nonzero_weight: float = 5.0,
    test_size: float = 0.20,
    random_state: int = 42,
    unsafe_threshold: float = 0.05,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    early_stopping_rounds: int = 30,
    save_model: bool = True,
    dataset_name: str | None = None,
    use_classifier: bool = False,
    resample_ratio: float | None = None,
    min_fresh_recall: float = 0.80,
    delete_weight: float = 1.0,
) -> dict:
    t0 = time.time()
    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path)
    if min_fresh_recall > 0.0 and "fresh_recall" in df.columns:
        before = len(df)
        df = df[df["fresh_recall"] >= min_fresh_recall].reset_index(drop=True)
        print(f"  Dropped {before - len(df):,} rows with fresh_recall < {min_fresh_recall}")
    print(f"  {len(df):,} rows  |  nonzero recall_drop: "
          f"{(df[TARGET] > 0).sum():,} ({100*(df[TARGET]>0).mean():.1f}%)")

    # Derive size_reduction_fraction from counts if not already in the CSV (v15 and earlier).
    # For templates with no inserts: equals delete_fraction exactly.
    if "size_reduction_fraction" not in df.columns:
        if "delete_count" in df.columns and "partition_size" in df.columns:
            ins = df["insert_count"] if "insert_count" in df.columns else 0
            net_shrink = (df["delete_count"] - ins).clip(lower=0)
            df["size_reduction_fraction"] = (net_shrink / df["partition_size"].clip(lower=1)).round(6)
        else:
            df["size_reduction_fraction"] = 0.0

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    # ── Stratified train/test split ───────────────────────────────────────────
    strat = _stratify_key(df)
    # Drop strat groups with < 2 members (can't split)
    valid_mask = strat.map(strat.value_counts() >= 2)
    df = df[valid_mask].reset_index(drop=True)
    strat = strat[valid_mask].reset_index(drop=True)

    train_idx, test_idx = train_test_split(
        df.index, test_size=test_size, stratify=strat, random_state=random_state
    )
    df_train = df.loc[train_idx].reset_index(drop=True)
    df_test  = df.loc[test_idx].reset_index(drop=True)

    print(f"  Train: {len(df_train):,}  |  Test: {len(df_test):,}")

    X_train = df_train[FEATURE_COLS]
    y_train = df_train[TARGET]
    X_test  = df_test[FEATURE_COLS]
    y_test  = df_test[TARGET]

    # Binary labels for classifier mode — positive if recall_drop >= unsafe_threshold
    y_train_bin = (y_train >= unsafe_threshold).astype(int)
    y_test_bin  = (y_test  >= unsafe_threshold).astype(int)

    # ── Early-stopping validation set (15% of train, stratified) ────────────
    strat_train = _stratify_key(df_train)
    tr_idx, val_idx = train_test_split(
        np.arange(len(df_train)), test_size=0.15,
        stratify=strat_train,
        random_state=random_state,
    )
    X_tr  = X_train.iloc[tr_idx]
    X_val = X_train.iloc[val_idx]

    # ── Delete upweighting vector (classifier only, built from full df_train) ──
    # Computed before resampling so indices stay aligned with df_train positions.
    w_delete = np.ones(len(df_train), dtype=np.float32)
    if delete_weight != 1.0 and "mutation_type" in df_train.columns:
        delete_pos = (
            (df_train["mutation_type"] == "delete") &
            (df_train[TARGET] >= unsafe_threshold)
        ).values
        w_delete[delete_pos] = delete_weight
        print(f"  Delete upweighting: {delete_pos.sum():,} delete positive rows → weight={delete_weight}")

    if use_classifier:
        # ── Classifier ───────────────────────────────────────────────────────
        # Optional: undersample negatives to resample_ratio negatives per positive
        if resample_ratio is not None:
            rng = np.random.default_rng(random_state)
            pos_idx = np.where(y_train_bin.values == 1)[0]
            neg_idx = np.where(y_train_bin.values == 0)[0]
            n_neg_keep = min(len(neg_idx), int(resample_ratio * len(pos_idx)))
            neg_keep = rng.choice(neg_idx, size=n_neg_keep, replace=False)
            keep = np.sort(np.concatenate([pos_idx, neg_keep]))
            X_train = X_train.iloc[keep].reset_index(drop=True)
            y_train_bin = y_train_bin.iloc[keep].reset_index(drop=True)
            # Rebuild early-stopping split from resampled train set
            strat_train2 = _stratify_key(df_train.iloc[keep].reset_index(drop=True))
            tr_idx, val_idx = train_test_split(
                np.arange(len(X_train)), test_size=0.15,
                stratify=strat_train2, random_state=random_state,
            )
            # Recompute X_tr/X_val from resampled X_train
            X_tr  = X_train.iloc[tr_idx]
            X_val = X_train.iloc[val_idx]
            print(f"  Resampled to ratio 1:{resample_ratio:.0f}  "
                  f"({len(pos_idx)} pos, {n_neg_keep} neg → {len(keep)} total)")

        y_tr  = y_train_bin.iloc[tr_idx]
        y_val = y_train_bin.iloc[val_idx]
        # Slice delete weights to match X_tr (respects resampling if applied)
        w_tr = w_delete[keep][tr_idx] if resample_ratio is not None else w_delete[tr_idx]
        n_neg = (y_train_bin == 0).sum()
        n_pos = (y_train_bin == 1).sum()
        spw   = 1.0 if resample_ratio is not None else n_neg / max(n_pos, 1)
        model = xgb.XGBClassifier(
            n_estimators          = n_estimators,
            learning_rate         = learning_rate,
            max_depth             = max_depth,
            subsample             = subsample,
            colsample_bytree      = colsample_bytree,
            tree_method           = "hist",
            random_state          = random_state,
            n_jobs                = -1,
            early_stopping_rounds = early_stopping_rounds,
            eval_metric           = "logloss",
            scale_pos_weight      = spw,
        )
        print(f"Training XGBoost classifier (scale_pos_weight={spw:.1f}, delete_weight={delete_weight}) ...")
        model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_val, y_val)], verbose=50)
        print(f"  Best iteration: {model.best_iteration}")

        prob_test  = model.predict_proba(X_test)[:, 1]
        prob_train = model.predict_proba(X_train)[:, 1]

        # Threshold sweep (0.05 steps for finer F1-optimal resolution)
        sweep = {}
        for t in [round(v * 0.05, 2) for v in range(2, 20)]:
            m = _threshold_metrics(y_test_bin.values, prob_test, t)
            sweep[str(t)] = m
        best_t = max(sweep, key=lambda t: sweep[t]["f1"])

        metrics = {
            "mode":              "classifier",
            "scale_pos_weight":  round(spw, 2),
            "delete_weight":     delete_weight,
            "threshold_sweep":   sweep,
            "best_threshold":    float(best_t),
            "threshold_metrics": sweep[best_t],
            "n_train":           len(df_train),
            "n_test":            len(df_test),
            "nonzero_frac_train": float((y_train > 0).mean()),
            "nonzero_frac_test":  float((y_test  > 0).mean()),
            "best_iteration":    int(model.best_iteration),
        }
        pred_test  = prob_test   # alias for OOD blocks below
        pred_train = prob_train
        _unsafe_threshold = float(best_t)

    else:
        # ── Regressor ────────────────────────────────────────────────────────
        w_train = _compute_sample_weights(y_train, nonzero_weight)
        y_tr  = y_train.iloc[tr_idx]
        y_val = y_train.iloc[val_idx]
        w_tr  = w_train[tr_idx]

        model = xgb.XGBRegressor(
            n_estimators          = n_estimators,
            learning_rate         = learning_rate,
            max_depth             = max_depth,
            subsample             = subsample,
            colsample_bytree      = colsample_bytree,
            tree_method           = "hist",
            random_state          = random_state,
            n_jobs                = -1,
            early_stopping_rounds = early_stopping_rounds,
            eval_metric           = "rmse",
        )
        print("Training XGBoost regressor ...")
        model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_val, y_val)], verbose=50)
        print(f"  Best iteration: {model.best_iteration}")

        pred_test  = model.predict(X_test).clip(0.0, 1.0)
        pred_train = model.predict(X_train).clip(0.0, 1.0)
        nonzero_mask = y_test.values > 0
        _unsafe_threshold = unsafe_threshold

        metrics = {
            "mode":              "regressor",
            "rmse_all":          _rmse(y_test, pred_test),
            "rmse_nonzero":      _rmse(y_test[nonzero_mask], pred_test[nonzero_mask]) if nonzero_mask.any() else None,
            "rmse_train":        _rmse(y_train, pred_train),
            "threshold_metrics": _threshold_metrics(y_test.values, pred_test, unsafe_threshold),
            "n_train":           len(df_train),
            "n_test":            len(df_test),
            "nonzero_frac_train": float((y_train > 0).mean()),
            "nonzero_frac_test":  float((y_test  > 0).mean()),
            "best_iteration":    int(model.best_iteration),
            "nonzero_weight":    nonzero_weight,
        }

    # ── OOD probe: marginal_contributor rows in test set ─────────────────────
    ood_mask = df_test["spatial_pattern"] == "marginal_contributor"
    if ood_mask.any():
        y_ood    = (y_test[ood_mask].values > 0).astype(int) if use_classifier else y_test[ood_mask].values
        p_ood    = pred_test[ood_mask]
        ood_entry = {
            "n":          int(ood_mask.sum()),
            "threshold_metrics": _threshold_metrics(y_ood, p_ood, _unsafe_threshold),
        }
        if not use_classifier:
            ood_entry["rmse_all"]    = _rmse(y_test[ood_mask].values, p_ood)
            raw_ood = y_test[ood_mask].values
            ood_entry["rmse_nonzero"] = _rmse(raw_ood[raw_ood > 0], p_ood[raw_ood > 0]) if (raw_ood > 0).any() else None
        metrics["ood_marginal_contributor"] = ood_entry

    # ── OOD probe: rank_targeted rows in test set ─────────────────────────────
    rt_mask = df_test["spatial_pattern"] == "rank_targeted"
    if rt_mask.any():
        rt_metrics = {}
        for window in ("close", "mid", "late"):
            w_mask = rt_mask & df_test["workload_name"].str.contains(f"_{window}_rank_")
            if not w_mask.any():
                continue
            raw_w = y_test[w_mask].values
            y_w   = (raw_w > 0).astype(int) if use_classifier else raw_w
            p_w   = pred_test[w_mask]
            entry = {
                "n":            int(w_mask.sum()),
                "nonzero_frac": round(float((raw_w > 0).mean()), 4),
                "threshold_metrics": _threshold_metrics(y_w, p_w, _unsafe_threshold),
            }
            if not use_classifier:
                entry["rmse_all"] = _rmse(raw_w, p_w)
            rt_metrics[f"{window}_rank"] = entry
        if rt_metrics:
            metrics["ood_rank_targeted"] = rt_metrics

    # ── Feature importance ────────────────────────────────────────────────────
    importance = dict(zip(FEATURE_COLS, model.feature_importances_.tolist()))
    metrics["feature_importance"] = dict(sorted(importance.items(), key=lambda x: -x[1]))

    # ── Print summary ─────────────────────────────────────────────────────────
    elapsed = round(time.time() - t0, 1)
    print(f"\n{'='*56}")
    if not use_classifier:
        print(f"RMSE (all test):      {metrics['rmse_all']:.5f}")
        print(f"RMSE (nonzero test):  {metrics['rmse_nonzero']:.5f}")
        print(f"RMSE (train):         {metrics['rmse_train']:.5f}")
    else:
        print(f"Mode: classifier  scale_pos_weight={metrics['scale_pos_weight']:.1f}")
        print(f"Best threshold (F1-optimal): {metrics['best_threshold']}")
    tm = metrics["threshold_metrics"]
    print(f"Threshold={_unsafe_threshold}  P={tm['precision']:.3f}  R={tm['recall']:.3f}  F1={tm['f1']:.3f}"
          f"  ({tm['n_true_pos']} true positives)")
    if "ood_marginal_contributor" in metrics:
        ood = metrics["ood_marginal_contributor"]
        ood_line = f"OOD (marginal_contributor n={ood['n']}):  F1={ood['threshold_metrics']['f1']:.3f}"
        if "rmse_all" in ood:
            ood_line += f"  RMSE={ood['rmse_all']:.5f}"
        print(ood_line)
    if "ood_rank_targeted" in metrics:
        for window, wd in metrics["ood_rank_targeted"].items():
            tm_w = wd["threshold_metrics"]
            rt_line = (f"OOD (rank_targeted/{window} n={wd['n']}):  "
                       f"nonzero_frac={wd['nonzero_frac']:.3f}  F1={tm_w['f1']:.3f}")
            if "rmse_all" in wd:
                rt_line += f"  RMSE={wd['rmse_all']:.5f}"
            print(rt_line)
    print(f"\nTop features:")
    for feat, imp in list(metrics["feature_importance"].items())[:6]:
        print(f"  {feat:30s}  {imp:.4f}")
    print(f"\nElapsed: {elapsed}s")

    # ── Save ──────────────────────────────────────────────────────────────────
    if save_model:
        os.makedirs(MODEL_DIR, exist_ok=True)
        name     = dataset_name or os.path.splitext(os.path.basename(csv_path))[0]
        suffix   = "_xgb_clf" if use_classifier else "_xgb"
        model_path   = os.path.join(MODEL_DIR, f"{name}{suffix}.ubj")
        metrics_path = os.path.join(MODEL_DIR, f"{name}{suffix}_metrics.json")
        model.save_model(model_path)
        metrics["elapsed_seconds"] = elapsed
        metrics["csv_path"]        = csv_path
        metrics["model_path"]      = model_path
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nModel:   {model_path}")
        print(f"Metrics: {metrics_path}")

    return metrics, model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train recall_drop XGBoost regressor")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--data",   default=None, help="Path to training CSV")
    group.add_argument("--auto",   action="store_true", help="Auto-pick latest version CSV")
    parser.add_argument("--nonzero-weight",       type=float, default=5.0,
                        help="Sample weight for rows with recall_drop > 0 (default: 5.0)")
    parser.add_argument("--test-size",            type=float, default=0.20)
    parser.add_argument("--unsafe-threshold",     type=float, default=0.05,
                        help="recall_drop threshold for precision/recall reporting (default: 0.05)")
    parser.add_argument("--n-estimators",         type=int,   default=500)
    parser.add_argument("--learning-rate",        type=float, default=0.05)
    parser.add_argument("--max-depth",            type=int,   default=6)
    parser.add_argument("--no-save",              action="store_true")
    parser.add_argument("--classifier",           action="store_true",
                        help="Train XGBClassifier (binary: recall_drop > 0) instead of regressor")
    parser.add_argument("--resample-ratio",       type=float, default=None,
                        help="Negatives per positive after undersampling (classifier only, e.g. 3.0)")
    parser.add_argument("--min-fresh-recall",     type=float, default=0.80,
                        help="Drop rows where fresh_recall < this threshold (default: 0.80)")
    parser.add_argument("--delete-weight",        type=float, default=1.0,
                        help="Extra sample weight for delete-mutation positive rows in classifier (default: 1.0)")
    parser.add_argument("--dataset-name",         default=None,
                        help="Override output model/metrics filename stem (default: derived from CSV name)")
    args = parser.parse_args()

    if args.auto or args.data is None:
        csv_path = _latest_csv(DATA_DIR)
        print(f"Auto-selected: {csv_path}")
    else:
        csv_path = args.data

    train(
        csv_path         = csv_path,
        nonzero_weight   = args.nonzero_weight,
        test_size        = args.test_size,
        unsafe_threshold = args.unsafe_threshold,
        n_estimators     = args.n_estimators,
        learning_rate    = args.learning_rate,
        max_depth        = args.max_depth,
        save_model       = not args.no_save,
        dataset_name     = args.dataset_name or os.path.splitext(os.path.basename(csv_path))[0],
        use_classifier   = args.classifier,
        resample_ratio   = args.resample_ratio,
        min_fresh_recall = args.min_fresh_recall,
        delete_weight    = args.delete_weight,
    )


if __name__ == "__main__":
    main()
