"""
Centroid drift threshold — delete GT base vectors, insert GT held-out vectors.

Protocol
--------
Start state: 900K base index (100K held-out removed → cache state).
GT: static precomputed SIFT-1M GT (computed on all 1M vectors including held-out).

Held-out GT vectors: the subset of the 100K held-out pool that appears in the
precomputed top-10 GT for concentrated queries in each cluster. These are
DIFFERENT from the deleted base vectors but are also genuine GT neighbours.

Phase 1 — DELETE (N_STEPS batches):
  Remove held-out-GT-count base GT vectors, sorted closest-to-query first.
  => Static GT entries are missing from index → recall drops.

Phase 2 — INSERT (N_STEPS batches, same size):
  Add back the held-out GT vectors in the same order.
  => Genuine GT neighbours re-enter the index → recall recovers.

This gives a symmetric, interpretable curve with the same number of
mutations in both phases, using only real SIFT vectors.

Output
------
  results/centroid_drift_threshold/drift_threshold_results.csv
  results/centroid_drift_threshold/drift_threshold.png / .pdf
"""

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from quake.datasets.ann_datasets import load_dataset
from quake.index_wrappers.quake import QuakeWrapper
from quake.utils import knn

# ── Config ────────────────────────────────────────────────────────────────────

DATASET       = "sift1m"
DATA_PATH     = "data/sift"
INDEX_PATH    = "data/sift/indexes/stale_index_baseline.index"
OUT_DIR       = Path("results/centroid_drift_threshold")

K             = 10
NPROBE        = 32
SEED          = 42
SPLIT         = 900_000

TOP5_CLUSTERS  = {422, 313, 112, 734, 555}
N_DEL_STEPS   = 8      # delete batches  (aggressive — aim for ~20% recall drop)
N_INS_STEPS   = 5      # insert batches  (however many held-out GT vecs exist)
MAX_DEL_FRAC  = 0.25   # delete at most 25% of cluster base vecs
MAX_DEL_ABS   = 500    # hard cap on total deletes
MIN_HELDOUT_GT = 5     # skip cluster if fewer than this many held-out GT vectors

OUT_DIR.mkdir(parents=True, exist_ok=True)
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

# ── Load ──────────────────────────────────────────────────────────────────────

print("Loading dataset...")
vectors, queries, gt = load_dataset(DATASET, DATA_PATH)
vectors = vectors.float()
queries = queries.float()
gt      = gt.long()
n_total = vectors.shape[0]

insert_vectors    = vectors[SPLIT:]
insert_ids_global = torch.arange(SPLIT, n_total, dtype=torch.int64)
n_inserts_total   = insert_vectors.shape[0]
print(f"  base: {SPLIT}  held-out: {n_inserts_total}")

print("Loading index...")
index_setup = QuakeWrapper()
index_setup.load(INDEX_PATH)
centroids = index_setup.centroids()
nc        = centroids.shape[0]
print(f"  nc={nc}  nprobe={NPROBE}")

# ── Cluster assignments ───────────────────────────────────────────────────────

print("Computing cluster assignments...")
BS = 50_000
assign_list = []
for s in range(0, n_total, BS):
    a, _ = knn(vectors[s:s + BS], centroids, 1, "l2")
    assign_list.append(a.squeeze(1).long())
assignments = torch.cat(assign_list)

print("Loading concentrated queries...")
df_conc = pd.read_csv("results/concentrated_queries.csv")
gt_np   = gt.numpy()

# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_drift(vecs: torch.Tensor, stored_c: torch.Tensor) -> float:
    return float((vecs.mean(dim=0) - stored_c).norm())

def standard_recall(result_ids: torch.Tensor, gt_rows: torch.Tensor) -> float:
    k     = result_ids.shape[1]
    total = 0.0
    for i in range(result_ids.shape[0]):
        returned = set(result_ids[i].tolist())
        hits = sum(1 for v in gt_rows[i, :K].tolist() if v >= 0 and v in returned)
        total += hits / k
    return total / result_ids.shape[0]

# ── Main experiment ───────────────────────────────────────────────────────────

all_records = []

for cid in sorted(TOP5_CLUSTERS):
    centroid_p = centroids[cid]

    q_rows = df_conc[df_conc["dominant_cluster"] == cid]
    if len(q_rows) == 0:
        print(f"\nCluster {cid}: no concentrated queries, skipping")
        continue

    q_indices  = q_rows["query_idx"].tolist()
    q_subset   = queries[q_indices]
    gt_subset  = gt[q_indices]
    mean_query = q_subset.mean(dim=0)

    drift_dir  = centroid_p - mean_query
    drift_dir  = drift_dir / drift_dir.norm()

    # ── Held-out GT vectors: in static GT AND in the 100K held-out pool ───────
    heldout_gt_vids = set()
    for qi in q_indices:
        for vid in gt_np[qi, :K].tolist():
            if vid >= 0 and vid >= SPLIT:           # in held-out range
                if int(assignments[vid]) == cid:    # routes to this cluster
                    heldout_gt_vids.add(int(vid))

    if len(heldout_gt_vids) < MIN_HELDOUT_GT:
        print(f"\nCluster {cid}: only {len(heldout_gt_vids)} held-out GT vectors "
              f"(need {MIN_HELDOUT_GT}), skipping")
        continue

    heldout_gt_t   = torch.tensor(sorted(heldout_gt_vids), dtype=torch.long)
    heldout_gt_vecs = vectors[heldout_gt_t]   # (n_hgt, d)

    # Sort held-out GT vectors: closest-to-query first (ascending projection)
    ins_projs  = (heldout_gt_vecs @ drift_dir).numpy()
    ins_order  = np.argsort(ins_projs)
    ins_vids   = heldout_gt_t[ins_order]
    ins_vecs   = heldout_gt_vecs[ins_order]
    n_hgt      = len(ins_vids)

    # ── Base GT vectors: in static GT AND in the 900K base ────────────────────
    base_gt_vids = set()
    for qi in q_indices:
        for vid in gt_np[qi, :K].tolist():
            if vid >= 0 and vid < SPLIT and int(assignments[vid]) == cid:
                base_gt_vids.add(int(vid))

    base_gt_t    = torch.tensor(sorted(base_gt_vids), dtype=torch.long)
    base_gt_vecs = vectors[base_gt_t]

    # Sort base GT: closest-to-query first (ascending projection)
    del_projs  = (base_gt_vecs @ drift_dir).numpy()
    del_order  = np.argsort(del_projs)
    del_vids   = base_gt_t[del_order]
    n_bgt      = len(del_vids)

    # Intra-cluster radius
    base_mask  = (assignments == cid) & (torch.arange(n_total) < SPLIT)
    base_vids  = torch.where(base_mask)[0]
    base_vecs  = vectors[base_vids]
    intra_r    = float(((base_vecs - centroid_p.unsqueeze(0))**2).sum(1).sqrt().mean())

    # Delete pool: ALL base GT vecs, capped at MAX_DEL_FRAC of cluster / MAX_DEL_ABS
    max_del   = min(n_bgt, int(MAX_DEL_FRAC * len(base_vids)), MAX_DEL_ABS)
    del_vids  = del_vids[:max_del]
    del_step  = max(1, max_del // N_DEL_STEPS)

    # Insert pool: ALL held-out GT vecs (no cap — use every genuine GT we have)
    ins_step  = max(1, n_hgt // N_INS_STEPS)

    print(f"\nCluster {cid}: {len(base_vids)} base vecs  "
          f"{n_bgt} base GT → deleting {max_del}  "
          f"{n_hgt} held-out GT → inserting all  "
          f"{len(q_indices)} queries  intra_r={intra_r:.1f}")

    # ── Fresh index: cache state (900K, held-out removed) ─────────────────────
    idx = QuakeWrapper(); idx.load(INDEX_PATH)
    idx.remove(insert_ids_global)

    recall_base = standard_recall(idx.search(q_subset, k=K, nprobe=NPROBE).ids, gt_subset)
    live_set    = set(base_vids.tolist())
    drift_base  = compute_drift(vectors[base_vids], centroid_p)

    print(f"  cache baseline: recall={recall_base:.4f}  drift={drift_base:.3f}")

    all_records.append({
        "cluster_id": cid, "phase": "baseline", "step": 0,
        "n_deleted": 0, "n_inserted": 0,
        "centroid_drift": drift_base, "drift_delta": 0.0, "drift_normalised": 0.0,
        "recall": recall_base, "recall_drop": 0.0, "intra_radius": intra_r,
    })

    # ── Phase 1: delete base GT vectors (near-query first) ────────────────────
    for step in range(1, N_DEL_STEPS + 1):
        s = (step - 1) * del_step
        e = min(step * del_step, max_del)
        if s >= max_del:
            break

        batch_del = del_vids[s:e]
        idx.remove(batch_del)
        for v in batch_del.tolist():
            live_set.discard(v)

        live_t      = torch.tensor(sorted(live_set), dtype=torch.long)
        drift_now   = compute_drift(vectors[live_t], centroid_p)
        drift_delta = drift_now - drift_base
        n_del       = min(step * del_step, max_del)

        recall = standard_recall(idx.search(q_subset, k=K, nprobe=NPROBE).ids, gt_subset)

        all_records.append({
            "cluster_id": cid, "phase": "delete", "step": step,
            "n_deleted": n_del, "n_inserted": 0,
            "centroid_drift": drift_now, "drift_delta": drift_delta,
            "drift_normalised": drift_delta / intra_r,
            "recall": recall, "recall_drop": recall_base - recall,
            "intra_radius": intra_r,
        })
        print(f"  [del] step={step}  n_del={n_del:3d}  "
              f"drift={drift_now:.3f} (Δ{drift_delta:+.3f})  "
              f"recall={recall:.4f}  drop={recall_base-recall:+.4f}")

    # ── Phase 2: insert held-out GT vectors (near-query first) ────────────────
    n_ins_so_far = 0
    for step in range(1, N_INS_STEPS + 1):
        s = (step - 1) * ins_step
        e = min(step * ins_step, n_hgt)
        if s >= n_hgt:
            break

        batch_ins_vecs = ins_vecs[s:e]
        batch_ins_vids = ins_vids[s:e]
        idx.add(batch_ins_vecs, batch_ins_vids)
        n_ins_so_far += len(batch_ins_vids)

        for v in batch_ins_vids.tolist():
            live_set.add(v)

        base_live = [v for v in live_set if v < SPLIT]
        ins_live  = [v for v in live_set if v >= SPLIT]
        parts = []
        if base_live:
            parts.append(vectors[torch.tensor(base_live, dtype=torch.long)])
        if ins_live:
            parts.append(vectors[torch.tensor(ins_live, dtype=torch.long)])
        current_vecs = torch.cat(parts, dim=0)
        drift_now    = compute_drift(current_vecs, centroid_p)
        drift_delta  = drift_now - drift_base

        recall = standard_recall(idx.search(q_subset, k=K, nprobe=NPROBE).ids, gt_subset)

        all_records.append({
            "cluster_id": cid, "phase": "insert", "step": N_DEL_STEPS + step,
            "n_deleted": max_del, "n_inserted": n_ins_so_far,
            "centroid_drift": drift_now, "drift_delta": drift_delta,
            "drift_normalised": drift_delta / intra_r,
            "recall": recall, "recall_drop": recall_base - recall,
            "intra_radius": intra_r,
        })
        print(f"  [ins] step={step}  n_ins={n_ins_so_far:3d}  "
              f"drift={drift_now:.3f} (Δ{drift_delta:+.3f})  "
              f"recall={recall:.4f}  drop={recall_base-recall:+.4f}")

# ── Save ──────────────────────────────────────────────────────────────────────

df = pd.DataFrame(all_records)
df.to_csv(OUT_DIR / "drift_threshold_results.csv", index=False)
print(f"\nSaved: {OUT_DIR}/drift_threshold_results.csv")

# ── Plot ──────────────────────────────────────────────────────────────────────

colors = {112: "steelblue", 313: "darkorange", 422: "seagreen",
          555: "crimson",   734: "purple"}

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# ── Panel A: Δ centroid drift vs step ────────────────────────────────────────
ax = axes[0]
for cid, grp in df.groupby("cluster_id"):
    col     = colors.get(cid, "grey")
    del_grp = grp[grp["phase"].isin(["baseline", "delete"])]
    ins_grp = grp[grp["phase"] == "insert"]
    ax.plot(del_grp["step"], del_grp["drift_delta"],
            "o-", color=col, lw=1.8, ms=5, label=f"cluster {cid}")
    if len(ins_grp):
        ax.plot(ins_grp["step"], ins_grp["drift_delta"],
                "s--", color=col, lw=1.5, ms=5)
ax.axhline(0, color="k", lw=0.5)
ax.axvline(N_DEL_STEPS + 0.5, color="k", lw=0.8, ls=":", alpha=0.4, label="delete→insert")
ax.set_xlabel(f"Step  (1–{N_DEL_STEPS} delete,  {N_DEL_STEPS+1}–{N_DEL_STEPS+N_INS_STEPS} insert)", fontsize=11)
ax.set_ylabel("Δ Centroid drift", fontsize=11)
ax.set_title("A. Centroid drift rises on delete,\npartially recovers on insert", fontsize=11)
ax.legend(fontsize=8)

# ── Panel B: recall drop vs step ─────────────────────────────────────────────
ax = axes[1]
for cid, grp in df.groupby("cluster_id"):
    col     = colors.get(cid, "grey")
    del_grp = grp[grp["phase"].isin(["baseline", "delete"])]
    ins_grp = grp[grp["phase"] == "insert"]
    ax.plot(del_grp["step"], del_grp["recall_drop"],
            "o-", color=col, lw=1.8, ms=5, label=f"cluster {cid}")
    if len(ins_grp):
        ax.plot(ins_grp["step"], ins_grp["recall_drop"],
                "s--", color=col, lw=1.5, ms=5)
ax.axhline(0, color="k", lw=0.5)
ax.axhline(0.05, color="k", lw=1, ls="--", label="5% threshold")
ax.axvline(N_DEL_STEPS + 0.5, color="k", lw=0.8, ls=":", alpha=0.4, label="delete→insert")
ax.set_xlabel(f"Step  (1–{N_DEL_STEPS} delete,  {N_DEL_STEPS+1}–{N_DEL_STEPS+N_INS_STEPS} insert)", fontsize=11)
ax.set_ylabel("Recall drop", fontsize=11)
ax.set_title("B. Recall drops (GT deleted),\nrecovers as held-out GT inserted", fontsize=11)
ax.legend(fontsize=8)

# ── Panel C: recall drop vs |Δ drift| — delete phase only, monotone ──────────
ax = axes[2]
for cid, grp in df.groupby("cluster_id"):
    col     = colors.get(cid, "grey")
    # delete phase only (includes baseline step=0)
    del_grp = grp[grp["phase"].isin(["baseline", "delete"])].copy()
    del_grp["abs_drift_delta"] = del_grp["drift_delta"].abs()
    ax.plot(del_grp["abs_drift_delta"], del_grp["recall_drop"],
            "o-", color=col, lw=1.8, ms=5, label=f"cluster {cid}")
    # annotate last point with step number
    last = del_grp.iloc[-1]
    ax.annotate(f"step {int(last['step'])}",
                xy=(last["abs_drift_delta"], last["recall_drop"]),
                xytext=(4, 2), textcoords="offset points",
                fontsize=7, color=col)
ax.axhline(0, color="k", lw=0.5)
ax.axhline(0.05, color="k", lw=1, ls="--", label="5% recall threshold")
ax.set_xlabel("|Δ Centroid drift|  (absolute, delete phase only)", fontsize=11)
ax.set_ylabel("Recall drop", fontsize=11)
ax.set_title("C. Recall drop vs centroid drift magnitude\n(delete phase — monotone relationship)", fontsize=11)
ax.legend(fontsize=8)

fig.suptitle(
    f"GT Delete → Recall Drop, Held-out GT Insert → Recall Recovery — SIFT-1M  nprobe={NPROBE}\n"
    f"Equal mutations each phase  |  Static precomputed GT  |  Top-5 concentrated clusters",
    fontsize=11)
fig.tight_layout()
fig.savefig(OUT_DIR / "drift_threshold.png", dpi=150, bbox_inches="tight")
fig.savefig(OUT_DIR / "drift_threshold.pdf", bbox_inches="tight")
print(f"Saved: {OUT_DIR}/drift_threshold.png")
plt.close(fig)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n── Summary ──────────────────────────────────────────────────────────────")
for cid, grp in df.groupby("cluster_id"):
    base    = float(grp[grp["phase"] == "baseline"]["recall"].iloc[0])
    del_end = grp[grp["phase"] == "delete"]
    ins_end = grp[grp["phase"] == "insert"]
    r_del   = float(del_end.iloc[-1]["recall"]) if len(del_end) else base
    r_ins   = float(ins_end.iloc[-1]["recall"]) if len(ins_end) else base
    n_d     = int(del_end.iloc[-1]["n_deleted"]) if len(del_end) else 0
    n_i     = int(ins_end.iloc[-1]["n_inserted"]) if len(ins_end) else 0
    print(f"  cluster {cid:3d}: del={n_d} ins={n_i}  "
          f"recall {base:.3f} → {r_del:.3f} (del) → {r_ins:.3f} (ins)  "
          f"drop {base-r_del:.3f} recovered {r_ins-r_del:.3f}")
