"""
Generate concentrated_queries.csv.

A query is "concentrated" if >= MIN_GT_IN_ONE_CLUSTER of its top-K GT
neighbours all fall in the same IVF cluster.

Output
------
results/concentrated_queries.csv
  query_idx          : 0-indexed query ID (into the SIFT-1M query set)
  dominant_cluster   : cluster ID that holds >= MIN_GT_IN_ONE_CLUSTER GT vecs
  gt_count_in_cluster: how many of the top-K GT vecs are in that cluster
"""

from pathlib import Path

import pandas as pd
import torch

from quake.datasets.ann_datasets import load_dataset
from quake.index_wrappers.quake import QuakeWrapper
from quake.utils import knn

# ── Config ────────────────────────────────────────────────────────────────────

DATASET             = "sift1m"
DATA_PATH           = "data/sift"
INDEX_PATH          = "data/sift/indexes/stale_index_baseline.index"
OUT_PATH            = Path("results/concentrated_queries.csv")

K                   = 10       # top-K GT neighbours to inspect
MIN_GT_IN_ONE_CLUSTER = 6      # threshold: >= 6/10 GT in one cluster

# ── Load ──────────────────────────────────────────────────────────────────────

print("Loading dataset...")
vectors, queries, gt = load_dataset(DATASET, DATA_PATH)
vectors = vectors.float()
gt      = gt.long()

print("Loading index...")
index = QuakeWrapper()
index.load(INDEX_PATH)
centroids = index.centroids()
nc        = centroids.shape[0]
print(f"  nc={nc}  queries={queries.shape[0]}")

# ── Cluster assignments for all 1M vectors ────────────────────────────────────

print("Computing cluster assignments for all vectors...")
BS = 50_000
assign_list = []
for s in range(0, vectors.shape[0], BS):
    a, _ = knn(vectors[s:s + BS], centroids, 1, "l2")
    assign_list.append(a.squeeze(1).long())
assignments = torch.cat(assign_list)   # (1M,)
gt_np = gt.numpy()

# ── Find concentrated queries ─────────────────────────────────────────────────

print(f"Finding queries with >= {MIN_GT_IN_ONE_CLUSTER}/{K} GT in one cluster...")
records = []

for qi in range(queries.shape[0]):
    # Count GT top-K hits per cluster
    cluster_counts: dict[int, int] = {}
    for vid in gt_np[qi, :K]:
        if vid < 0:
            continue
        cid = int(assignments[vid])
        cluster_counts[cid] = cluster_counts.get(cid, 0) + 1

    if not cluster_counts:
        continue

    dom_cid   = max(cluster_counts, key=cluster_counts.__getitem__)
    dom_count = cluster_counts[dom_cid]

    if dom_count >= MIN_GT_IN_ONE_CLUSTER:
        records.append({
            "query_idx":           qi,
            "dominant_cluster":    dom_cid,
            "gt_count_in_cluster": dom_count,
        })

df = pd.DataFrame(records)
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_PATH, index=False)

print(f"\nConcentrated queries: {len(df)} / {queries.shape[0]} "
      f"({100*len(df)/queries.shape[0]:.1f}%)")
print(f"Top 10 clusters by query count:")
print(df["dominant_cluster"].value_counts().head(10).to_string())
print(f"\nSaved: {OUT_PATH}")
