"""
Build the stale baseline IVF index from the SIFT-1M dataset.

Builds on all 1M vectors with nc=1024 clusters (IVF-flat, L2).
Centroids are frozen after build — no incremental recomputation.
This is the "stale" index used by all recall degradation experiments.

Output
------
data/sift/indexes/stale_index_baseline.index
"""

from pathlib import Path

import torch

from quake.datasets.ann_datasets import load_dataset
from quake.index_wrappers.quake import QuakeWrapper

# ── Config ────────────────────────────────────────────────────────────────────

DATASET    = "sift1m"
DATA_PATH  = "data/sift"
INDEX_PATH = "data/sift/indexes/stale_index_baseline.index"
NC         = 1024     # number of IVF clusters  (sqrt(1M) ≈ 1000, round up)

# ── Build ─────────────────────────────────────────────────────────────────────

print("Loading dataset...")
vectors, _, _ = load_dataset(DATASET, DATA_PATH)
vectors = vectors.float()
ids     = torch.arange(vectors.shape[0], dtype=torch.int64)
print(f"  vectors: {vectors.shape}")

print(f"Building IVF index  nc={NC}  metric=l2 ...")
index = QuakeWrapper()
index.build(vectors, nc=NC, metric="l2", ids=ids)

Path(INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)
index.save(INDEX_PATH)
print(f"Saved: {INDEX_PATH}")
