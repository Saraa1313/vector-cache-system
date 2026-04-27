# Instructions for running compound experiments

Clone the quake repo and run these scripts from within it

Run the python scripts in the following order
  build_baseline_index.py          ← run first  (~2 min)
  generate_concentrated_queries.py ← run second (~1 min)
  centroid_drift_threshold.py      ← run third  (~5 min)

data/sift/                         ← only external dependency
uiuc-vectordbs/quake/
└── data/
    └── sift/
        └── sift/
            ├── sift_base.fvecs
            ├── sift_query.fvecs
            ├── sift_groundtruth.ivecs
            └── sift_learn.fvecs
