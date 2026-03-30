import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
METADATA_DIR = os.path.join(PROJECT_ROOT, "metadata")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

CACHE_META_PATH = os.path.join(METADATA_DIR, "cache_metadata.json")
REMOTE_META_SNAPSHOT_PATH = os.path.join(METADATA_DIR, "remote_metadata_snapshot.json")

MINIO_ENDPOINT = "localhost:9002"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
MINIO_SECURE = False
MINIO_BUCKET = "vector-index"

DIM = 8
NUM_PARTITIONS = 4
VECS_PER_PARTITION = 100

TOPK = 5
QUERY_NODE_HOST = "127.0.0.1"
QUERY_NODE_PORT = 5050

# ensure directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

