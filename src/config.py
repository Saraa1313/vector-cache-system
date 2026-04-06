import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# MINIO_ENDPOINT = "172.22.152.105:9002"
MINIO_ENDPOINT = "localhost:9002"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
MINIO_SECURE = False
MINIO_BUCKET = "vector-index"

DATA_DIR = os.path.join(PROJECT_ROOT, "data")

DIM = 128       
N_CENTROIDS = 1024
N_PROBE = [20]

TOPK = 10
CACHE_SIZE = 200   # max centroid objects held in memory
QUERY_NODE_ID  = "qnode-0"
# QUERY_NODE_HOST = "172.22.152.104"
QUERY_NODE_HOST = "127.0.0.1"
GRPC_PORT = 50051
NUMBER_OF_QUERIES = 100

CENTROIDS_PATH = os.path.join(PROJECT_ROOT, "centroids.npy")

DYNAMODB_REGION        = "us-east-1"
DYNAMODB_IP            = "127.0.0.1"  
DYNAMODB_PORT          = 8000
DYNAMODB_ENDPOINT      = f"http://{DYNAMODB_IP}:{DYNAMODB_PORT}"
WAL_TABLE              = "WAL"
META_TABLE             = "VectorIndexMeta"
WORKER_POLL_INTERVAL_S = 1.0