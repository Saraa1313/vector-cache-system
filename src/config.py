import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MINIO_ENDPOINT = "localhost:9002"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
MINIO_SECURE = False
MINIO_BUCKET = "vector-index"

DATA_DIR = os.path.join(PROJECT_ROOT, "data")

DIM = 128       
N_CENTROIDS = 1000
N_PROBE = [1, 2, 5, 10, 20, 50, 100]

TOPK = 10
QUERY_NODE_HOST = "127.0.0.1"
GRPC_PORT = 50051
NUMBER_OF_QUERIES = 10000

CENTROIDS_PATH = os.path.join(PROJECT_ROOT, "centroids.npy")
