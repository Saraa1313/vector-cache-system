#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

_cfg() { python -c "import sys; sys.path.insert(0,'src'); from config import $1; print($1)"; }
MINIO_PORT="$(_cfg MINIO_ENDPOINT | cut -d: -f2)"
MINIO_USER="$(_cfg MINIO_ACCESS_KEY)"
MINIO_PASS="$(_cfg MINIO_SECRET_KEY)"
GRPC_PORT="$(_cfg GRPC_PORT)"

cleanup() {
    echo ""
    echo "Shutting down processes ..."
    kill "$NODE_PID" "$MINIO_PID" 2>/dev/null || true
    wait "$NODE_PID" "$MINIO_PID" 2>/dev/null || true

    # echo "Deleting data directories ..."
    # rm -rf ~/minio/data
    # rm -f  "$PROJECT_ROOT/centroids.npy"
    echo "Cleanup done."
}
trap cleanup EXIT

echo "Starting MinIO ..."
mkdir -p ~/minio/data
export MINIO_ROOT_USER="$MINIO_USER"
export MINIO_ROOT_PASSWORD="$MINIO_PASS"
minio server ~/minio/data --address ":${MINIO_PORT}" --console-address ":$((MINIO_PORT + 1))" &
MINIO_PID=$!
sleep 2

echo "Building index ..."
python src/admin/build_index.py

echo "Starting query node ..."
python src/query/start_query_node.py &
NODE_PID=$!
sleep 10

echo "Running client ..."
python src/client/client.py
