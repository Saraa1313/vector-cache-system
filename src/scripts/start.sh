#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

_cfg() { python -c "import sys; sys.path.insert(0,'src'); from config import $1; print($1)"; }
MINIO_PORT="$(_cfg MINIO_ENDPOINT | cut -d: -f2)"
MINIO_USER="$(_cfg MINIO_ACCESS_KEY)"
MINIO_PASS="$(_cfg MINIO_SECRET_KEY)"
GRPC_PORT="$(_cfg GRPC_PORT)"
DYNAMO_PORT="$(_cfg DYNAMODB_PORT)"

DYNAMO_JAR="/usr/local/lib/dynamodb-local/DynamoDBLocal.jar"

cleanup() {
    echo ""
    echo "Shutting down processes ..."
    kill "$NODE_PID" "$WORKER_PID" "$MINIO_PID" "$DYNAMO_PID" 2>/dev/null || true
    wait "$NODE_PID" "$WORKER_PID" "$MINIO_PID" "$DYNAMO_PID" 2>/dev/null || true
    rm -rf $PROJECT_ROOT/worker_state.json
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

DYNAMO_PORT="$DYNAMO_PORT" bash src/scripts/start_dynamo.sh
DYNAMO_PID=$(cat /tmp/dynamodb-local.pid)

echo "Starting query node ..."
python src/query/start_query_node.py &
NODE_PID=$!
sleep 2

echo "Starting worker node ..."
python src/worker/start_worker_node.py &
WORKER_PID=$!
sleep 2

echo "Running client ..."
python src/client/client.py --queries src/client/concentrated_queries.csv
python src/client/update_client.py --csv src/client/update_workload.csv --limit 500
sleep 10
python src/admin/compute_ground_truth.py --queries src/client/concentrated_queries.csv
sleep 5
python src/client/client.py --queries src/client/concentrated_queries.csv --gt data/sift/current_groundtruth.npy