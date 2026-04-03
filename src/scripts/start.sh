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

echo "Starting DynamoDB Local ..."
mkdir -p /tmp/dynamodb-local-logs

JAVA17=$(ls -d /usr/lib/jvm/java-17-openjdk-* 2>/dev/null | head -1)
if [ -n "$JAVA17" ] && [ -x "$JAVA17/bin/java" ]; then export PATH="$JAVA17/bin:$PATH"; fi
java -jar "$DYNAMO_JAR" -inMemory -port "$DYNAMO_PORT" \
    > /tmp/dynamodb-local-logs/dynamodb.log 2>&1 &
DYNAMO_PID=$!

echo -n "  Waiting for DynamoDB..."
attempt=0
while [ $attempt -lt 30 ]; do
    if nc -z localhost "$DYNAMO_PORT" 2>/dev/null; then
        echo " ready"
        break
    fi
    sleep 1
    attempt=$((attempt + 1))
done
if [ $attempt -ge 30 ]; then
    echo " TIMEOUT"
    echo "Error: DynamoDB did not start. Check /tmp/dynamodb-local-logs/dynamodb.log"
    exit 1
fi

echo "Creating DynamoDB tables ..."
python src/admin/create_dynamo_tables.py

echo "Starting query node ..."
python src/query/start_query_node.py &
NODE_PID=$!
sleep 2

echo "Starting worker node ..."
python src/worker/start_worker_node.py &
WORKER_PID=$!
sleep 2

echo "Running client ..."
python src/client/client.py
# python src/client/insert_client.py 10
# sleep 5
# python src/client/insert_client.py 10
# sleep 5