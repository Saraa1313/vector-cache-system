#!/bin/bash
# Starts DynamoDB Local and creates tables.
# Run this on whichever node hosts DynamoDB.
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

DYNAMO_JAR="/usr/local/lib/dynamodb-local/DynamoDBLocal.jar"
DYNAMO_PORT="${DYNAMO_PORT:-8000}"

echo "Starting DynamoDB Local ..."
mkdir -p /tmp/dynamodb-local-logs

JAVA17=$(ls -d /usr/lib/jvm/java-17-openjdk-* 2>/dev/null | head -1)
if [ -n "$JAVA17" ] && [ -x "$JAVA17/bin/java" ]; then export PATH="$JAVA17/bin:$PATH"; fi
java -jar "$DYNAMO_JAR" -inMemory -port "$DYNAMO_PORT" \
    > /tmp/dynamodb-local-logs/dynamodb.log 2>&1 &
echo $! > /tmp/dynamodb-local.pid

echo -n "  Waiting for DynamoDB..."
attempt=0
while [ $attempt -lt 30 ]; do
    if nc -z localhost "$DYNAMO_PORT" 2>/dev/null; then
        echo " ready (PID $(cat /tmp/dynamodb-local.pid))"
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
