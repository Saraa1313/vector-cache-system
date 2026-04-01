#!/bin/bash
# Regenerate Python stubs from vector_search.proto
set -e
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT/src"
python3 -m grpc_tools.protoc -I proto --python_out=proto --grpc_python_out=proto proto/vector_search.proto
echo "OK: $ROOT/src/proto/vector_search_pb2*.py"
