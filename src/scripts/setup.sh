#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"
REQ="$PROJECT_ROOT/requirements.txt"

echo "==> Python / pip"
if ! python3 -m pip --version >/dev/null 2>&1; then
  echo "pip missing; trying ensurepip ..."
  if ! python3 -m ensurepip --user 2>/dev/null; then
    echo "Install pip first, e.g.: python3 -m ensurepip --user"
    echo "  or: curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && python3 /tmp/get-pip.py --user"
    exit 1
  fi
fi
python3 -m pip install --user -r "$REQ"

echo "==> SIFT1M dataset (base, query, ground truth)"
bash "$PROJECT_ROOT/src/scripts/download_sift.sh"

echo "==> MinIO server binary"
if command -v minio >/dev/null 2>&1; then
  echo "minio already on PATH ($(command -v minio))"
else
  MINIO_INSTALL_DIR="${MINIO_INSTALL_DIR:-$HOME/.local/bin}"
  mkdir -p "$MINIO_INSTALL_DIR"
  ARCH=$(uname -m)
  case "$ARCH" in
    x86_64) MINIO_ARCH=amd64 ;;
    aarch64|arm64) MINIO_ARCH=arm64 ;;
    *)
      echo "Unsupported machine: $ARCH (need x86_64 or aarch64)"
      exit 1
      ;;
  esac
  URL="https://dl.min.io/server/minio/release/linux-${MINIO_ARCH}/minio"
  echo "Downloading MinIO → $MINIO_INSTALL_DIR/minio"
  curl -fsSL "$URL" -o "$MINIO_INSTALL_DIR/minio"
  chmod +x "$MINIO_INSTALL_DIR/minio"
  if [[ ":$PATH:" != *":$MINIO_INSTALL_DIR:"* ]]; then
    echo ""
    echo "Add MinIO to PATH, e.g.:"
    echo "  export PATH=\"$MINIO_INSTALL_DIR:\$PATH\""
    echo "Or point start.sh at it:"
    echo "  export MINIO_BIN=\"$MINIO_INSTALL_DIR/minio\""
  fi
fi

echo "==> Java 17"
if java -version 2>&1 | grep -q 'version "17'; then
  echo "Java 17 already installed"
else
  sudo apt-get install -y openjdk-17-jdk
  JAVA17=$(ls -d /usr/lib/jvm/java-17-openjdk-* 2>/dev/null | head -1)
  if [ -n "$JAVA17" ]; then
    sudo update-alternatives --set java "$JAVA17/bin/java" 2>/dev/null || true
  fi
fi

echo "==> DynamoDB Local"
DYNAMO_JAR="/usr/local/lib/dynamodb-local/DynamoDBLocal.jar"
if [ -f "$DYNAMO_JAR" ]; then
  echo "DynamoDB Local already installed"
else
  TMP=$(mktemp -d)
  echo "Downloading DynamoDB Local → $TMP"
  curl -fsSL "https://d1ni2b6xgvw0s0.cloudfront.net/v2.x/dynamodb_local_latest.tar.gz" \
    -o "$TMP/dynamodb_local.tar.gz"
  tar -xzf "$TMP/dynamodb_local.tar.gz" -C "$TMP"
  sudo mkdir -p /usr/local/lib/dynamodb-local
  sudo mv "$TMP/DynamoDBLocal.jar" /usr/local/lib/dynamodb-local/
  sudo mv "$TMP/DynamoDBLocal_lib" /usr/local/lib/dynamodb-local/
  rm -rf "$TMP"
  echo "DynamoDB Local installed to /usr/local/lib/dynamodb-local/"
fi

echo ""
echo "Setup finished. Next: bash src/scripts/start.sh"
