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

echo ""
echo "Setup finished. Next: bash src/scripts/start.sh"
