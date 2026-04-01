set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DEST_DIR="${DEST_DIR:-$PROJECT_ROOT/data/sift}"
mkdir -p "$DEST_DIR"

HF="https://huggingface.co/datasets/qbo-odp/sift1m/resolve/main"

download_one() {
  local name="$1"
  local out="$DEST_DIR/$name"
  local url="$HF/$name"
  if [ -f "$out" ]; then
    echo "Already present: $out"
    return 0
  fi
  echo "Downloading $name → $out"
  curl -fL --progress-bar -C - -o "$out" "$url"
}

download_one "sift_base.fvecs"
download_one "sift_query.fvecs"
download_one "sift_groundtruth.ivecs"
echo "All SIFT1M files ready under $DEST_DIR"
