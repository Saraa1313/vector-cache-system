# Vector Cache System (Quake + MinIO)

This project implements a **disaggregated vector search system** with:

- **Quake** as the authoritative IVF-style index
- **MinIO** as object storage (S3-like)
- a **query node cache** that stores snapshot partitions locally
- a **local Quake index on the query node** for serving queries with `nprobe`
- **snapshot-based freshness tracking**

---

## Architecture

```text
Updates
  ↓
Quake (authority node)
  ↓
snapshot export
  ↓
MinIO (object storage)
  ↓
query node cache
  ↓
local Quake rebuild
  ↓
query serving (nprobe search)
Key Idea
The authoritative index lives in Quake
Snapshots are exported to MinIO
Query node pulls snapshots into a local cache
Query node rebuilds a local Quake index
Queries are served using local IVF + nprobe
Cache freshness is tracked via snapshot versions
Project Structure
src/
  admin/
    quake_backend.py
    quake_state.py
    build_authoritative_index_quake.py
    apply_updates_quake.py
    export_snapshot_to_minio.py

  storage/
    object_store.py

  query/
    cache_manager.py
    warm_cache.py
    local_quake_engine.py
    start_query_node.py

  client/
    client.py

  config.py
Requirements
Python 3.10
MinIO installed locally
Homebrew (Mac):
cmake
libomp
openblas
A working Quake fork
Important: Quake Setup

Quake is NOT inside this repo.

You must:

1. Clone Quake separately
cd ~
git clone https://github.com/metonymic-smokey/quake.git quake-fork
cd quake-fork
2. Install into your environment
source ~/vector-cache-prototype/venv_quake/bin/activate
pip install -e . --no-build-isolation
3. Verify
python -c "import quake; print('quake works')"

👉 Because we use pip install -e ., any changes to Quake reflect automatically.

Setup
1. Create environment
python3.10 -m venv venv_quake
source venv_quake/bin/activate
2. Install dependencies
pip install minio numpy torch pybind11 flask requests
Running the System
Step 1: Start MinIO
mkdir -p ~/minio/data

export MINIO_ROOT_USER=minioadmin
export MINIO_ROOT_PASSWORD=minioadmin

minio server ~/minio/data --address ":9002" --console-address ":9003"

MinIO UI:

http://127.0.0.1:9003
Step 2: Build authoritative index
python src/admin/build_authoritative_index_quake.py
Step 3: Warm cache
python src/query/warm_cache.py
Step 4: Start query node
python src/query/start_query_node.py
Step 5: Build local Quake on query node
curl -X POST http://127.0.0.1:5050/rebuild_local_index
Step 6: Run client
python src/client/client.py
Updates

Apply updates to the system:

python src/admin/apply_updates_quake.py

Then refresh cache:

python src/query/warm_cache.py
curl -X POST http://127.0.0.1:5050/rebuild_local_index
Cache Status

Check freshness:

curl http://127.0.0.1:5050/cache_status

Shows:

remote snapshot version
cached snapshot version
stale/fresh status
per-partition version info
Data Locations
MinIO
partition files: part_0000.npz
metadata

UI:

http://127.0.0.1:9003
Authority state
state/
  ids.npy
  vectors.npy
  meta.json
Cache
cache/
metadata/
How Query Serving Works
Query arrives at query node
Local Quake index is used
nprobe controls how many partitions are searched
Results returned
Common Issues
ModuleNotFoundError: quake
source venv_quake/bin/activate
ModuleNotFoundError: minio
pip install minio
Ports already in use
kill -9 $(lsof -ti :9002)
kill -9 $(lsof -ti :9003)
kill -9 $(lsof -ti :5050)
Query returns 500

Check Flask logs in query node terminal.

Current Capabilities
Quake authoritative indexing
Snapshot export to MinIO
Cache warmup
Local Quake rebuild
Query serving with nprobe
Snapshot version tracking
Next Step

Add a controller to decide:

serve stale cache
or refresh before query
Quick Run
# Terminal 1
minio server ~/minio/data --address ":9002" --console-address ":9003"

# Terminal 2
python src/admin/build_authoritative_index_quake.py

# Terminal 3
python src/query/warm_cache.py

# Terminal 4
python src/query/start_query_node.py

# Terminal 5
curl -X POST http://127.0.0.1:5050/rebuild_local_index

# Terminal 6
python src/client/client.py
