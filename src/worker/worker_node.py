import json
import os
import time
import boto3
import numpy as np
import faiss
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import CENTROIDS_PATH, WORKER_POLL_INTERVAL_S, DYNAMODB_REGION, DYNAMODB_ENDPOINT, META_TABLE, QUERY_NODE_ID
from storage.object_store import ObjectStore
from storage.wal import WALClient

STATE_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "worker_state.json")


class WorkerNode:
    def __init__(self):
        self.store = ObjectStore()
        self.wal = WALClient()

        self.centroids = np.load(CENTROIDS_PATH)
        d = self.centroids.shape[1]
        self._centroid_index = faiss.IndexFlatL2(d)
        self._centroid_index.add(np.ascontiguousarray(self.centroids))

        print("Scanning MinIO to build id→centroid map...", flush=True)
        self.id_to_centroid: dict[int, int] = {}
        for cid in self.store.list_centroid_ids():
            ids, _ = self.store.load_centroid(cid)
            for vid in ids:
                self.id_to_centroid[int(vid)] = cid
        print(f"  {len(self.id_to_centroid)} vectors indexed", flush=True)

        # Initialise the ID counter to max existing ID + 1
        if self.id_to_centroid:
            max_id = max(self.id_to_centroid.keys())
            try:
                db = boto3.resource(
                    "dynamodb",
                    region_name=DYNAMODB_REGION,
                    endpoint_url=DYNAMODB_ENDPOINT,
                    aws_access_key_id="local",
                    aws_secret_access_key="local",
                )
                db.Table(META_TABLE).update_item(
                    Key={"counter_id": "global"},
                    UpdateExpression="SET next_id = :val",
                    ConditionExpression="attribute_not_exists(next_id)",
                    ExpressionAttributeValues={":val": max_id + 1},
                )
                print(f"  ID counter initialised to {max_id + 1}", flush=True)
            except Exception:
                pass  

        # Load last applied sequence number
        self.last_applied_seq: int = 0
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE) as f:
                self.last_applied_seq = json.load(f).get("last_applied_seq", 0)
        print(f"  Resuming from seq_id > {self.last_applied_seq}", flush=True)

    def _nearest_centroid(self, vector: np.ndarray) -> int:
        q = np.ascontiguousarray(vector.reshape(1, -1).astype(np.float32))
        _, I = self._centroid_index.search(q, 1)
        return int(I[0, 0])

    def _persist_state(self):
        with open(STATE_FILE, "w") as f:
            json.dump({"last_applied_seq": self.last_applied_seq}, f)

    def _fetch_partitions(self, partition_ids: set[int]) -> dict[int, tuple]:
        if not partition_ids:
            return {}
        with ThreadPoolExecutor(max_workers=len(partition_ids)) as executor:
            futures = {executor.submit(self.store.load_centroid, cid): cid
                       for cid in partition_ids}
            return {futures[f]: f.result() for f in as_completed(futures)}

    def _write_partitions(self, partitions: dict[int, tuple]):
        with ThreadPoolExecutor(max_workers=len(partitions)) as executor:
            for cid, (ids, vecs) in partitions.items():
                executor.submit(self.store.save_centroid, cid, ids, vecs)

    def apply_batch(self, entries: list[dict]):
        target_partitions: set[int] = set()
        for e in entries:
            op = e["operation"]
            if op == "add":
                vec = np.array([float(x) for x in e["vector"]], dtype=np.float32)
                target_partitions.add(self._nearest_centroid(vec))
            elif op == "delete":
                vid = int(e["vector_id"])
                if vid in self.id_to_centroid:
                    target_partitions.add(self.id_to_centroid[vid])
            elif op == "update":
                vid = int(e["vector_id"])
                if vid in self.id_to_centroid:
                    target_partitions.add(self.id_to_centroid[vid])
                new_vec = np.array([float(x) for x in e["vector"]], dtype=np.float32)
                target_partitions.add(self._nearest_centroid(new_vec))

        partitions = self._fetch_partitions(target_partitions)

        modified: set[int] = set()
        for e in entries:
            op = e["operation"]
            if op == "add":
                vid = int(e["vector_id"])
                vec = np.array([float(x) for x in e["vector"]], dtype=np.float32)
                cid = self._nearest_centroid(vec)
                ids, vecs = partitions[cid]
                partitions[cid] = (
                    np.append(ids, np.int64(vid)),
                    np.vstack([vecs, vec.reshape(1, -1)]),
                )
                self.id_to_centroid[vid] = cid
                modified.add(cid)

            elif op == "delete":
                vid = int(e["vector_id"])
                cid = self.id_to_centroid.get(vid)
                if cid is None:
                    continue
                ids, vecs = partitions[cid]
                mask = ids != vid
                partitions[cid] = (ids[mask], vecs[mask])
                del self.id_to_centroid[vid]
                modified.add(cid)

            elif op == "update":
                vid = int(e["vector_id"])
                old_cid = self.id_to_centroid.get(vid)
                if old_cid is None:
                    continue
                new_vec = np.array([float(x) for x in e["vector"]], dtype=np.float32)
                new_cid = self._nearest_centroid(new_vec)

                if old_cid == new_cid:
                    ids, vecs = partitions[old_cid]
                    vecs = vecs.copy()
                    vecs[ids == vid] = new_vec
                    partitions[old_cid] = (ids, vecs)
                    modified.add(old_cid)
                else:
                    # Remove from old
                    ids, vecs = partitions[old_cid]
                    mask = ids != vid
                    partitions[old_cid] = (ids[mask], vecs[mask])
                    modified.add(old_cid)
                    # Add to new
                    ids, vecs = partitions[new_cid]
                    partitions[new_cid] = (
                        np.append(ids, np.int64(vid)),
                        np.vstack([vecs, new_vec.reshape(1, -1)]),
                    )
                    self.id_to_centroid[vid] = new_cid
                    modified.add(new_cid)

        self._write_partitions({cid: partitions[cid] for cid in modified})
        print(f"  Applied {len(entries)} entries "
              f"(seq {int(entries[0]['seq_id'])}–{int(entries[-1]['seq_id'])}), "
              f"modified {len(modified)} partitions", flush=True)

    def run(self):
        print(f"Worker polling WAL every {WORKER_POLL_INTERVAL_S}s", flush=True)
        while True:
            entries = self.wal.read_after(self.last_applied_seq)
            if entries:
                self.apply_batch(entries)
                self.last_applied_seq = int(entries[-1]["seq_id"])
                self._persist_state()
            time.sleep(WORKER_POLL_INTERVAL_S)
