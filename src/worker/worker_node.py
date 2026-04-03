import json
import os
import time
from collections import defaultdict
from decimal import Decimal

import boto3
import numpy as np
import faiss
from concurrent.futures import ThreadPoolExecutor, as_completed

import grpc
import proto.vector_search_pb2 as pb2
import proto.vector_search_pb2_grpc as pb2_grpc

from config import (CENTROIDS_PATH, WORKER_POLL_INTERVAL_S,
                    DYNAMODB_REGION, DYNAMODB_ENDPOINT,
                    META_TABLE, PARTITION_META_TABLE,
                    QUERY_NODE_HOST, GRPC_PORT)
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

        channel = grpc.insecure_channel(f"{QUERY_NODE_HOST}:{GRPC_PORT}")
        self._qnode_stub = pb2_grpc.VectorSearchStub(channel)

        self._db = boto3.resource(
            "dynamodb",
            region_name=DYNAMODB_REGION,
            endpoint_url=DYNAMODB_ENDPOINT,
            aws_access_key_id="local",
            aws_secret_access_key="local",
        )
        self._part_meta = self._db.Table(PARTITION_META_TABLE)

        self.partition_version: dict[int, int] = {}
        self.partition_size: dict[int, int] = {}

        print("Scanning MinIO to build id→centroid map...", flush=True)
        self.id_to_centroid: dict[int, int] = {}
        for cid in self.store.list_centroid_ids():
            ids, _, version = self.store.load_centroid(cid)
            for vid in ids:
                self.id_to_centroid[int(vid)] = cid
            self.partition_version[cid] = version
            self.partition_size[cid] = len(ids)
        print(f"  {len(self.id_to_centroid)} vectors indexed", flush=True)

        # Initialise the ID counter to max existing ID + 1
        if self.id_to_centroid:
            max_id = max(self.id_to_centroid.keys())
            try:
                self._db.Table(META_TABLE).update_item(
                    Key={"counter_id": "global"},
                    UpdateExpression="SET next_id = :val",
                    ConditionExpression="attribute_not_exists(next_id)",
                    ExpressionAttributeValues={":val": max_id + 1},
                )
                print(f"  ID counter initialised to {max_id + 1}", flush=True)
            except Exception:
                pass

        # Write current partition versions to DynamoDB metadata table
        self._init_partition_metadata()

        # Load last applied sequence number
        self.last_applied_seq: int = 0
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE) as f:
                self.last_applied_seq = json.load(f).get("last_applied_seq", 0)
        print(f"  Resuming from seq_id > {self.last_applied_seq}", flush=True)


    def _init_partition_metadata(self):
        n_centroids = len(self.centroids)
        print(f"Initialising partition metadata for {n_centroids} partitions ...", flush=True)

        def _write_one(cid):
            version = self.partition_version.get(cid, 1)
            centroid_vec = [Decimal(str(float(x))) for x in self.centroids[cid]]
            self._part_meta.put_item(Item={
                "Partition_ID":   cid,
                "Version_ID":     version,
                "Number_of_Inserts":  0,
                "Number_of_Updates":  0,
                "Number_of_Deletes":  0,
                "partition_size":     self.partition_size.get(cid, 0),
                "fraction_vectors_touched_this_version": Decimal("0"),
                "membership_change_count_this_version":  0,
                "new_centroid":   centroid_vec,
            })

        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = [executor.submit(_write_one, cid) for cid in range(n_centroids)]
            for f in as_completed(futures):
                f.result()

        print("Partition metadata initialised.", flush=True)

    def _write_partition_metadata(self, cid: int, ids: np.ndarray, vecs: np.ndarray,
                                  inserts: int, updates: int, deletes: int,
                                  membership_changes: int, size_before: int):
        version = self.partition_version[cid]

        touched = updates + deletes
        fraction = Decimal(str(round(100.0 * touched / size_before, 4))) if size_before > 0 else Decimal("0")

        new_centroid = (np.mean(vecs, axis=0) if len(ids) > 0
                        else np.zeros(self.centroids.shape[1], dtype=np.float32))
        centroid_vec = [Decimal(str(float(x))) for x in new_centroid]

        self._part_meta.put_item(Item={
            "Partition_ID":   cid,
            "Version_ID":     version,
            "Number_of_Inserts":  inserts,
            "Number_of_Updates":  updates,
            "Number_of_Deletes":  deletes,
            "partition_size":     len(ids),
            "fraction_vectors_touched_this_version": fraction,
            "membership_change_count_this_version":  membership_changes,
            "new_centroid":   centroid_vec,
        })


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
            result = {}
            for f in as_completed(futures):
                cid = futures[f]
                ids, vecs, version = f.result()
                self.partition_version[cid] = version
                result[cid] = (ids, vecs)
            return result

    def _write_partitions(self, partitions: dict[int, tuple]):
        with ThreadPoolExecutor(max_workers=len(partitions)) as executor:
            for cid, (ids, vecs) in partitions.items():
                executor.submit(self.store.save_centroid, cid, ids, vecs,
                                self.partition_version[cid])

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

        # Capture partition sizes before any mutations
        partition_size_before: dict[int, int] = {cid: len(data[0]) for cid, data in partitions.items()}

        # Per-partition mutation counters
        p_inserts:    dict[int, int] = defaultdict(int)
        p_updates:    dict[int, int] = defaultdict(int)
        p_deletes:    dict[int, int] = defaultdict(int)
        p_membership: dict[int, int] = defaultdict(int)

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
                p_inserts[cid] += 1

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
                p_deletes[cid] += 1

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
                    p_updates[old_cid] += 1
                else:
                    # Remove from old partition
                    ids, vecs = partitions[old_cid]
                    mask = ids != vid
                    partitions[old_cid] = (ids[mask], vecs[mask])
                    modified.add(old_cid)
                    p_updates[old_cid] += 1
                    p_membership[old_cid] += 1
                    # Add to new partition
                    ids, vecs = partitions[new_cid]
                    partitions[new_cid] = (
                        np.append(ids, np.int64(vid)),
                        np.vstack([vecs, new_vec.reshape(1, -1)]),
                    )
                    self.id_to_centroid[vid] = new_cid
                    modified.add(new_cid)
                    p_updates[new_cid] += 1
                    p_membership[new_cid] += 1

        # Increment versions and update sizes for all modified partitions before writing
        for cid in modified:
            self.partition_version[cid] = self.partition_version.get(cid, 1) + 1
            self.partition_size[cid] = len(partitions[cid][0])

        self._write_partitions({cid: partitions[cid] for cid in modified})

        # Write DynamoDB metadata for each modified partition in parallel
        with ThreadPoolExecutor(max_workers=len(modified)) as executor:
            for cid in modified:
                ids, vecs = partitions[cid]
                executor.submit(
                    self._write_partition_metadata,
                    cid, ids, vecs,
                    p_inserts[cid], p_updates[cid], p_deletes[cid], p_membership[cid],
                    partition_size_before.get(cid, 0),
                )

        last_seq = int(entries[-1]["seq_id"])
        print(f"  Applied {len(entries)} entries "
              f"(seq {int(entries[0]['seq_id'])}–{last_seq}), "
              f"modified {len(modified)} partitions", flush=True)
        print("  Partition metadata after batch:", flush=True)
        for cid in sorted(modified):
            ids, vecs = partitions[cid]
            size_before = partition_size_before.get(cid, 0)
            touched = p_updates[cid] + p_deletes[cid]
            fraction = round(100.0 * touched / size_before, 4) if size_before > 0 else 0.0
            new_centroid = np.mean(vecs, axis=0) if len(vecs) > 0 else np.zeros(vecs.shape[1])
            centroid_norm = float(np.linalg.norm(new_centroid))
            print(f"    cid={cid:04d}  ver={self.partition_version[cid]}"
                  f"  size_before={size_before} size_after={len(ids)}"
                  f"  ins={p_inserts[cid]} upd={p_updates[cid]} del={p_deletes[cid]}"
                  f"  membership_changes={p_membership[cid]}"
                  f"  frac_touched={fraction:.4f}%"
                  f"  centroid_norm={centroid_norm:.4f}", flush=True)

        try:
            self._qnode_stub.NotifyBatchApplied(pb2.BatchAppliedNotification(
                last_seq_id=last_seq,
                modified_partition_ids=list(modified),
            ))
        except grpc.RpcError as e:
            print(f"  Warning: could not notify query node: {e}", flush=True)

    def run(self):
        print(f"Worker polling WAL every {WORKER_POLL_INTERVAL_S}s", flush=True)
        while True:
            entries = self.wal.read_after(self.last_applied_seq)
            if entries:
                self.apply_batch(entries)
                self.last_applied_seq = int(entries[-1]["seq_id"])
                self._persist_state()
            time.sleep(WORKER_POLL_INTERVAL_S)
