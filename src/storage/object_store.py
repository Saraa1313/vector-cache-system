import io
import json
import numpy as np
from minio import Minio
from minio.error import S3Error

from config import (
    MINIO_ENDPOINT,
    MINIO_ACCESS_KEY,
    MINIO_SECRET_KEY,
    MINIO_SECURE,
    MINIO_BUCKET,
)

REMOTE_METADATA_OBJECT = "metadata/remote_metadata.json"


class ObjectStore:
    def __init__(self):
        self.client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=MINIO_SECURE,
        )
        self.bucket = MINIO_BUCKET
        self._ensure_bucket()

    def _ensure_bucket(self):
        if not self.client.bucket_exists(self.bucket):
            self.client.make_bucket(self.bucket)

    def save_partition(self, fname: str, ids, vectors):
        buf = io.BytesIO()
        np.savez(
            buf,
            ids=np.array(ids, dtype=np.int64),
            vectors=np.array(vectors, dtype=np.float32),
        )
        raw = buf.getvalue()

        self.client.put_object(
            self.bucket,
            f"partitions/{fname}",
            io.BytesIO(raw),
            length=len(raw),
            content_type="application/octet-stream",
        )

    def load_partition(self, fname: str):
        response = self.client.get_object(self.bucket, f"partitions/{fname}")
        try:
            raw = response.read()
        finally:
            response.close()
            response.release_conn()

        buf = io.BytesIO(raw)
        arr = np.load(buf)
        return arr["ids"], arr["vectors"]

    def list_partition_names(self):
        names = []
        for obj in self.client.list_objects(self.bucket, prefix="partitions/", recursive=True):
            if obj.object_name.endswith(".npz"):
                names.append(obj.object_name.split("/")[-1])
        return sorted(names)

    def write_metadata(self, metadata: dict):
        raw = json.dumps(metadata, indent=2).encode("utf-8")
        self.client.put_object(
            self.bucket,
            REMOTE_METADATA_OBJECT,
            io.BytesIO(raw),
            length=len(raw),
            content_type="application/json",
        )

    def read_metadata(self) -> dict:
        try:
            response = self.client.get_object(self.bucket, REMOTE_METADATA_OBJECT)
            try:
                return json.loads(response.read().decode("utf-8"))
            finally:
                response.close()
                response.release_conn()
        except S3Error:
            return {}