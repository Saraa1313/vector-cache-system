import io
import numpy as np
from minio import Minio

from config import (
    MINIO_ENDPOINT,
    MINIO_ACCESS_KEY,
    MINIO_SECRET_KEY,
    MINIO_SECURE,
    MINIO_BUCKET,
)


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

    def _key(self, centroid_id: int) -> str:
        return f"centroids/{centroid_id:04d}.npz"

    def save_centroid(self, centroid_id: int, ids, vectors):
        buf = io.BytesIO()
        np.savez(buf,
                 ids=np.array(ids, dtype=np.int64),
                 vectors=np.array(vectors, dtype=np.float32))
        raw = buf.getvalue()
        self.client.put_object(
            self.bucket, self._key(centroid_id),
            io.BytesIO(raw), length=len(raw),
            content_type="application/octet-stream",
        )

    def load_centroid(self, centroid_id: int):
        response = self.client.get_object(self.bucket, self._key(centroid_id))
        try:
            raw = response.read()
        finally:
            response.close()
            response.release_conn()
        arr = np.load(io.BytesIO(raw))
        return arr["ids"], arr["vectors"]

    def list_centroid_ids(self):
        ids = []
        for obj in self.client.list_objects(self.bucket, prefix="centroids/", recursive=True):
            if obj.object_name.endswith(".npz"):
                stem = obj.object_name.split("/")[-1].replace(".npz", "")
                ids.append(int(stem))
        return sorted(ids)
