import numpy as np
import faiss


def train_coarse_quantizer(xb, d, nlist, seed=42):
    xb = np.ascontiguousarray(xb.astype(np.float32))
    kmeans = faiss.Kmeans(d, nlist, niter=20, verbose=True, seed=seed, gpu=False)
    kmeans.train(xb)
    return np.ascontiguousarray(kmeans.centroids.astype(np.float32))


def assign_to_centroids(x, centroids):
    x = np.ascontiguousarray(x.astype(np.float32))
    centroids = np.ascontiguousarray(centroids.astype(np.float32))
    index = faiss.IndexFlatL2(centroids.shape[1])
    index.add(centroids)
    _, I = index.search(x, 1)
    return I[:, 0]
