import numpy as np


def read_fvecs(path: str) -> np.ndarray:
    fv = np.fromfile(path, dtype=np.float32)
    d = fv.view(np.int32)[0]
    return fv.reshape(-1, d + 1)[:, 1:]


def read_ivecs(path: str) -> np.ndarray:
    iv = np.fromfile(path, dtype=np.int32)
    d = iv[0]
    return iv.reshape(-1, d + 1)[:, 1:]
