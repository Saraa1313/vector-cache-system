import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import requests

from config import DIM, QUERY_NODE_HOST, QUERY_NODE_PORT, TOPK


def main():
    q = np.random.randn(DIM).astype("float32")

    url = f"http://{QUERY_NODE_HOST}:{QUERY_NODE_PORT}/search"
    payload = {
        "query": q.tolist(),
        "topk": TOPK,
    }

    r = requests.post(url, json=payload)
    print("status", r.status_code)
    print("text", r.text)

    try:
        print(r.json())
    except Exception as e:
        print("json parse error", e)



if __name__ == "__main__":
    main()