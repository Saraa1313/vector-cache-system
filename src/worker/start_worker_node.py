import os
import sys
_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_src)

from worker.worker_node import WorkerNode

if __name__ == "__main__":
    WorkerNode().run()
