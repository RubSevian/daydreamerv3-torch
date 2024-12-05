import collections
import threading
import time
import uuid

import numpy as np
from tools import convert


class FixedLength:

    def __init__(self, store, chunk, length=0, sync=0, minlen=0):
        self.store = store
        self.chunk = chunk  # batch_length
        self.minlen = minlen
        self.length = length
        self.random = np.random.RandomState(seed=0)
        self.ongoing = collections.defaultdict(lambda: collections.defaultdict(list))
        if sync:
            self.last_scan = time.time()
            # TODO: How can we propagate exceptions from this worker thread?
            self.thread = threading.Thread(target=self._sync, args=(sync,), daemon=True)
            self.thread.start()

    def __len__(self):
        return self.store.steps

    @property
    def stats(self):
        return {f"replay_{k}": v for k, v in self.store.stats().items()}

    def add(self, tran, env_id):
        if tran["is_first"]:
            self.ongoing[env_id].clear()
        ep = self.ongoing[env_id]
        [ep[k].append(v) for k, v in tran.items()]
        if tran["is_last"] or (self.length and len(ep["is_first"]) >= self.length):
            self.add_traj(self.ongoing.pop(env_id))

    def add_traj(self, traj):
        length = len(next(iter(traj.values())))
        if length < self.chunk or length < self.minlen:
            print(f"Skipping short trajectory of length {length}.")
            return
        traj = {k: v for k, v in traj.items() if not k.startswith("log_")}
        traj = {k: convert(v) for k, v in traj.items()}
        self.store[uuid.uuid4().hex] = traj

    def dataset(self):
        while True:
            traj = self._sample()
            if traj is None:
                print("Waiting for episodes.")
                time.sleep(1)
                continue
            yield traj

    def _sample(self):
        keys = self.store.keys()
        if not keys:
            return None
        traj = self.store[keys[self.random.randint(0, len(keys))]]
        total = len(next(iter(traj.values())))
        lower = 0
        upper = total - self.chunk + 1
        index = self.random.randint(lower, upper)
        index = np.clip(index, 0, total - self.chunk)
        chunk = {k: traj[k][index : index + self.chunk] for k in traj.keys()}
        chunk["is_first"] = np.zeros(len(chunk["is_first"]), bool)
        chunk["is_first"][0] = True
        return chunk

    def _sync(self, interval):
        while True:
            time.sleep(max(0, self.last_scan + interval - time.time()))
            self.last_scan = time.time()
            self.store.sync()
