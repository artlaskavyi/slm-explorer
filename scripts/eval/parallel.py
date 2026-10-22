"""Run independent jobs in a pool and print progress."""

import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed


def run_parallel(fn, jobs, workers, processes=False):
    """Call fn(*args) for every {key: args} in jobs; yield (key, result) as they finish.
    processes=True for CPU-bound work, threads otherwise (network, git)."""
    pool_cls = ProcessPoolExecutor if processes else ThreadPoolExecutor
    t0 = time.time()
    with pool_cls(workers) as pool:
        futures = {pool.submit(fn, *args): key for key, args in jobs.items()}
        for n, fut in enumerate(as_completed(futures), 1):
            key = futures[fut]
            yield key, fut.result()
            elapsed = time.time() - t0
            left = elapsed / n * (len(futures) - n) / 60
            print(f"[{n}/{len(futures)}] {key}  {elapsed:.0f}s elapsed, ~{left:.0f} min left", flush=True)
