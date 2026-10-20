"""Download everything the SWE-Explore baselines need (run on the machine with fast internet).

1. The benchmark file (848 tasks) from the HF Hub.
2. Task metadata (repo, base_commit, problem_statement) from SWE-bench Verified,
   SWE-bench Pro and SWE-bench Multilingual -> tasks.json.
3. One bare git repository per benchmark repository (64), holding only the
   base commits the tasks need (shallow fetch by SHA, depth 1).
4. The Potion static embedding model for the potion baseline.

The output directory is self-contained: copy it to the other machine as is
(rsync over Thunderbolt), then run run_baselines.py there. Re-running the script
skips what is already downloaded. Run from the repository root:
    python scripts/eval/fetch_bench.py
    python scripts/eval/fetch_bench.py --only-repos psf/requests   # smoke test
"""

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download, snapshot_download

OUTPUT_DIR = Path("data/swe_explore")
BENCH_REPO = "SWE-Explore-Bench/SWE-Explore-Bench"
BENCH_FILE = "bench.final.public.jsonl"
# (HF dataset, parquet file); Pro has two configs, both are read and merged.
SOURCES = {
    "verified": [("princeton-nlp/SWE-bench_Verified", "data/test-00000-of-00001.parquet")],
    "multilingual": [("SWE-bench/SWE-bench_Multilingual", "data/test-00000-of-00001.parquet")],
    "pro": [
        ("ScaleAI/SWE-bench_Pro", "data/default/test-00000-of-00001.parquet"),
        ("ScaleAI/SWE-bench_Pro", "data/v1/test-00000-of-00001.parquet"),
    ],
}
POTION_MODEL = "minishlab/potion-base-8M"
N_WORKERS = 6
SHAS_PER_FETCH = 50


def git(args, cwd=None):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)


def load_tasks(bench_path, raw_dir):
    """instance_id -> {dataset, repo, base_commit, problem_statement} for every benchmark task."""
    meta = {}
    for dataset, files in SOURCES.items():
        for hf_repo, filename in files:
            path = hf_hub_download(hf_repo, filename, repo_type="dataset", local_dir=raw_dir / hf_repo.replace("/", "__"))
            df = pd.read_parquet(path, columns=["instance_id", "repo", "base_commit", "problem_statement"])
            for row in df.itertuples(index=False):
                iid = row.instance_id.removeprefix("instance_")  # Pro ids carry an "instance_" prefix
                meta.setdefault((dataset, iid), row)

    tasks, missing = {}, []
    with open(bench_path) as f:
        for line in f:
            rec = json.loads(line)
            iid, dataset = rec["instance_id"], rec["dataset"]
            row = meta.get((dataset, iid))
            if row is None:
                missing.append(iid)
                continue
            tasks[iid] = {
                "dataset": dataset,
                "repo": row.repo,
                "base_commit": row.base_commit,
                "problem_statement": row.problem_statement,
            }
    return tasks, missing


def fetch_repo(repo, shas, git_dir):
    """Shallow-fetch the given commits into a bare repository; return the SHAs still missing."""
    if not git_dir.exists():
        git(["init", "--bare", "-q", str(git_dir)])
    need = [s for s in shas if git(["cat-file", "-e", f"{s}^{{commit}}"], cwd=git_dir).returncode != 0]
    url = f"https://github.com/{repo}.git"
    for i in range(0, len(need), SHAS_PER_FETCH):
        batch = need[i : i + SHAS_PER_FETCH]
        for attempt in range(3):
            if git(["fetch", "-q", "--depth", "1", url, *batch], cwd=git_dir).returncode == 0:
                break
            time.sleep(5 * (attempt + 1))
    return [s for s in shas if git(["cat-file", "-e", f"{s}^{{commit}}"], cwd=git_dir).returncode != 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUTPUT_DIR)
    ap.add_argument("--only-repos", nargs="*", help="fetch only these repositories (owner/name)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    bench_path = Path(hf_hub_download(BENCH_REPO, BENCH_FILE, repo_type="dataset", local_dir=a.out))
    tasks, missing = load_tasks(bench_path, a.out / "hf_raw")
    with open(a.out / "tasks.json", "w") as f:
        json.dump(tasks, f, indent=1, ensure_ascii=False)
    print(f"tasks.json: {len(tasks)} tasks, {len(missing)} without metadata", flush=True)
    for iid in missing:
        print(f"  no metadata: {iid}", flush=True)

    snapshot_download(POTION_MODEL, local_dir=a.out / "potion-base-8M")
    print(f"Potion model: {a.out / 'potion-base-8M'}", flush=True)

    by_repo = defaultdict(set)
    for t in tasks.values():
        by_repo[t["repo"]].add(t["base_commit"])
    if a.only_repos:
        by_repo = {r: s for r, s in by_repo.items() if r in a.only_repos}
    git_root = a.out / "git"
    git_root.mkdir(exist_ok=True)

    # Repositories with many commits first: they take longest.
    order = sorted(by_repo, key=lambda r: -len(by_repo[r]))
    failed, t0 = {}, time.time()
    with ThreadPoolExecutor(N_WORKERS) as pool:
        futures = {
            pool.submit(fetch_repo, r, sorted(by_repo[r]), git_root / (r.replace("/", "__") + ".git")): r
            for r in order
        }
        for n, fut in enumerate(as_completed(futures), 1):
            repo = futures[fut]
            still_missing = fut.result()
            if still_missing:
                failed[repo] = still_missing
            status = f"{len(still_missing)} commits missing" if still_missing else "ok"
            print(f"[{n}/{len(futures)}] {repo}: {len(by_repo[repo])} commits, {status}  {time.time() - t0:.0f}s", flush=True)

    with open(a.out / "fetch_failed.json", "w") as f:
        json.dump(failed, f, indent=1)
    n_bad = sum(len(v) for v in failed.values())
    print(f"Done: {len(by_repo)} repositories, {n_bad} commits not fetched (see fetch_failed.json)")
    print(f"Size: {subprocess.run(['du', '-sh', str(a.out)], capture_output=True, text=True).stdout.strip()}")
    sys.exit(1 if missing or n_bad else 0)


if __name__ == "__main__":
    main()
