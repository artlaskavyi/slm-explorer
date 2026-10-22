"""Download everything the SWE-Explore baselines need into data/swe_explore/.

1. The benchmark file (848 tasks) from the HF Hub.
2. Task metadata (repo, base_commit, problem_statement) from SWE-bench Verified,
   SWE-bench Pro and SWE-bench Multilingual -> tasks.json.
3. One bare git repository per benchmark repository (64), holding only the
   base commits the tasks need (shallow fetch by SHA, depth 1).
4. The Potion static embedding model for the potion baseline.

Re-running skips what is already downloaded. From the repository root:
    python scripts/eval/fetch_bench.py
    python scripts/eval/fetch_bench.py --only-repos psf/requests   # smoke test
"""

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict

import pandas as pd
from huggingface_hub import hf_hub_download, snapshot_download

from bench_data import BENCH_PATH, DATA_DIR, POTION_PATH, git_dir, read_jsonl
from parallel import run_parallel

BENCH_REPO = "SWE-Explore-Bench/SWE-Explore-Bench"
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


def load_metadata():
    """instance_id -> {dataset, repo, base_commit, problem_statement} for every benchmark
    task, and the ids without metadata."""
    meta = {}
    for dataset, files in SOURCES.items():
        for hf_repo, filename in files:
            path = hf_hub_download(hf_repo, filename, repo_type="dataset", local_dir=DATA_DIR / "hf_raw" / hf_repo.replace("/", "__"))
            df = pd.read_parquet(path, columns=["instance_id", "repo", "base_commit", "problem_statement"])
            for row in df.itertuples(index=False):
                iid = row.instance_id.removeprefix("instance_")  # Pro ids carry an "instance_" prefix
                meta.setdefault((dataset, iid), row)

    tasks, missing = {}, []
    for rec in read_jsonl(BENCH_PATH):
        iid, dataset = rec["instance_id"], rec["dataset"]
        row = meta.get((dataset, iid))
        if row is None:
            missing.append(iid)
            continue
        tasks[iid] = {"dataset": dataset, "repo": row.repo, "base_commit": row.base_commit, "problem_statement": row.problem_statement}
    return tasks, missing


def fetch_repo(repo, shas):
    """Shallow-fetch the given commits into the bare repository; return the SHAs still missing."""
    path = git_dir(repo)
    if not path.exists():
        git(["init", "--bare", "-q", str(path)])

    def missing():
        return [s for s in shas if git(["cat-file", "-e", f"{s}^{{commit}}"], cwd=path).returncode != 0]

    url = f"https://github.com/{repo}.git"
    need = missing()
    for i in range(0, len(need), SHAS_PER_FETCH):
        batch = need[i : i + SHAS_PER_FETCH]
        for attempt in range(3):
            if git(["fetch", "-q", "--depth", "1", url, *batch], cwd=path).returncode == 0:
                break
            time.sleep(5 * (attempt + 1))
    return missing()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-repos", nargs="*", help="fetch only these repositories (owner/name)")
    a = ap.parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    hf_hub_download(BENCH_REPO, BENCH_PATH.name, repo_type="dataset", local_dir=DATA_DIR)
    tasks, missing = load_metadata()
    (DATA_DIR / "tasks.json").write_text(json.dumps(tasks, indent=1, ensure_ascii=False))
    print(f"tasks.json: {len(tasks)} tasks, {len(missing)} without metadata", flush=True)
    for iid in missing:
        print(f"  no metadata: {iid}", flush=True)

    snapshot_download(POTION_MODEL, local_dir=POTION_PATH)

    by_repo = defaultdict(set)
    for t in tasks.values():
        by_repo[t["repo"]].add(t["base_commit"])
    if a.only_repos:
        by_repo = {r: s for r, s in by_repo.items() if r in a.only_repos}
    (DATA_DIR / "git").mkdir(exist_ok=True)

    # Repositories with many commits first: they take longest.
    jobs = {r: (r, sorted(by_repo[r])) for r in sorted(by_repo, key=lambda r: -len(by_repo[r]))}
    failed = {}
    for repo, still_missing in run_parallel(fetch_repo, jobs, N_WORKERS):
        if still_missing:
            failed[repo] = still_missing
            print(f"  {repo}: {len(still_missing)} of {len(by_repo[repo])} commits missing", flush=True)

    (DATA_DIR / "fetch_failed.json").write_text(json.dumps(failed, indent=1))
    n_bad = sum(len(v) for v in failed.values())
    print(f"Done: {len(by_repo)} repositories, {n_bad} commits not fetched (see fetch_failed.json)")
    sys.exit(1 if missing or n_bad else 0)


if __name__ == "__main__":
    main()
