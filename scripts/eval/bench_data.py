"""Where the SWE-Explore data lives on disk and how it is read.

data/swe_explore/ (written by fetch_bench.py):
    bench.final.public.jsonl   the benchmark: 848 tasks with their ground truth
    tasks.json                 instance_id -> dataset, repo, base_commit, problem_statement
    git/<owner>__<repo>.git    bare repositories holding the base commits
    repos/<instance_id>/       checkout of base_commit (made on first use)
    potion-base-8M/            the embedding model of the potion baseline
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_DIR = ROOT / "eval" / "SWE-Explore-Bench"
DATA_DIR = ROOT / "data" / "swe_explore"
RESULTS_DIR = ROOT / "results" / "baselines"
BENCH_PATH = DATA_DIR / "bench.final.public.jsonl"
POTION_PATH = DATA_DIR / "potion-base-8M"


def use_official_code():
    """Make the official package (eval, explorers) importable."""
    if str(OFFICIAL_DIR) not in sys.path:
        sys.path.insert(0, str(OFFICIAL_DIR))


def git_dir(repo):
    return DATA_DIR / "git" / (repo.replace("/", "__") + ".git")


def snapshot_dir(iid):
    return DATA_DIR / "repos" / iid


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def load_tasks(datasets=None, limit=None):
    """Benchmark records (optionally only some parts / the first N) and tasks.json."""
    records = [r for r in read_jsonl(BENCH_PATH) if not datasets or r["dataset"] in datasets][:limit]
    tasks = json.loads((DATA_DIR / "tasks.json").read_text())
    missing = [r["instance_id"] for r in records if r["instance_id"] not in tasks]
    assert not missing, f"no metadata in tasks.json for {len(missing)} tasks, e.g. {missing[:3]}"
    return records, tasks


def load_done(path):
    """instance_ids already written to a results JSONL (re-running skips them)."""
    return {r["instance_id"] for r in read_jsonl(path)} if path.is_file() else set()


def make_snapshot(iid, task):
    """Check out base_commit into repos/<instance_id>; returns an error string or None."""
    dst = snapshot_dir(iid)
    if dst.is_dir():
        return None
    tmp = dst.with_name(dst.name + ".tmp")
    index = dst.with_name(dst.name + ".index")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    # Not `git archive`: it drops files marked export-ignore in .gitattributes.
    env = {**os.environ, "GIT_INDEX_FILE": str(index)}
    git = ["git", f"--git-dir={git_dir(task['repo'])}", f"--work-tree={tmp}"]
    error = None
    for args in (["read-tree", task["base_commit"]], ["checkout-index", "--all", "--force"]):
        proc = subprocess.run(git + args, env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            error = f"git {args[0]} failed: {proc.stderr.strip()}"
            break
    index.unlink(missing_ok=True)
    if error:
        shutil.rmtree(tmp, ignore_errors=True)
        return error
    os.replace(tmp, dst)
    return None


class Files:
    """Lazy line access to the files of one snapshot (repo_dir None = no snapshot)."""

    def __init__(self, repo_dir):
        self.repo_dir = repo_dir
        self._lines = {}

    def lines(self, path):
        if path not in self._lines:
            try:
                self._lines[path] = (self.repo_dir / path).read_text(errors="ignore").splitlines() if self.repo_dir else []
            except OSError:
                self._lines[path] = []
        return self._lines[path]

    def count(self, path):
        return len(self.lines(path))
