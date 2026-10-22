"""Run the SWE-Explore baselines without an LLM; scoring comes from bench.py.

Methods, over every source file of the repository snapshot (SOURCE_EXTS, without
vendor/build dirs and files over 1 MB), in 80-line chunks with 20 lines of overlap
and no chunk limit:
  bm25       BM25 on identifier tokens (the token pattern of the official TF-IDF)
  tfidf      official TFIDFExplorer
  potion     official PotionExplorer (static embeddings, potion-base-8M)
  random     80-line windows from random source files, seed per task
  mentioned  files named in the issue text (paths, stack traces, dotted modules)
  oracle     the core ground truth
How they differ from the paper: bachelors/swe_explore_report.md, section 4. The run
that reproduces the paper's settings (results/baselines/2026-10-20_official) was made
with the copy of this script saved next to its results.

Each run goes to results/baselines/<run>/: config.json, <method>.jsonl (one row per
task; re-running skips finished tasks), summary.md, summary.csv. From the repository root:
    python scripts/eval/run_baselines.py --run 2026-10-22_final
    python scripts/eval/run_baselines.py --run smoke --methods bm25 mentioned --datasets verified --limit 20
"""

import argparse
import functools
import json
import os
import random
import re
import time
import traceback
from pathlib import Path

import numpy as np

import bench
import bench_data
from parallel import run_parallel
from summarize import summarize

bench_data.use_official_code()
import explorers.chunking  # noqa: E402
from explorers.base import ContextRegion, ExplorerResult  # noqa: E402

METHODS = ["random", "oracle", "bm25", "tfidf", "potion", "mentioned"]
NATURAL_STOP = {"oracle", "mentioned"}  # stop on their own: also scored on the whole package
MAX_REGIONS = 1000  # regions requested from a ranking method; 3 x core reaches ~48k lines
SEED = 42
CHUNK_SIZE, CHUNK_OVERLAP = 80, 20  # as the official explorers
NO_CHUNK_LIMIT = 10**9
SOURCE_EXTS = {
    ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".java", ".kt", ".scala", ".go", ".rs",
    ".c", ".h", ".cc", ".cpp", ".hpp", ".cs", ".rb", ".php", ".swift", ".m",
    ".toml", ".cfg", ".ini", ".yaml", ".yml", ".json", ".md", ".rst", ".txt",
}
SKIP_DIRS = {"node_modules", "vendor", "__pycache__", "build", "dist"}
MAX_FILE_BYTES = 1_000_000


@functools.lru_cache(maxsize=4)
def list_source_files(repo_dir):
    """Repository-relative paths of source files, sorted (deterministic order)."""
    files = []
    for dirpath, dirnames, filenames in os.walk(repo_dir):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith("."))
        for name in sorted(filenames):
            p = os.path.join(dirpath, name)
            if os.path.splitext(name)[1].lower() in SOURCE_EXTS and os.path.isfile(p) and os.path.getsize(p) <= MAX_FILE_BYTES:
                files.append(Path(p).relative_to(repo_dir).as_posix())
    return tuple(files)


def source_files_for_chunking(repo_root):
    for rel in list_source_files(Path(repo_root)):
        yield Path(repo_root) / rel


# The official chunker indexes only .py and docs. Worker processes import this module
# too, so the replacement applies there as well.
explorers.chunking.iter_source_files = source_files_for_chunking


# path with an optional line: foo/bar.py, foo/bar.py:12, .../blob/<sha>/foo/bar.py#L12
PATH_RE = re.compile(r"((?:[\w.-]+/)*[\w.-]+\.[A-Za-z]{1,5})(?::(\d+)|#L(\d+))?")
TRACE_RE = re.compile(r'File "([^"]+)", line (\d+)')
DOTTED_RE = re.compile(r"\b([a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*){1,})\b")
TRACE_WINDOW = 20


def mentioned_files(repo_dir, issue, files):
    """Files named in the issue, in order of appearance: stack-trace frames (a window
    around the line), paths (whole file, or a window if a line is given), dotted
    module names (whole file). Paths are matched by their longest existing suffix."""
    paths = list_source_files(repo_dir)
    path_set = set(paths)

    def resolve(cand):
        parts = [p for p in cand.split("/") if p not in ("", ".")]
        for i in range(len(parts)):
            sub = "/".join(parts[i:])
            if sub in path_set:
                return [sub]
            if i < len(parts) - 1:
                hits = [f for f in paths if f.endswith("/" + sub)]
                if hits:
                    return hits
        return []

    found = []  # (path, line or None)
    for path, line in TRACE_RE.findall(issue):
        found += [(rel, int(line)) for rel in resolve(path)]
    for path, line, anchor in PATH_RE.findall(issue):
        ln = line or anchor
        found += [(rel, int(ln) if ln else None) for rel in resolve(path)]
    for dotted in DOTTED_RE.findall(issue):  # flask.blueprints -> flask/blueprints.py
        parts = dotted.split(".")
        for n in range(len(parts), 1, -1):
            stem = "/".join(parts[:n])
            hits = [f for f in paths if f.endswith(stem + ".py") or f.endswith(stem + "/__init__.py")]
            if hits:
                found += [(h, None) for h in hits]
                break

    out, seen = [], set()
    for rel, line in found:
        n = max(1, files.count(rel))
        reg = (rel, max(1, line - TRACE_WINDOW), min(n, line + TRACE_WINDOW)) if line else (rel, 1, n)
        if reg not in seen:
            seen.add(reg)
            out.append(reg)
    return out


def random_windows(repo_dir, iid, n, files):
    """n windows of CHUNK_SIZE lines: a random source file, then a random start in it."""
    paths = list_source_files(repo_dir)
    rng = random.Random(f"{SEED}:{iid}")
    out, seen = [], set()
    for _ in range(n * 20):
        if not paths or len(out) >= n:
            break
        path = rng.choice(paths)
        n_lines = files.count(path)
        if not n_lines:
            continue
        start = rng.randint(1, max(1, n_lines - CHUNK_SIZE + 1))
        if (path, start) not in seen:
            seen.add((path, start))
            out.append((path, start, min(n_lines, start + CHUNK_SIZE - 1)))
    return out


IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]+")  # the token pattern of the official TF-IDF


def code_tokens(text):
    return [t.lower() for t in IDENT_RE.findall(text)]


class CodeBM25:
    """BM25 over the same chunks as the official explorers, on identifier tokens
    (the official BM25Explorer splits on whitespace). Same interface as theirs."""

    def __init__(self, repo_dir):
        from rank_bm25 import BM25Okapi
        self._chunks = explorers.chunking.chunk_repo(repo_dir, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, max_chunks=NO_CHUNK_LIMIT)
        docs = [code_tokens(c.content) for c in self._chunks]
        self._bm25 = BM25Okapi(docs) if any(docs) else None

    def explore(self, *, instance_id, query, top_k):
        tokens = code_tokens(query)
        if self._bm25 is None or not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        top = [i for i in np.argsort(-scores, kind="stable")[:top_k] if scores[i] > 0]
        return [
            ExplorerResult(instance_id, float(scores[i]), [ContextRegion(self._chunks[i].path, self._chunks[i].start, self._chunks[i].end)])
            for i in top
        ]


@functools.cache
def potion_model():
    from explorers.rag_potion import _load_potion_model
    return _load_potion_model(str(bench_data.POTION_PATH))


def build_index(method, repo_dir):
    if method == "bm25":
        return CodeBM25(repo_dir)
    if method == "tfidf":
        from explorers.rag_tfidf import TFIDFExplorer
        return TFIDFExplorer(repo_dir, max_chunks=NO_CHUNK_LIMIT)
    if method == "potion":
        from explorers.rag_potion import PotionExplorer
        return PotionExplorer(repo_dir, model_path=str(bench_data.POTION_PATH), _model=potion_model(), max_chunks=NO_CHUNK_LIMIT)
    raise ValueError(method)


def run_method(method, rec, issue, repo_dir, files):
    """Ranked regions, number of indexed chunks (None without an index), seconds spent
    on indexing (chunking + index build) and on the search itself."""
    iid = rec["instance_id"]
    if method == "potion":
        potion_model()  # loaded once per process, not counted as indexing
    t0 = time.perf_counter()
    if method == "oracle":
        return bench.core_regions(rec["ground_truth"]), None, 0.0, time.perf_counter() - t0
    if method == "mentioned":
        return mentioned_files(repo_dir, issue, files), None, 0.0, time.perf_counter() - t0
    if method == "random":
        return random_windows(repo_dir, iid, MAX_REGIONS, files), None, 0.0, time.perf_counter() - t0
    explorer = build_index(method, repo_dir)
    t1 = time.perf_counter()
    results = explorer.explore(instance_id=iid, query=issue, top_k=MAX_REGIONS)
    regions = [(r.path, r.start, r.end) for res in results for r in res.regions]
    return regions, len(explorer._chunks), t1 - t0, time.perf_counter() - t1


def run_task(rec, task, methods):
    """All methods for one task. Returns {method: row}."""
    iid = rec["instance_id"]
    err = bench_data.make_snapshot(iid, task)
    repo_dir = bench_data.snapshot_dir(iid)
    files = bench_data.Files(repo_dir if err is None else None)
    rows = {}
    for method in methods:
        preds, n_chunks, error, t_index, t_search = [], None, err, 0.0, 0.0
        if err is None:
            try:
                preds, n_chunks, t_index, t_search = run_method(method, rec, task["problem_statement"], repo_dir, files)
            except Exception:
                error = traceback.format_exc(limit=3)
        metrics, consumed = bench.score(rec, preds, files, bench.cutoff_labels(method in NATURAL_STOP))
        rows[method] = {
            "instance_id": iid,
            "dataset": rec["dataset"],
            "index_seconds": round(t_index, 3),
            "search_seconds": round(t_search, 3),
            "error": error,
            "index_chunks": n_chunks,
            "regions": preds[:consumed],  # as many as the largest cutoff used
            "metrics": metrics,
        }
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="results go to results/baselines/<run>/")
    ap.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    ap.add_argument("--datasets", nargs="*", help="only these parts: verified pro multilingual")
    ap.add_argument("--limit", type=int, help="first N tasks (smoke test)")
    a = ap.parse_args()
    out = bench_data.RESULTS_DIR / a.run
    out.mkdir(parents=True, exist_ok=True)
    records, tasks = bench_data.load_tasks(a.datasets, a.limit)

    config = {
        "methods": a.methods, "tasks": len(records), "datasets": a.datasets or "all",
        "cutoffs": bench.cutoff_labels(True), "regions_requested": MAX_REGIONS,
        "chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP, "max_chunks": None,
        "indexed_files": sorted(SOURCE_EXTS), "skip_dirs": sorted(SKIP_DIRS), "max_file_bytes": MAX_FILE_BYTES,
        "bm25_tokens": "identifiers, lowercase",
        "random": f"{CHUNK_SIZE}-line windows from random source files, seed {SEED}:<instance_id>",
        "issue_text": "problem_statement from SWE-bench Verified / Pro / Multilingual",
        "snapshots": "git read-tree + checkout-index of base_commit (export-ignore not applied)",
    }
    (out / "config.json").write_text(json.dumps(config, indent=1))

    done = {m: bench_data.load_done(out / f"{m}.jsonl") for m in a.methods}
    jobs = {}
    for r in records:
        left = [m for m in a.methods if r["instance_id"] not in done[m]]
        if left:
            jobs[r["instance_id"]] = (r, tasks[r["instance_id"]], left)
    print(f"Run {a.run}: {len(records)} tasks, {len(jobs)} to run, methods {a.methods}", flush=True)

    sinks = {m: open(out / f"{m}.jsonl", "a") for m in a.methods}
    for _, rows in run_parallel(run_task, jobs, max(1, (os.cpu_count() or 4) - 2), processes=True):
        for method, row in rows.items():
            sinks[method].write(json.dumps(row) + "\n")
            sinks[method].flush()
            if row["error"]:
                print(f"  {method} failed on {row['instance_id']}: {row['error'].strip().splitlines()[-1]}", flush=True)
    for f in sinks.values():
        f.close()
    print(summarize(out))


if __name__ == "__main__":
    main()
