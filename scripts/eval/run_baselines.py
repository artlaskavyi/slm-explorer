"""Run the SWE-Explore baselines without an LLM and score them with the official evaluator.

Methods: the official explorers (random, oracle, bm25, tfidf, potion) imported
unchanged from eval/SWE-Explore-Bench, plus `mentioned` (files named in the
issue text). Differences from the official eval_runner.py:
  - a method that fails on a task scores an empty package (zeros), it is not skipped;
  - wall-clock time per task and package size are recorded;
  - the summary is split by benchmark part (verified, pro, multilingual) plus the
    overall mean over all tasks, which is what the paper reports.

Input is the directory written by fetch_bench.py. Repository snapshots are
created on first use with `git archive` from the bare repositories.
Per-task rows go to <out>/<method>.jsonl (re-running skips finished tasks);
the tables go to <out>/summary.md and <out>/summary.csv.
Run from the repository root:
    python scripts/eval/run_baselines.py
    python scripts/eval/run_baselines.py --methods bm25 mentioned --datasets verified --limit 20
"""

import argparse
import collections
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval" / "SWE-Explore-Bench"))
from eval import ExploreEvaluator  # noqa: E402

DATA_DIR = ROOT / "data" / "swe_explore"
OUT_DIR = ROOT / "results" / "baselines"
BENCH_FILE = "bench.final.public.jsonl"
TOP_K = [5, 10, 20]
LOCAL_METHODS = ["bm25", "tfidf", "potion", "mentioned"]  # need the repository snapshot
POOL_METHODS = ["random", "oracle"]  # need only the benchmark file
PARTS = ["verified", "pro", "multilingual", "all"]

# Same list as eval_runner.METRICS (importing it pulls typer and rich into the workers).
METRICS = [
    "precision", "recall", "f1_score",
    "hit_file_rate", "noise_file_rate", "hit_region_rate", "noise_region_rate",
    "weighted_core_coverage", "context_efficiency", "optional_coverage",
    "ndcg_at_100", "ndcg_at_300", "ndcg_at_500",
    "recall_at_100", "recall_at_300", "recall_at_500",
    "first_useful_hit",
]
# Columns of Table 6 in the paper, in its order, then our additions.
SUMMARY = [
    ("hit_region_rate", "HitReg"), ("precision", "Prec"), ("recall", "Rec"), ("f1_score", "F1"),
    ("hit_file_rate", "HitFile"), ("ndcg_at_500", "nDCG@500"), ("recall_at_500", "Rec@500"),
    ("first_useful_hit", "FUH"), ("context_efficiency", "CtxEff"), ("noise_region_rate", "NoiseReg"),
    ("modified_file_recall", "ModFile"), ("n_lines", "Lines"), ("seconds", "Sec"),
]


# ── repository snapshots ────────────────────────────────────────────────

def snapshot_dir(data_dir, iid):
    return data_dir / "repos" / iid


def make_snapshot(data_dir, iid, task):
    """git archive <base_commit> into repos/<instance_id>; returns an error string or None."""
    dst = snapshot_dir(data_dir, iid)
    if dst.is_dir():
        return None
    git_dir = data_dir / "git" / (task["repo"].replace("/", "__") + ".git")
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    archive = subprocess.Popen(["git", f"--git-dir={git_dir}", "archive", task["base_commit"]], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    untar = subprocess.run(["tar", "-x", "-C", str(tmp)], stdin=archive.stdout, capture_output=True)
    archive.stdout.close()
    err = archive.stderr.read().decode()
    if archive.wait() != 0 or untar.returncode != 0:
        shutil.rmtree(tmp, ignore_errors=True)
        return f"git archive failed: {err.strip() or untar.stderr.decode().strip()}"
    os.replace(tmp, dst)
    return None


def line_count(path):
    try:
        with open(path, "rb") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


def gt_line_counts(repo_dir, gt):
    """path -> number of lines for every ground-truth file (as eval_runner._build_file_line_counts)."""
    paths = {r["path"] for r in gt.get("read_core_regions") or []}
    for regions in (gt.get("read_optional_regions_map") or {}).values():
        paths.update(r["path"] for r in regions)
    return {p: n for p in paths if (n := line_count(repo_dir / p))}


# ── files mentioned in the issue ────────────────────────────────────────

SOURCE_EXTS = {
    ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".java", ".kt", ".scala", ".go", ".rs",
    ".c", ".h", ".cc", ".cpp", ".hpp", ".cs", ".rb", ".php", ".swift", ".m",
    ".toml", ".cfg", ".ini", ".yaml", ".yml", ".json", ".md", ".rst", ".txt",
}
SKIP_DIRS = {"node_modules", "vendor", "__pycache__", "build", "dist"}
# path with an optional line: foo/bar.py, foo/bar.py:12, .../blob/<sha>/foo/bar.py#L12
PATH_RE = re.compile(r"((?:[\w.-]+/)*[\w.-]+\.[A-Za-z]{1,5})(?::(\d+)|#L(\d+))?")
TRACE_RE = re.compile(r'File "([^"]+)", line (\d+)')
DOTTED_RE = re.compile(r"\b([a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*){1,})\b")
TRACE_WINDOW = 20


def list_source_files(repo_dir):
    files = []
    for dirpath, dirnames, filenames in os.walk(repo_dir):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith("."))
        rel_dir = Path(dirpath).relative_to(repo_dir)
        for name in sorted(filenames):
            if Path(name).suffix.lower() in SOURCE_EXTS:
                files.append((rel_dir / name).as_posix())
    return files


def mentioned_files(repo_dir, issue):
    """Files named in the issue, in order of appearance: stack-trace frames (a window
    around the line), paths (whole file, or a window if a line is given), dotted
    module names (whole file). Paths are matched by their longest existing suffix."""
    files = list_source_files(repo_dir)
    file_set = set(files)

    def resolve(cand):
        parts = [p for p in cand.split("/") if p not in ("", ".")]
        for i in range(len(parts)):
            sub = "/".join(parts[i:])
            if sub in file_set:
                return [sub]
            if i < len(parts) - 1:
                hits = [f for f in files if f.endswith("/" + sub)]
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
            hits = [f for f in files if f.endswith(stem + ".py") or f.endswith(stem + "/__init__.py")]
            if hits:
                found += [(h, None) for h in hits]
                break

    out, seen = [], set()
    for rel, line in found:
        n = max(1, line_count(repo_dir / rel))
        reg = (rel, max(1, line - TRACE_WINDOW), min(n, line + TRACE_WINDOW)) if line else (rel, 1, n)
        if reg not in seen:
            seen.add(reg)
            out.append(reg)
    return out


# ── methods ─────────────────────────────────────────────────────────────

_W = {}  # per-process state: evaluator, potion model


def init_worker(bench_path, potion_path):
    _W["evaluator"] = ExploreEvaluator(bench_path)
    _W["potion_path"] = potion_path
    _W["potion"] = None


def to_regions(results):
    return [(r.path, r.start, r.end) for res in results for r in res.regions]


def run_local(method, repo_dir, issue, iid, max_k):
    if method == "mentioned":
        return mentioned_files(repo_dir, issue)
    if method == "bm25":
        from explorers.bm25 import BM25Explorer
        explorer = BM25Explorer(repo_dir)
    elif method == "tfidf":
        from explorers.rag_tfidf import TFIDFExplorer
        explorer = TFIDFExplorer(repo_dir)
    elif method == "potion":
        from explorers.rag_potion import PotionExplorer, _load_potion_model
        if _W["potion"] is None:
            _W["potion"] = _load_potion_model(_W["potion_path"])
        explorer = PotionExplorer(repo_dir, model_path=_W["potion_path"], _model=_W["potion"])
    else:
        raise ValueError(method)
    return to_regions(explorer.explore(instance_id=iid, query=issue, top_k=max_k))


def score(iid, gt, preds, line_counts, repo_dir, top_k):
    """Official metrics for every K plus package size and recall of the files all agents modified."""
    ev = _W["evaluator"]
    modified = set(gt.get("modified_core_files") or [])
    out = {}
    for k in top_k:
        sliced = list(preds[:k])
        ev._current_instance_id = iid
        ev._current_file_line_counts = line_counts
        m = {name: getattr(ev, f"evaluate_{name}")(sliced, gt) for name in METRICS}
        files = {p for p, _, _ in sliced}
        m["modified_file_recall"] = len(files & modified) / len(modified) if modified else 0.0
        n_lines = 0
        for path, s, e in sliced:
            n = line_counts.get(path) or (line_count(repo_dir / path) if repo_dir else 0)
            end = n if e < 0 else min(e, n) if n else e  # end=-1 means end of file
            n_lines += max(0, end - max(1, s) + 1)
        m["n_lines"] = n_lines
        out[k] = m
    return out


def run_task(rec, task, methods, data_dir, top_k):
    """All repository-based methods for one task. Returns {method: row}."""
    iid, gt = rec["instance_id"], rec["ground_truth"]
    rows = {}
    err = make_snapshot(data_dir, iid, task)
    repo_dir = snapshot_dir(data_dir, iid)
    line_counts = gt_line_counts(repo_dir, gt) if err is None else {}
    for method in methods:
        t0 = time.perf_counter()
        preds, error = [], err
        if err is None:
            try:
                preds = run_local(method, repo_dir, task["problem_statement"], iid, max(top_k))
            except Exception:
                error = traceback.format_exc(limit=3)
        rows[method] = {
            "instance_id": iid,
            "dataset": rec["dataset"],
            "seconds": round(time.perf_counter() - t0, 3),
            "error": error,
            "regions": preds[: max(top_k)],
            "metrics": score(iid, gt, preds, line_counts, repo_dir if err is None else None, top_k),
        }
    return rows


# ── summary ─────────────────────────────────────────────────────────────

def summarize(out_dir, methods, top_k):
    rows = []  # (method, k, part, n, n_errors, {metric: mean})
    for method in methods:
        path = out_dir / f"{method}.jsonl"
        if not path.is_file():
            continue
        recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        for k in top_k:
            for part in PARTS:
                sub = [r for r in recs if part == "all" or r["dataset"] == part]
                if not sub:
                    continue
                means = {}
                for key, _ in SUMMARY:
                    vals = [r["seconds"] if key == "seconds" else r["metrics"][str(k)][key] for r in sub]
                    means[key] = sum(vals) / len(vals)
                rows.append((method, k, part, len(sub), sum(bool(r["error"]) for r in sub), means))

    with open(out_dir / "summary.csv", "w") as f:
        f.write("method,k,part,n,errors," + ",".join(key for key, _ in SUMMARY) + "\n")
        for method, k, part, n, n_err, means in rows:
            f.write(f"{method},{k},{part},{n},{n_err}," + ",".join(f"{means[key]:.4f}" for key, _ in SUMMARY) + "\n")

    lines = ["# SWE-Explore baselines", "",
             "Means over tasks. A failed task counts as an empty package. `all` is the overall",
             "mean over every task (the paper's numbers); parts are reported separately.", ""]
    head = "| Method | n | err | " + " | ".join(label for _, label in SUMMARY) + " |"
    for k in top_k:
        for part in PARTS:
            sel = [r for r in rows if r[1] == k and r[2] == part]
            if not sel:
                continue
            lines += [f"## K={k}, {part}", "", head, "|---" * (len(SUMMARY) + 3) + "|"]
            for method, _, _, n, n_err, means in sel:
                cells = [f"{means[key]:.0f}" if key == "n_lines" else f"{means[key]:.2f}" if key == "seconds" else f"{means[key]:.3f}" for key, _ in SUMMARY]
                lines.append(f"| {method} | {n} | {n_err} | " + " | ".join(cells) + " |")
            lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines))
    return "\n".join(lines)


# ── main ────────────────────────────────────────────────────────────────

def load_done(path):
    if not path.is_file():
        return set()
    return {json.loads(l)["instance_id"] for l in path.read_text().splitlines() if l.strip()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DATA_DIR, help="directory written by fetch_bench.py")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--methods", nargs="+", default=POOL_METHODS + LOCAL_METHODS)
    ap.add_argument("--top-k", nargs="+", type=int, default=TOP_K)
    ap.add_argument("--datasets", nargs="*", help="only these parts: verified pro multilingual")
    ap.add_argument("--limit", type=int, help="first N tasks (smoke test)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    unknown = set(a.methods) - set(POOL_METHODS + LOCAL_METHODS)
    assert not unknown, f"unknown methods: {unknown}"

    bench_path = a.data / BENCH_FILE
    all_records = [json.loads(l) for l in bench_path.read_text().splitlines() if l.strip()]
    tasks = json.loads((a.data / "tasks.json").read_text())
    records = [r for r in all_records if not a.datasets or r["dataset"] in a.datasets]
    if a.limit:
        records = records[: a.limit]
    selected = {r["instance_id"] for r in records}
    potion_path = str(a.data / "potion-base-8M")
    init_worker(bench_path, potion_path)
    print(f"Tasks: {len(records)}  methods: {a.methods}  K: {a.top_k}", flush=True)
    missing = [iid for iid in selected if iid not in tasks]
    assert not missing, f"no metadata in tasks.json for {len(missing)} tasks, e.g. {missing[:3]}"

    # Snapshots first: every method needs the file line counts for scoring.
    todo = [r["instance_id"] for r in records if not snapshot_dir(a.data, r["instance_id"]).is_dir()]
    if todo:
        t0 = time.time()
        with ThreadPoolExecutor(8) as pool:
            futures = {pool.submit(make_snapshot, a.data, iid, tasks[iid]): iid for iid in todo}
            for n, fut in enumerate(as_completed(futures), 1):
                if err := fut.result():
                    print(f"  snapshot {futures[fut]}: {err}", flush=True)
                if n % 50 == 0 or n == len(todo):
                    print(f"[{n}/{len(todo)}] snapshots  {time.time() - t0:.0f}s", flush=True)

    # random and oracle: the official explorers, in benchmark order (random keeps one
    # RNG with seed 42 across tasks, as in eval_runner.py).
    from explorers.baselines import OracleExplorer, RandomExplorer
    for method in [m for m in a.methods if m in POOL_METHODS]:
        sink_path = a.out / f"{method}.jsonl"
        if load_done(sink_path) >= selected:
            print(f"{method}: already done", flush=True)
            continue
        explorer = (RandomExplorer if method == "random" else OracleExplorer)(bench_path)
        with open(sink_path, "w") as sink:
            for rec in all_records:
                iid = rec["instance_id"]
                t0 = time.perf_counter()
                preds = to_regions(explorer.explore(instance_id=iid, query="", top_k=max(a.top_k)))
                if iid not in selected:
                    continue
                repo_dir = snapshot_dir(a.data, iid)
                repo_dir = repo_dir if repo_dir.is_dir() else None
                line_counts = gt_line_counts(repo_dir, rec["ground_truth"]) if repo_dir else {}
                row = {
                    "instance_id": iid, "dataset": rec["dataset"],
                    "seconds": round(time.perf_counter() - t0, 3), "error": None,
                    "regions": preds[: max(a.top_k)],
                    "metrics": score(iid, rec["ground_truth"], preds, line_counts, repo_dir, a.top_k),
                }
                sink.write(json.dumps(row) + "\n")
        print(f"{method}: {len(selected)} tasks", flush=True)

    # Repository-based methods: one task per worker process, all methods for that task.
    local = [m for m in a.methods if m in LOCAL_METHODS]
    if local:
        done = {m: load_done(a.out / f"{m}.jsonl") for m in local}
        todo = [r for r in records if any(r["instance_id"] not in done[m] for m in local)]
        print(f"Repository methods {local}: {len(todo)} tasks to run", flush=True)
        sinks = {m: open(a.out / f"{m}.jsonl", "a") for m in local}
        t0 = time.time()
        with ProcessPoolExecutor(a.workers, initializer=init_worker, initargs=(bench_path, potion_path)) as pool:
            futures = {
                pool.submit(run_task, r, tasks[r["instance_id"]], [m for m in local if r["instance_id"] not in done[m]], a.data, a.top_k): r["instance_id"]
                for r in todo
            }
            for n, fut in enumerate(as_completed(futures), 1):
                for method, row in fut.result().items():
                    sinks[method].write(json.dumps(row) + "\n")
                    sinks[method].flush()
                    if row["error"]:
                        print(f"  {method} failed on {row['instance_id']}: {row['error'].strip().splitlines()[-1]}", flush=True)
                elapsed = time.time() - t0
                eta = elapsed / n * (len(futures) - n)
                print(f"[{n}/{len(futures)}] {futures[fut]}  {elapsed:.0f}s elapsed, ~{eta / 60:.0f} min left", flush=True)
        for f in sinks.values():
            f.close()

    print(summarize(a.out, a.methods, a.top_k))


if __name__ == "__main__":
    main()
