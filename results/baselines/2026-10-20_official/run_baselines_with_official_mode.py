"""Run the SWE-Explore baselines without an LLM and score them with the official evaluator.

Two index modes:
  official  the explorers exactly as in eval/SWE-Explore-Bench: chunks of .py and docs
            only, at most 3,000 chunks per repository, BM25 split on whitespace,
            random = ground-truth regions sampled from the whole benchmark.
            Used only to reproduce Table 6 of the paper (K=5).
  full      every source file (SOURCE_EXTS, without vendor/build dirs and files over
            1 MB), no chunk limit, BM25 on identifier tokens (as TF-IDF), random =
            80-line windows from random source files of the repository.
Methods: random, oracle, bm25, tfidf, potion (official classes) and `mentioned`
(files named in the issue text).

Every package is cut in two ways: the first K regions (as in the paper) and a
budget of m x (lines of the core ground truth of the task), regions taken in rank
order until the budget is filled, the last one trimmed (C1, C1.5, C2, C3; C1 is the
main protocol, see bachelors/swe_explore_report.md, section 3). Fixed line budgets
(--budgets) are still available. Differences from the official eval_runner.py:
  - a method that fails on a task scores an empty package (zeros), it is not skipped;
  - indexing and search time per task, package size (unique lines, ~tokens) and
    index size are recorded;
  - the summary is split by benchmark part (verified, pro, multilingual) plus the
    overall mean over all tasks, which is what the paper reports.

Input is the directory written by fetch_bench.py. Repository snapshots are
created on first use with `git archive` from the bare repositories.
Each run goes to results/baselines/<run name>/: config.json, <method>.jsonl
(one row per task; re-running skips finished tasks), summary.md, summary.csv.
Run from the repository root:
    python scripts/eval/run_baselines.py --index full --run 2026-10-22_final
    python scripts/eval/run_baselines.py --index official --run 2026-10-20_official --core-multipliers
    python scripts/eval/run_baselines.py --methods bm25 mentioned --datasets verified --limit 20 --run smoke
"""

import argparse
import collections
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval" / "SWE-Explore-Bench"))
from eval import ExploreEvaluator  # noqa: E402

DATA_DIR = ROOT / "data" / "swe_explore"
RESULTS_DIR = ROOT / "results" / "baselines"
BENCH_FILE = "bench.final.public.jsonl"
TOP_K = [5, 10, 20]
# Budget per task = multiplier x lines of the core ground truth. 1 is the main protocol
# (budget = size of the answer, so precision = recall: line-level R-precision); 1.5-3 are
# the sensitivity check (bachelors/swe_explore_report.md, section 3).
CORE_MULTIPLIERS = [1, 1.5, 2, 3]
BUDGETS = []  # optional fixed budgets in unique lines (the 2026-10-20_full run used 500-5000)
MAX_REGIONS = 1000  # regions requested from a method; 3 x core reaches ~48k lines
SEED = 42
CHUNK_SIZE, CHUNK_OVERLAP = 80, 20  # as the official explorers
FULL_MAX_CHUNKS = 10**9
MAX_FILE_BYTES = 1_000_000
CHARS_PER_TOKEN = 4  # rough package size in tokens
PARTS = ["verified", "pro", "multilingual", "all"]
METHODS = ["random", "oracle", "bm25", "tfidf", "potion", "mentioned"]
# Methods that stop on their own: also scored on the whole package (cutoff `full`).
NATURAL_STOP = {"oracle", "mentioned"}
# Need only the benchmark file, run in the main process in benchmark order.
POOL_METHODS = {"official": {"random", "oracle"}, "full": {"oracle"}}

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
    ("modified_file_recall", "ModFile"), ("n_lines", "Lines"), ("n_tokens", "Tok"),
    ("index_seconds", "Index s"), ("search_seconds", "Search s"),
]
ROW_FIELDS = {"index_seconds", "search_seconds"}  # stored per task, not per cutoff


# ── repository snapshots ────────────────────────────────────────────────

def snapshot_dir(data_dir, iid):
    return data_dir / "repos" / iid


def make_snapshot(data_dir, iid, task):
    """Check out <base_commit> into repos/<instance_id>; returns an error string or None.

    read-tree + checkout-index with a private index file, not `git archive`: archive
    honours `export-ignore` in .gitattributes and silently drops e.g. the tests/ of
    some PHP repositories (77 tasks of Pro and Multilingual)."""
    dst = snapshot_dir(data_dir, iid)
    if dst.is_dir():
        return None
    git_dir = data_dir / "git" / (task["repo"].replace("/", "__") + ".git")
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    env = {**os.environ, "GIT_INDEX_FILE": str(tmp.with_name(tmp.name + ".index"))}
    git = ["git", f"--git-dir={git_dir}", f"--work-tree={tmp}"]
    for args in (["read-tree", task["base_commit"]], ["checkout-index", "--all", "--force"]):
        proc = subprocess.run(git + args, env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            shutil.rmtree(tmp, ignore_errors=True)
            Path(env["GIT_INDEX_FILE"]).unlink(missing_ok=True)
            return f"git {args[0]} failed: {proc.stderr.strip()}"
    Path(env["GIT_INDEX_FILE"]).unlink(missing_ok=True)
    os.replace(tmp, dst)
    return None


class Files:
    """Lazy line access to the files of one snapshot."""

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


def gt_line_counts(files, gt):
    """path -> number of lines for every ground-truth file (as eval_runner._build_file_line_counts)."""
    paths = {r["path"] for r in gt.get("read_core_regions") or []}
    for regions in (gt.get("read_optional_regions_map") or {}).values():
        paths.update(r["path"] for r in regions)
    return {p: n for p in paths if (n := files.count(p))}


# ── source files ────────────────────────────────────────────────────────

SOURCE_EXTS = {
    ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".java", ".kt", ".scala", ".go", ".rs",
    ".c", ".h", ".cc", ".cpp", ".hpp", ".cs", ".rb", ".php", ".swift", ".m",
    ".toml", ".cfg", ".ini", ".yaml", ".yml", ".json", ".md", ".rst", ".txt",
}
SKIP_DIRS = {"node_modules", "vendor", "__pycache__", "build", "dist"}


def list_source_files(repo_dir):
    """Repository-relative paths of source files, sorted (deterministic order)."""
    files = []
    for dirpath, dirnames, filenames in os.walk(repo_dir):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith("."))
        for name in sorted(filenames):
            p = os.path.join(dirpath, name)
            if os.path.splitext(name)[1].lower() in SOURCE_EXTS and os.path.isfile(p) and os.path.getsize(p) <= MAX_FILE_BYTES:
                files.append(Path(p).relative_to(repo_dir).as_posix())
    return files


def full_source_files(repo_root):
    """Replacement for explorers.chunking.iter_source_files in the full index."""
    for rel in list_source_files(repo_root):
        yield Path(repo_root) / rel


# ── files mentioned in the issue ────────────────────────────────────────

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


# ── methods ─────────────────────────────────────────────────────────────

_W = {}  # per-process state: evaluator, potion model, index mode


def init_worker(bench_path, potion_path, index):
    _W.update(evaluator=ExploreEvaluator(bench_path), potion_path=potion_path, potion=None, index=index)
    if index == "full":
        import explorers.chunking
        explorers.chunking.iter_source_files = full_source_files


def to_regions(results):
    return [(r.path, r.start, r.end) for res in results for r in res.regions]


IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]+")  # the token pattern of the official TF-IDF


def code_tokens(text):
    return [t.lower() for t in IDENT_RE.findall(text)]


class CodeBM25:
    """BM25 over the same chunks as the official explorer, on identifier tokens."""

    def __init__(self, repo_dir):
        from explorers.chunking import chunk_repo
        from rank_bm25 import BM25Okapi
        self._chunks = chunk_repo(repo_dir, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, max_chunks=FULL_MAX_CHUNKS)
        docs = [code_tokens(c.content) for c in self._chunks]
        self._bm25 = BM25Okapi(docs) if any(docs) else None

    def search(self, issue, n):
        query = code_tokens(issue)
        if self._bm25 is None or not query:
            return []
        scores = self._bm25.get_scores(query)
        top = np.argsort(-scores, kind="stable")[:n]
        return [(self._chunks[i].path, self._chunks[i].start, self._chunks[i].end) for i in top if scores[i] > 0]


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


def run_local(method, repo_dir, issue, iid, n, files):
    """Ranked regions of one method, the number of chunks it indexed (None if no index)
    and the seconds spent on indexing (chunking + index build) and on the search itself."""
    full = _W["index"] == "full"
    max_chunks = FULL_MAX_CHUNKS if full else 3000
    t0 = time.perf_counter()
    if method == "mentioned":
        return mentioned_files(repo_dir, issue, files), None, 0.0, time.perf_counter() - t0
    if method == "random":  # only in the full index; the official one is a pool method
        return random_windows(repo_dir, iid, n, files), None, 0.0, time.perf_counter() - t0
    if method == "potion" and _W["potion"] is None:  # model load is per process, not per task
        from explorers.rag_potion import _load_potion_model
        _W["potion"] = _load_potion_model(_W["potion_path"])
        t0 = time.perf_counter()
    if method == "bm25" and full:
        explorer = CodeBM25(repo_dir)
        t1 = time.perf_counter()
        return explorer.search(issue, n), len(explorer._chunks), t1 - t0, time.perf_counter() - t1
    if method == "bm25":
        from explorers.bm25 import BM25Explorer
        explorer = BM25Explorer(repo_dir, max_chunks=max_chunks)
    elif method == "tfidf":
        from explorers.rag_tfidf import TFIDFExplorer
        explorer = TFIDFExplorer(repo_dir, max_chunks=max_chunks)
    elif method == "potion":
        from explorers.rag_potion import PotionExplorer
        explorer = PotionExplorer(repo_dir, model_path=_W["potion_path"], _model=_W["potion"], max_chunks=max_chunks)
    else:
        raise ValueError(method)
    t1 = time.perf_counter()
    regions = to_regions(explorer.explore(instance_id=iid, query=issue, top_k=n))
    return regions, len(explorer._chunks), t1 - t0, time.perf_counter() - t1


# ── cutting and scoring ─────────────────────────────────────────────────

def cutoff_labels(top_k, multipliers, budgets):
    return [str(k) for k in top_k] + [f"C{m:g}" for m in multipliers] + [f"L{b}" for b in budgets]


def cut(preds, label, files, core_lines):
    """Cut a ranked package. Returns (regions kept, number of ranked regions consumed).
      `K`        first K regions;
      `C<m>`     regions in rank order until m x core_lines unique lines are filled;
      `L<n>`     the same with a fixed budget of n unique lines;
      `full`     the whole package (only for NATURAL_STOP methods).
    end=-1 means end of file; the region that crosses the budget is trimmed."""
    if label == "full":
        return list(preds), len(preds)
    if label[0] not in "CL":
        k = int(label)
        return list(preds[:k]), min(k, len(preds))
    budget = round(float(label[1:]) * core_lines) if label[0] == "C" else int(label[1:])
    used, out, consumed = 0, [], 0
    covered = collections.defaultdict(set)
    for path, s, e_orig in preds:
        if used >= budget:
            break
        consumed += 1
        n = files.count(path)
        e = n if e_orig < 0 else min(e_orig, n) if n else e_orig
        s = max(1, s)
        # Lines past the end of the file (a ground-truth artifact: the evaluator still counts
        # them) cost no budget, and the region keeps its recorded end.
        new = [x for x in range(s, e + 1) if x not in covered[path]]
        if used + len(new) > budget:
            new = new[: budget - used]
            out.append((path, s, new[-1]))
        elif new or (n and e_orig > n):
            out.append((path, s, e_orig))
        covered[path].update(new)
        used += len(new)
    return out, consumed


def package_size(regions, files):
    """Unique lines and approximate tokens of a package."""
    covered = collections.defaultdict(set)
    for path, s, e in regions:
        n = files.count(path)
        e = n if e < 0 else min(e, n) if n else e
        covered[path].update(range(max(1, s), e + 1))
    chars = 0
    for path, lines in covered.items():
        text = files.lines(path)
        chars += sum(len(text[i - 1]) + 1 for i in lines if i <= len(text))
    return sum(len(v) for v in covered.values()), chars // CHARS_PER_TOKEN


def core_lines(gt, files):
    """Unique lines of the core ground truth: the base of the C<m> budgets."""
    return package_size([(r["path"], r["start"], r["end"]) for r in gt.get("read_core_regions") or []], files)[0]


def score(iid, gt, preds, files, labels):
    """Official metrics for every cutoff plus package size and recall of the files all agents modified.
    Returns (metrics by cutoff, number of ranked regions the largest cutoff consumed)."""
    ev = _W["evaluator"]
    modified = set(gt.get("modified_core_files") or [])
    line_counts = gt_line_counts(files, gt)
    n_core = core_lines(gt, files)
    out, consumed_max = {}, 0
    for label in labels:
        sliced, consumed = cut(preds, label, files, n_core)
        consumed_max = max(consumed_max, consumed)
        ev._current_instance_id = iid
        ev._current_file_line_counts = line_counts
        m = {name: getattr(ev, f"evaluate_{name}")(sliced, gt) for name in METRICS}
        names = {p for p, _, _ in sliced}
        m["modified_file_recall"] = len(names & modified) / len(modified) if modified else 0.0
        m["n_lines"], m["n_tokens"] = package_size(sliced, files)
        m["n_regions"] = len(sliced)
        m["core_lines"] = n_core
        out[label] = m
    return out, consumed_max


def run_task(rec, task, methods, data_dir, labels, n_regions):
    """All repository-based methods for one task. Returns {method: row}."""
    iid, gt = rec["instance_id"], rec["ground_truth"]
    rows = {}
    err = make_snapshot(data_dir, iid, task)
    repo_dir = snapshot_dir(data_dir, iid)
    files = Files(repo_dir if err is None else None)
    for method in methods:
        preds, n_chunks, error, t_index, t_search = [], None, err, 0.0, 0.0
        if err is None:
            try:
                preds, n_chunks, t_index, t_search = run_local(method, repo_dir, task["problem_statement"], iid, n_regions, files)
            except Exception:
                error = traceback.format_exc(limit=3)
        metrics, consumed = score(iid, gt, preds, files, labels + (["full"] if method in NATURAL_STOP else []))
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


# ── summary ─────────────────────────────────────────────────────────────

def summarize(out_dir, methods, labels):
    rows = []  # (method, cutoff, part, n, n_errors, {metric: mean})
    for method in methods:
        path = out_dir / f"{method}.jsonl"
        if not path.is_file():
            continue
        recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        for label in labels + ["full"]:
            if not recs or label not in recs[0]["metrics"]:
                continue
            for part in PARTS:
                sub = [r for r in recs if part == "all" or r["dataset"] == part]
                if not sub:
                    continue
                means = {}
                for key, _ in SUMMARY:
                    vals = [r[key] if key in ROW_FIELDS else r["metrics"][label][key] for r in sub]
                    means[key] = sum(vals) / len(vals)
                rows.append((method, label, part, len(sub), sum(bool(r["error"]) for r in sub), means))

    with open(out_dir / "summary.csv", "w") as f:
        f.write("method,cutoff,part,n,errors," + ",".join(key for key, _ in SUMMARY) + "\n")
        for method, label, part, n, n_err, means in rows:
            f.write(f"{method},{label},{part},{n},{n_err}," + ",".join(f"{means[key]:.4f}" for key, _ in SUMMARY) + "\n")

    lines = ["# SWE-Explore baselines", "",
             "Means over tasks. A failed task counts as an empty package. Cutoff `K` = first K regions;",
             "`C<m>` = regions in rank order up to m x (lines of the core ground truth of the task);",
             "`L<n>` = up to n unique lines; `full` = the whole package of a method that stops on its own.",
             "Part `all` is the overall mean over every task (the paper's numbers); parts are reported",
             "separately. Run parameters: config.json.", ""]
    head = "| Method | n | err | " + " | ".join(label for _, label in SUMMARY) + " |"
    titles = {"C": "budget {} x core lines", "L": "budget {} lines"}
    for label in labels + ["full"]:
        for part in PARTS:
            sel = [r for r in rows if r[1] == label and r[2] == part]
            if not sel:
                continue
            title = "whole package" if label == "full" else titles[label[0]].format(label[1:]) if label[0] in titles else f"K={label}"
            lines += [f"## {title}, {part}", "", head, "|---" * (len(SUMMARY) + 3) + "|"]
            for method, _, _, n, n_err, means in sel:
                cells = [f"{means[key]:.0f}" if key in ("n_lines", "n_tokens") else f"{means[key]:.2f}" if key in ROW_FIELDS else f"{means[key]:.3f}" for key, _ in SUMMARY]
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
    ap.add_argument("--run", required=True, help="run name: results go to results/baselines/<run>/")
    ap.add_argument("--index", choices=["official", "full"], default="full")
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--top-k", nargs="*", type=int, default=TOP_K)
    ap.add_argument("--core-multipliers", nargs="*", type=float, default=CORE_MULTIPLIERS, help="budgets as multiples of the core lines")
    ap.add_argument("--budgets", nargs="*", type=int, default=BUDGETS, help="fixed line budgets; empty = none")
    ap.add_argument("--datasets", nargs="*", help="only these parts: verified pro multilingual")
    ap.add_argument("--limit", type=int, help="first N tasks (smoke test)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    a = ap.parse_args()
    out = RESULTS_DIR / a.run
    out.mkdir(parents=True, exist_ok=True)
    assert not set(a.methods) - set(METHODS), f"unknown methods: {set(a.methods) - set(METHODS)}"
    labels = cutoff_labels(a.top_k, a.core_multipliers, a.budgets)
    n_regions = max(a.top_k + ([MAX_REGIONS] if a.budgets or a.core_multipliers else []))

    bench_path = a.data / BENCH_FILE
    all_records = [json.loads(l) for l in bench_path.read_text().splitlines() if l.strip()]
    tasks = json.loads((a.data / "tasks.json").read_text())
    records = [r for r in all_records if not a.datasets or r["dataset"] in a.datasets]
    if a.limit:
        records = records[: a.limit]
    selected = {r["instance_id"] for r in records}
    missing = [iid for iid in selected if iid not in tasks]
    assert not missing, f"no metadata in tasks.json for {len(missing)} tasks, e.g. {missing[:3]}"
    potion_path = str(a.data / "potion-base-8M")
    init_worker(bench_path, potion_path, a.index)

    config = {
        "index": a.index, "methods": a.methods, "top_k": a.top_k,
        "core_multipliers": a.core_multipliers, "line_budgets": a.budgets,
        "regions_requested": n_regions, "tasks": len(records), "datasets": a.datasets or "all",
        "chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP,
        "max_chunks": FULL_MAX_CHUNKS if a.index == "full" else 3000,
        "indexed_files": sorted(SOURCE_EXTS) if a.index == "full" else "official chunking.EXTS (.py and docs)",
        "skip_dirs": sorted(SKIP_DIRS) if a.index == "full" else None,
        "max_file_bytes": MAX_FILE_BYTES if a.index == "full" else None,
        "bm25_tokens": "identifiers, lowercase" if a.index == "full" else "whitespace (official)",
        "random": f"{CHUNK_SIZE}-line windows from random source files, seed {SEED}:<instance_id>" if a.index == "full"
                  else "official RandomExplorer: ground-truth regions of the whole benchmark, seed 42",
        "issue_text": "problem_statement from SWE-bench Verified / Pro / Multilingual",
    }
    (out / "config.json").write_text(json.dumps(config, indent=1))
    print(f"Run {a.run}: {len(records)} tasks, index {a.index}, methods {a.methods}, cutoffs {labels}", flush=True)

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

    # Pool methods (oracle; random in the official index): the official explorers, in
    # benchmark order (the official random keeps one RNG with seed 42 across tasks).
    from explorers.baselines import OracleExplorer, RandomExplorer
    for method in [m for m in a.methods if m in POOL_METHODS[a.index]]:
        sink_path = out / f"{method}.jsonl"
        if load_done(sink_path) >= selected:
            print(f"{method}: already done", flush=True)
            continue
        explorer = (RandomExplorer if method == "random" else OracleExplorer)(bench_path)
        with open(sink_path, "w") as sink:
            for rec in all_records:
                iid = rec["instance_id"]
                t0 = time.perf_counter()
                preds = to_regions(explorer.explore(instance_id=iid, query="", top_k=n_regions))
                if iid not in selected:
                    continue
                t_search = time.perf_counter() - t0
                repo_dir = snapshot_dir(a.data, iid)
                metrics, consumed = score(iid, rec["ground_truth"], preds, Files(repo_dir if repo_dir.is_dir() else None),
                                          labels + (["full"] if method in NATURAL_STOP else []))
                row = {
                    "instance_id": iid, "dataset": rec["dataset"],
                    "index_seconds": 0.0, "search_seconds": round(t_search, 3), "error": None, "index_chunks": None,
                    "regions": preds[:consumed],
                    "metrics": metrics,
                }
                sink.write(json.dumps(row) + "\n")
        print(f"{method}: {len(selected)} tasks", flush=True)

    # Repository-based methods: one task per worker process, all methods for that task.
    local = [m for m in a.methods if m not in POOL_METHODS[a.index]]
    if local:
        done = {m: load_done(out / f"{m}.jsonl") for m in local}
        todo = [r for r in records if any(r["instance_id"] not in done[m] for m in local)]
        print(f"Repository methods {local}: {len(todo)} tasks to run", flush=True)
        sinks = {m: open(out / f"{m}.jsonl", "a") for m in local}
        t0 = time.time()
        with ProcessPoolExecutor(a.workers, initializer=init_worker, initargs=(bench_path, potion_path, a.index)) as pool:
            futures = {
                pool.submit(run_task, r, tasks[r["instance_id"]], [m for m in local if r["instance_id"] not in done[m]], a.data, labels, n_regions): r["instance_id"]
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

    print(summarize(out, a.methods, labels))


if __name__ == "__main__":
    main()
