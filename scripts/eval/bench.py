"""SWE-Explore evaluation protocol: package cutoffs and scoring with the official evaluator.

Used by run_baselines.py (methods without an LLM) and run_agent.py (LLM explorers).
Every method writes one JSONL row per task: instance_id, dataset, error, regions
(ranked [path, start, end]), metrics (by cutoff) and its own timing fields.
Protocol: bachelors/swe_explore_report.md, section 3.
"""

import collections
import functools

import bench_data

TOP_K = [5, 10, 20]
# Budget per task = multiplier x lines of the core ground truth. 1 is the main protocol
# (budget = size of the answer, so precision = recall: line-level R-precision); 1.5-3 are
# the sensitivity check.
CORE_MULTIPLIERS = [1, 1.5, 2, 3]
CHARS_PER_TOKEN = 4  # rough package size in tokens

# Same list as eval_runner.METRICS (importing it pulls typer and rich into the workers).
METRICS = [
    "precision", "recall", "f1_score",
    "hit_file_rate", "noise_file_rate", "hit_region_rate", "noise_region_rate",
    "weighted_core_coverage", "context_efficiency", "optional_coverage",
    "ndcg_at_100", "ndcg_at_300", "ndcg_at_500",
    "recall_at_100", "recall_at_300", "recall_at_500",
    "first_useful_hit",
]


def cutoff_labels(full_package=False):
    """`K` = first K regions; `C<m>` = m x core lines; `full` = the whole package, only for
    methods that stop on their own (LLM explorers, files named in the issue, oracle)."""
    return [str(k) for k in TOP_K] + [f"C{m:g}" for m in CORE_MULTIPLIERS] + (["full"] if full_package else [])


def parse_cutoff(label):
    """'5' -> ('k', 5), 'C1.5' -> ('core', 1.5), 'full' -> ('full', None)."""
    if label == "full":
        return "full", None
    if label.startswith("C"):
        return "core", float(label[1:])
    return "k", int(label)


def clip_end(end, n_lines):
    """Region end within the file: -1 means end of file; unknown length (0) keeps the end."""
    if end < 0:
        return n_lines
    return min(end, n_lines) if n_lines else end


def cut(preds, label, files, core_lines):
    """Cut a ranked package; returns (regions kept, number of ranked regions consumed).
    Under a budget, regions are taken in rank order until m x core_lines unique lines are
    filled and the last one is trimmed."""
    kind, value = parse_cutoff(label)
    if kind == "full":
        return list(preds), len(preds)
    if kind == "k":
        return list(preds[:value]), min(value, len(preds))
    budget = round(value * core_lines)
    used, out, consumed = 0, [], 0
    covered = collections.defaultdict(set)
    for path, s, e_orig in preds:
        if used >= budget:
            break
        consumed += 1
        n = files.count(path)
        s = max(1, s)
        # Lines past the end of the file (a ground-truth artifact: the evaluator still counts
        # them) cost no budget, and the region keeps its recorded end.
        new = [x for x in range(s, clip_end(e_orig, n) + 1) if x not in covered[path]]
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
        covered[path].update(range(max(1, s), clip_end(e, files.count(path)) + 1))
    chars = 0
    for path, lines in covered.items():
        text = files.lines(path)
        chars += sum(len(text[i - 1]) + 1 for i in lines if i <= len(text))
    return sum(len(v) for v in covered.values()), chars // CHARS_PER_TOKEN


def core_regions(gt):
    return [(r["path"], r["start"], r["end"]) for r in gt.get("read_core_regions") or []]


@functools.cache
def evaluator():
    bench_data.use_official_code()
    from eval import ExploreEvaluator
    return ExploreEvaluator(bench_data.BENCH_PATH)


def score(rec, preds, files, labels):
    """Official metrics for every cutoff plus package size and recall of the files all
    agents modified. Returns (metrics by cutoff, number of ranked regions consumed)."""
    ev, gt = evaluator(), rec["ground_truth"]
    # path -> lines of every ground-truth file, as eval_runner._build_file_line_counts
    paths = {r["path"] for r in gt.get("read_core_regions") or []}
    for regions in (gt.get("read_optional_regions_map") or {}).values():
        paths.update(r["path"] for r in regions)
    line_counts = {p: n for p in paths if (n := files.count(p))}
    modified = set(gt.get("modified_core_files") or [])
    n_core = package_size(core_regions(gt), files)[0]
    out, consumed_max = {}, 0
    for label in labels:
        sliced, consumed = cut(preds, label, files, n_core)
        consumed_max = max(consumed_max, consumed)
        # The evaluator reads the current task from these attributes (as eval_runner.py does).
        ev._current_instance_id = rec["instance_id"]
        ev._current_file_line_counts = line_counts
        m = {name: getattr(ev, f"evaluate_{name}")(sliced, gt) for name in METRICS}
        m["modified_file_recall"] = len({p for p, _, _ in sliced} & modified) / len(modified) if modified else 0.0
        m["n_lines"], m["n_tokens"] = package_size(sliced, files)
        m["n_regions"] = len(sliced)
        m["core_lines"] = n_core
        out[label] = m
    return out, consumed_max
