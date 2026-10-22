"""Summary tables of a results run: means by method, cutoff and benchmark part.

Reads every <method>.jsonl of the run directory, writes summary.csv and summary.md.
Methods follow the order of config.json, then the remaining files alphabetically.
    python scripts/eval/summarize.py results/baselines/2026-10-22_final
"""

import argparse
import csv
import json
from pathlib import Path

from bench import parse_cutoff
from bench_data import read_jsonl

PARTS = ["verified", "pro", "multilingual", "all"]
# Columns of Table 6 in the paper, in its order, then our additions.
SUMMARY = [
    ("hit_region_rate", "HitReg"), ("precision", "Prec"), ("recall", "Rec"), ("f1_score", "F1"),
    ("hit_file_rate", "HitFile"), ("ndcg_at_500", "nDCG@500"), ("recall_at_500", "Rec@500"),
    ("first_useful_hit", "FUH"), ("context_efficiency", "CtxEff"), ("noise_region_rate", "NoiseReg"),
    ("modified_file_recall", "ModFile"), ("n_lines", "Lines"), ("n_tokens", "Tok"),
    ("index_seconds", "Index s"), ("search_seconds", "Search s"),
]
ROW_FIELDS = {"index_seconds", "search_seconds"}  # stored per task, not per cutoff
INTRO = """\
# SWE-Explore results

Means over tasks. A failed task counts as an empty package. Cutoff `K` = first K regions;
`C<m>` = regions in rank order up to m x (lines of the core ground truth of the task);
`full` = the whole package of a method that stops on its own. Part `all` is the mean
over every task (the paper's numbers); parts are reported separately. Run parameters:
config.json.
"""


def method_files(run_dir):
    config = run_dir / "config.json"
    order = json.loads(config.read_text()).get("methods", []) if config.is_file() else []
    return sorted(run_dir.glob("*.jsonl"), key=lambda p: (order.index(p.stem) if p.stem in order else len(order), p.stem))


def mean_rows(run_dir):
    """[(method, cutoff, part, n, n_errors, {column: mean})]"""
    rows = []
    for path in method_files(run_dir):
        recs = read_jsonl(path)
        if not recs:
            continue
        for label in recs[0]["metrics"]:
            for part in PARTS:
                sub = [r for r in recs if part == "all" or r["dataset"] == part]
                if not sub:
                    continue
                means = {}
                for key, _ in SUMMARY:
                    vals = [r.get(key, 0.0) if key in ROW_FIELDS else r["metrics"][label][key] for r in sub]
                    means[key] = sum(vals) / len(vals)
                rows.append((path.stem, label, part, len(sub), sum(bool(r["error"]) for r in sub), means))
    return rows


def cutoff_title(label):
    kind, value = parse_cutoff(label)
    if kind == "full":
        return "whole package"
    return f"budget {value:g} x core lines" if kind == "core" else f"K={value}"


def cell(key, value):
    if key in ("n_lines", "n_tokens"):
        return f"{value:.0f}"
    return f"{value:.2f}" if key in ROW_FIELDS else f"{value:.3f}"


def summarize(run_dir):
    """Write summary.csv and summary.md; return the markdown."""
    rows = mean_rows(run_dir)
    with open(run_dir / "summary.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["method", "cutoff", "part", "n", "errors"] + [key for key, _ in SUMMARY])
        for method, label, part, n, n_err, means in rows:
            w.writerow([method, label, part, n, n_err] + [f"{means[key]:.4f}" for key, _ in SUMMARY])

    lines = [INTRO]
    head = "| Method | n | err | " + " | ".join(name for _, name in SUMMARY) + " |"
    for label in dict.fromkeys(r[1] for r in rows):
        for part in PARTS:
            sel = [r for r in rows if r[1] == label and r[2] == part]
            if not sel:
                continue
            lines += [f"## {cutoff_title(label)}, {part}", "", head, "|---" * (len(SUMMARY) + 3) + "|"]
            for method, _, _, n, n_err, means in sel:
                lines.append(f"| {method} | {n} | {n_err} | " + " | ".join(cell(k, means[k]) for k, _ in SUMMARY) + " |")
            lines.append("")
    text = "\n".join(lines)
    (run_dir / "summary.md").write_text(text)
    return text


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    print(summarize(ap.parse_args().run_dir))
