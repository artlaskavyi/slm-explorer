"""Stage 2: statistics over the transformed shards, used to choose filter thresholds.

Reads only the metric columns, so the whole dataset fits in memory.
Run from the repository root:
    python scripts/preprocessing/stats_transformed.py
"""

import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from filter_dataset import BENCHMARK_REPOS  # noqa: E402

INPUT_GLOB = "data/transformed_shards/*.parquet"
METRIC_COLUMNS = [
    "instance_id", "repo", "dataset", "cut_kind", "had_edit",
    "n_calls", "n_clipped", "finish_think_tokens", "qwen_tokens",
]
PERCENTILES = [0.5, 0.75, 0.9, 0.95, 0.99]
LENGTH_LIMITS = [16384, 24576, 32768, 40960, 49152, 65536]
REPO_CAPS = [250, 500, 1000, 2000]


def main():
    df = pd.concat(
        [pd.read_parquet(f, columns=METRIC_COLUMNS) for f in sorted(glob.glob(INPUT_GLOB))],
        ignore_index=True,
    )
    print(f"Rows: {len(df):,}   tasks: {df.instance_id.nunique():,}   repos: {df.repo.nunique():,}")
    print("\ncut_kind:\n" + df.cut_kind.value_counts().to_string())

    is_bench = df.repo.str.lower().isin(BENCHMARK_REPOS)
    print(f"\nhad_edit == False: {(~df.had_edit).sum():,}   benchmark repos: {is_bench.sum():,}")
    df = df[df.had_edit & ~is_bench]
    print(f"Candidates after these two filters: {len(df):,}")

    numeric = ["n_calls", "n_clipped", "finish_think_tokens", "qwen_tokens"]
    print("\n" + df[numeric].describe(percentiles=PERCENTILES).T.round(0).to_string())
    print(f"\nRows with >=1 clipped output: {(df.n_clipped > 0).mean():.2%}")

    print("\nLength limit (qwen_tokens):")
    for limit in LENGTH_LIMITS:
        kept = df.qwen_tokens <= limit
        print(f"  <= {limit:>6}: keeps {kept.mean():6.2%} ({kept.sum():,} rows), "
              f"tasks kept {df[kept].instance_id.nunique() / df.instance_id.nunique():.2%}")

    # Are long rows concentrated in specific repos / sources, or harder tasks in general?
    long_rows = df.qwen_tokens > 32768
    print(f"\nRows > 32768: {long_rows.sum():,}")
    print("  median n_calls: long %.0f vs rest %.0f" % (df[long_rows].n_calls.median(), df[~long_rows].n_calls.median()))
    print("  share of long rows by source dataset:\n" + df.groupby("dataset").apply(
        lambda g: f"{(g.qwen_tokens > 32768).mean():.2%} of {len(g):,}", include_groups=False).to_string())
    long_share = df.groupby("repo").qwen_tokens.apply(lambda s: (s > 32768).mean())
    repo_size = df.repo.value_counts()
    big = repo_size[repo_size >= 200].index
    print("  repos (>=200 rows) with the highest share of long rows:")
    print(long_share[big].sort_values(ascending=False).head(10).round(3).to_string())
    print("  long rows covered by top-10 repos: %.2f%%" % (
        100 * df[long_rows].repo.value_counts().head(10).sum() / max(1, long_rows.sum())))

    print("\nTrajectories per repo:")
    print(repo_size.describe(percentiles=PERCENTILES).round(0).to_string())
    print("Top repos:\n" + repo_size.head(10).to_string())
    for cap in REPO_CAPS:
        print(f"  cap {cap:>5} per repo: keeps {repo_size.clip(upper=cap).sum():,} rows")

    candidates = repo_size[(repo_size >= 250) & (repo_size <= 450)]
    print(f"\nValidation repo candidates (250-450 rows): {len(candidates)} repos, {candidates.sum():,} rows")


if __name__ == "__main__":
    main()
