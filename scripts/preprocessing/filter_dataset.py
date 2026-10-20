"""Stage 3: choose the rows that stay, using only the metric columns.

Writes a small list of kept rows (trajectory_id, repo, shard); the messages are
read again only in stage 4. Run from the repository root:
    python scripts/preprocessing/filter_dataset.py
"""

import glob
import os

import pandas as pd

# =====================================================================
# FILTER THRESHOLDS (chosen after stats_transformed.py)
# =====================================================================
INPUT_GLOB = "data/transformed_shards/*.parquet"
OUTPUT_FILE = "data/kept_rows.parquet"
MAX_QWEN_TOKENS = 40960  # keeps 96.7% of rows; 32k would drop the longest explorations

# Repositories of SWE-Explore (verified part = SWE-bench Verified) present in SWE-Zero.
# Pro and multilingual repositories do not occur in SWE-Zero.
BENCHMARK_REPOS = {
    "astropy/astropy",
    "django/django",
    "matplotlib/matplotlib",
    "mwaskom/seaborn",
    "pallets/flask",
    "psf/requests",
    "pydata/xarray",
    "pylint-dev/astroid",
    "pylint-dev/pylint",
    "pytest-dev/pytest",
    "scikit-learn/scikit-learn",
    "sphinx-doc/sphinx",
    "sympy/sympy",
}


def main():
    df = pd.concat(
        [
            pd.read_parquet(f, columns=["trajectory_id", "repo", "had_edit", "qwen_tokens"])
            .assign(shard=os.path.basename(f))
            for f in sorted(glob.glob(INPUT_GLOB))
        ],
        ignore_index=True,
    )
    assert df["trajectory_id"].is_unique, "trajectory_id is not unique"
    initial_count = len(df)
    print(f"Total transformed trajectories: {initial_count:,}")

    before = len(df)
    df = df[df["had_edit"]]
    print(f"Dropped had_edit == False: {before - len(df):,} rows")

    before = len(df)
    df = df[~df["repo"].str.lower().isin(BENCHMARK_REPOS)]
    print(f"Dropped benchmark repositories: {before - len(df):,} rows")

    before = len(df)
    df = df[df["qwen_tokens"] <= MAX_QWEN_TOKENS]
    print(f"Dropped qwen_tokens > {MAX_QWEN_TOKENS}: {before - len(df):,} rows")

    print(f"Kept: {len(df):,} ({len(df) / initial_count:.2%})")
    df[["trajectory_id", "repo", "shard"]].to_parquet(OUTPUT_FILE, index=False)
    print(f"Saved kept rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
