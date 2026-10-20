"""Stage 4: write val.jsonl (held-out repositories) and train.jsonl (shuffled once).

train.jsonl is the whole training pool in a fixed random order (SEED), so any
subset is its first N lines and subsets are nested:
    head -n 5000 data/sft_ready/train.jsonl > data/sft_ready/train_5k.jsonl

The shuffle never holds the whole pool in memory: rows go to N_BUCKETS temporary
files by their random rank, then each bucket is sorted and appended.
Run from the repository root:
    python scripts/preprocessing/split_dataset.py
"""

import glob
import json
import os

import numpy as np
import pandas as pd

KEPT_ROWS_FILE = "data/kept_rows.parquet"
SHARDS_DIR = "data/transformed_shards"
OUTPUT_DIR = "data/sft_ready"
TOOLS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools.json")
SEED = 42
N_BUCKETS = 64

# Held-out repositories for validation: 8 medium-sized repos (250–450 rows each),
# 2,647 rows. Picked at random among candidates and checked against the training
# pool: median length 19,876 vs 19,613 tokens, median 18 vs 19 calls, source mix
# within 4 p.p. (SWE-Fixer, SWE-Gym, SWE-rebench).
VAL_REPOS = [
    "openmc-dev/openmc",
    "fonttools/fonttools",
    "jazzband/pip-tools",
    "voxel51/fiftyone",
    "OpenMined/PySyft",
    "python-poetry/poetry",
    "DataDog/dd-agent",
    "narwhals-dev/narwhals",
]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(TOOLS_PATH) as f:
        tools = json.load(f)

    kept = pd.read_parquet(KEPT_ROWS_FILE)
    is_val = kept["repo"].isin(VAL_REPOS)
    assert is_val.any(), "No validation rows found"
    train_ids = kept.loc[~is_val, "trajectory_id"].to_numpy()
    val_ids = set(kept.loc[is_val, "trajectory_id"])
    assert set(kept.loc[~is_val, "repo"]).isdisjoint(VAL_REPOS), "Repo overlap detected!"

    # Random rank of every training row: its position in train.jsonl.
    ranks = np.random.default_rng(SEED).permutation(len(train_ids))
    rank_of = dict(zip(train_ids, ranks))
    bucket_size = -(-len(train_ids) // N_BUCKETS)
    print(f"Train rows: {len(train_ids):,}   validation rows: {len(val_ids):,}", flush=True)

    # Pass 1: stream the shards; validation rows go straight to val.jsonl,
    # training rows go to the bucket of their rank.
    tmp_dir = os.path.join(OUTPUT_DIR, "tmp_buckets")
    os.makedirs(tmp_dir, exist_ok=True)
    buckets = [open(os.path.join(tmp_dir, f"{b:03d}.txt"), "w", encoding="utf-8") for b in range(N_BUCKETS)]
    shards = sorted(set(kept["shard"]))
    with open(os.path.join(OUTPUT_DIR, "val.jsonl"), "w", encoding="utf-8") as val_file:
        for i, shard in enumerate(shards, 1):
            df = pd.read_parquet(os.path.join(SHARDS_DIR, shard), columns=["trajectory_id", "messages_json"])
            for trajectory_id, messages_json in zip(df["trajectory_id"], df["messages_json"]):
                line = json.dumps({"messages": json.loads(messages_json), "tools": tools}, ensure_ascii=False)
                if trajectory_id in val_ids:
                    val_file.write(line + "\n")
                elif trajectory_id in rank_of:
                    rank = rank_of[trajectory_id]
                    buckets[rank // bucket_size].write(f"{rank}\t{line}\n")
            print(f"[{i}/{len(shards)}] read {shard}", flush=True)
    for b in buckets:
        b.close()

    # Pass 2: sort each bucket by rank and append it to train.jsonl.
    with open(os.path.join(OUTPUT_DIR, "train.jsonl"), "w", encoding="utf-8") as train_file:
        for b in range(N_BUCKETS):
            path = os.path.join(tmp_dir, f"{b:03d}.txt")
            with open(path, encoding="utf-8") as f:
                rows = sorted((line.split("\t", 1) for line in f), key=lambda r: int(r[0]))
            train_file.writelines(r[1] for r in rows)
            os.remove(path)
            print(f"[{b + 1}/{N_BUCKETS}] wrote bucket", flush=True)
    os.rmdir(tmp_dir)

    print(f"Saved {len(train_ids):,} training rows and {len(val_ids):,} validation rows to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
