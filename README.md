# Repository Context Retrieval Method for Agentic Software Engineering Systems

This repository contains source code and scripts used during the development of the repository context retrieval method:
- **Dataset processing:** SWE-Zero trajectory parsing, filtering, and prompt preprocessing.
- **Model training:** LoRA fine-tuning workflow for SLMs.
- **Evaluation:** SWE-Explore benchmarking and comparisons.

# Data Preprocessing Implementation Plan
## 1. Prompt Replacement
- **Message 0 (System):** Replace completely with a custom prompt enforcing read-only exploration, `<think>` tag usage and the clipping limit. Tool schemas are not described in the prompt: `scripts/preprocessing/tools.json` (`str_replace_editor` view-only, `execute_bash` as in OpenHands, `finish` without parameters) is passed as `tools=` to the chat template, in training and at inference.
- **Message 1 (User):** Keep content up to `</issue_description>`. Replace all subsequent text with a custom read-only Call-to-Action.

## 2. Trace Transformation
- **Slicing:** Truncate the trajectory array exactly before `first_edit_message_index`, where an edit is defined as:
  - `str_replace_editor` with `command != "view"`.
  - `execute_bash` where any simple command (split on `&&`, `||`, `;`, `|`) is a write: `rm`, `mv`, `cp`, `mkdir`, `touch`, `chmod`, `patch`, `tee`, `sed -i`, `perl -i`, `git apply/checkout/commit/...`, `find -delete`/`-exec rm`, or `echo`/`cat`/`printf` with `>`. Only the program word is checked, so patterns inside `grep`/`find` arguments are not edits.
  - Trajectories without an edit are cut before the teacher's own `finish` (`had_edit = False`).
- **Target Synthesis:** Append a final `role: assistant` message containing an empty `finish` tool call (`{"name": "finish", "arguments": "{}"}`).
- **CoT Distillation (Unified `<think>` with Header Stripping):**
  - Clean `message.content` across all assistant turns by stripping standalone rubric headers (`r"(?m)^\s*#{1,4}\s*(?:Phase\s+\d+|\d+\.\d+)\b[^\n]*\n?"`) and conversational greetings (`r"(?i)^I'll help you implement[^\n]*\n+"`).
  - For every `think` tool call, extract `thought` and any cleaned `message.content`, delete the `think` turn and its `"Your thought has been logged."` tool response, and buffer the text for the next assistant message.
  - On each remaining `role: assistant` turn, combine any buffered text with the turn's cleaned `message.content`, wrap the combined string in a single `<think>\n...\n</think>` block as `content` (or `""` if empty), and retain `tool_calls`. Text order follows the teacher: earlier buffered text, then the turn's prose, then its thought. Reasoning buffered before the edit goes to the synthetic `finish`.
- **Output Clipping:** Truncate any `role: tool` content exceeding 8,000 Qwen tokens and append `\n<response clipped>` (OpenHands itself clips file views at ~5.9k tokens).
- **Arguments** stay JSON strings in the stored rows; the training collator converts them to dicts before `apply_chat_template` (the Qwen 3.5 template iterates `arguments|items`).
- **Metric columns:** `cut_kind`, `had_edit`, `n_calls`, `n_clipped`, `finish_think_tokens`, `qwen_tokens` (row rendered with the Qwen 3.5 chat template and `tools.json`).

## 3. Data Filtering
*Note: Applied post-transformation; thresholds serve as baseline defaults subject to adjustment.*
- Drop trajectories where `had_edit == False` (no edit found during slicing).
- Drop trajectories belonging to evaluation benchmark repositories (e.g., `sympy`, `django`, `scikit-learn`, `matplotlib`, `requests`).
- Drop trajectories where `qwen_tokens` exceeds the length limit (default 32,768; chosen after `stats_transformed.py`).

## 4. Train/Validation Split
- Split the dataset strictly grouped by `repo` (zero repository overlap between train and validation sets).
- Lock in a fixed list of ~5–10 medium-sized repositories totaling 2,000–3,000 trajectories for the validation set, keeping all remaining repositories in the training set.


# Data preprocessing guide:
Dependencies: `pandas pyarrow transformers jinja2`. Paths and thresholds are constants at the top of each script; run from the repository root.

```bash
# 1. Transform all raw shards, no rows dropped (~20 min, 8 processes) -> data/transformed_shards/
python scripts/preprocessing/transform_traces.py

# 2. Statistics over metric columns -> choose MAX_QWEN_TOKENS, per-repo cap, VAL_REPOS
python scripts/preprocessing/stats_transformed.py

# 3. Apply filters (had_edit, benchmark repos, length limit) -> data/filtered_dataset.parquet
python scripts/preprocessing/filter_dataset.py

# 4. Export repo-grouped train.jsonl and val.jsonl, each row {"messages", "tools"} -> data/sft_ready/
python scripts/preprocessing/split_dataset.py
```

# Setup
```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

# Repository layout
| Path | Role |
|---|---|
| `data/SWE-Zero-openhands-trajectories/` | raw dataset: 64 parquet shards (~11 GB) + OpenHands `tools.json` |
| `data/eda_results/` | EDA outputs (aggregates, command counts, per-row metadata) |
| `data/transformed_shards/` | stage 1 output: 64 parquet shards, `messages_json` + metric columns (~7.2 GB) |
| `data/filtered_dataset.parquet`, `data/sft_ready/` | stage 3 and 4 outputs |
| `notebooks/` | EDA (`eda.ipynb`, `eda.py`) and the original preprocessing plan (`data_preprocessing.ipynb`, superseded by this README) |
| `scripts/preprocessing/` | pipeline stages 1–4 and `tools.json` (tool schemas for training and inference) |
| `eval/SWE-Explore-Bench/` | upstream SWE-Explore benchmark code (benchmark data is downloaded separately from the HF Hub) |

# SWE-Explore baselines (2026-10-20)
Methods without an LLM: `random`, `oracle`, `bm25`, `tfidf`, `potion` (official explorers from `eval/SWE-Explore-Bench`, unchanged) and `mentioned` (files named in the issue). Metrics come from the official `ExploreEvaluator`.