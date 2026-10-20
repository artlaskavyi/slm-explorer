"""Stage 1: transform raw SWE-Zero trajectories into exploration-only traces.

No filtering here: every row is written, together with metric columns that the
analysis stage (stats_transformed.py) uses to choose filter thresholds.

Run from the repository root:
    python scripts/preprocessing/transform_traces.py
"""

import glob
import json
import os
import re
import shlex
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from transformers import AutoTokenizer

# =====================================================================
# 1. CONSTANTS & PROMPTS
# =====================================================================
INPUT_GLOB = "data/SWE-Zero-openhands-trajectories/data/*.parquet"
OUTPUT_DIR = "data/transformed_shards"
TOOLS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools.json")
TOKENIZER_NAME = "Qwen/Qwen3.5-4B"
MAX_TOOL_TOKENS = 8000  # Qwen tokens; the same limit is used at inference
N_WORKERS = 8

with open(TOOLS_PATH) as f:
    TOOLS = json.load(f)

SYSTEM_PROMPT = """You are an expert AI Code Explorer. Your sole objective is to navigate a code repository, analyze the issue description, and localize the exact files, classes, and functions responsible for the bug or feature request.

<RULES>
1. READ-ONLY EXPLORATION: Inspect the codebase using only `str_replace_editor` (with `command="view"`) and read-only `execute_bash` commands (`cd`, `find`, `grep`, `egrep`, `ls`, `cat`, `head`, `tail`, `wc`, `awk`, `diff`, `git`, `pwd`, `tree`). Do NOT modify, create, or delete any files.
2. EXPLICIT REASONING: Before every tool call, wrap your step-by-step analysis and next-step hypothesis inside `<think>...</think>` tags.
3. HANDLING CLIPPED OUTPUTS: Tool outputs exceeding 8,000 tokens are truncated with `<response clipped>`. Use targeted line ranges or `grep` rather than reading massive files in full.
4. COMPLETION: Once you have localized the exact code sections needed to resolve the issue, call the `finish` tool with empty arguments.
</RULES>"""

USER_CTA = (
    "\n\nExplore the repository to locate the exact files and code blocks that "
    "need to be modified to resolve the issue described in <issue_description>. "
    "Do not modify any files. Always reason inside <think>...</think> tags "
    "before calling a tool, and call the `finish` tool once you have gathered "
    "sufficient context to implement the fix."
)

# Strips standalone Phase headers (e.g. "## Phase 1. READING", "### 1.1 Problem Analysis")
PHASE_HEADER_REGEX = re.compile(r"(?m)^\s*#{1,4}\s*(?:Phase\s+\d+|\d+\.\d+)\b[^\n]*\n?")
GREETING_REGEX = re.compile(r"(?i)^\s*I'll help you implement[^\n]*\n*")

# Edit detection: only the program word of each command segment is checked,
# so `grep -n "def patch"` or `grep -r "mkdir"` are not edits.
WRITE_PROGRAMS = {
    "rm", "mv", "cp", "mkdir", "rmdir", "touch", "chmod", "chown", "ln",
    "truncate", "patch", "tee",
}
GIT_WRITE_SUBCOMMANDS = {
    "apply", "checkout", "commit", "reset", "clean", "stash", "revert", "mv",
    "rm", "restore", "switch", "merge", "rebase", "cherry-pick", "am",
}
SEGMENT_SEPARATORS = {"&&", "||", ";", "|", "&", "(", ")"}
SED_INPLACE_FLAG = re.compile(r"-[a-zA-Z]*i\S*|--in-place\S*")


# =====================================================================
# 2. EDIT DETECTION
# =====================================================================
def parse_args(tool_call: dict) -> dict:
    raw_args = tool_call.get("function", {}).get("arguments", "{}")
    if isinstance(raw_args, dict):
        return raw_args
    try:
        return json.loads(raw_args)
    except Exception:
        return {}


def split_segments(cmd: str) -> list[list[str]]:
    """Splits a shell command into simple commands (lists of words)."""
    cmd = cmd.replace("\n", " ; ")
    try:
        lexer = shlex.shlex(cmd, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:  # unbalanced quotes
        tokens = re.sub(r"&&|\|\||[;|&()]", " ; ", cmd).split()

    segments, current = [], []
    for token in tokens:
        if token in SEGMENT_SEPARATORS:
            if current:
                segments.append(current)
            current = []
        else:
            current.append(token)
    if current:
        segments.append(current)
    return segments


def segment_is_write(words: list[str]) -> bool:
    while words and ("=" in words[0] or words[0] in {"sudo", "command", "xargs"}):
        words = [w for w in words[1:] if not (words[0] == "xargs" and w.startswith("-"))]
    if not words:
        return False

    program, args = os.path.basename(words[0]), words[1:]
    if program in WRITE_PROGRAMS:
        return True
    if program in {"sed", "perl"}:
        return any(SED_INPLACE_FLAG.fullmatch(a) for a in args)
    if program == "git":
        return bool(args) and args[0] in GIT_WRITE_SUBCOMMANDS
    if program == "find":
        if "-delete" in args:
            return True
        return any(
            a in {"-exec", "-execdir"} and i + 1 < len(args) and args[i + 1] in WRITE_PROGRAMS
            for i, a in enumerate(args)
        )
    if program in {"echo", "cat", "printf"}:
        return any(
            a in {">", ">>"} and i + 1 < len(args) and args[i + 1] != "/dev/null"
            for i, a in enumerate(args)
        )
    return False


def edit_kind(tool_call: dict) -> str | None:
    """Returns the kind of edit ("editor:<command>" or "bash"), or None for reads."""
    func_name = tool_call.get("function", {}).get("name", "")
    args = parse_args(tool_call)

    if func_name == "str_replace_editor" and args.get("command") != "view":
        return f"editor:{args.get('command')}"
    if func_name == "execute_bash":
        if any(segment_is_write(s) for s in split_segments(args.get("command", ""))):
            return "bash"
    return None


# =====================================================================
# 3. TRACE TRANSFORMATION
# =====================================================================
def clean_assistant_prose(text: str) -> str:
    if not text:
        return ""
    cleaned = GREETING_REGEX.sub("", text)
    cleaned = PHASE_HEADER_REGEX.sub("", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def transform_user_prompt(content: str) -> str:
    tag = "</issue_description>"
    idx = content.find(tag)
    if idx != -1:
        return content[: idx + len(tag)] + USER_CTA
    return content + USER_CTA


def cut_at_first_edit(trajectory: list) -> tuple[list, str]:
    """Pass 1: keeps the messages after the prompt and before the first edit.

    Without an edit the trajectory is cut before the teacher's own `finish`.
    """
    for idx, msg in enumerate(trajectory):
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            kind = edit_kind(tc)
            if kind:
                return list(trajectory[2:idx]), kind
            if tc.get("function", {}).get("name") == "finish":
                return list(trajectory[2:idx]), "none"
    return list(trajectory[2:]), "none"


def assistant_message(reasoning: list[str], tool_calls: list) -> dict:
    joined = "\n\n".join(reasoning)
    content = f"<think>\n{joined}\n</think>" if reasoning else ""
    return {"role": "assistant", "content": content, "tool_calls": tool_calls}


def merge_thoughts(body: list) -> list:
    """Pass 2: folds `think` calls and prose into the next real tool call.

    Text order follows the teacher: buffered text from earlier turns, then the
    turn's prose, then its `think` thought. `think` responses are dropped, and
    the buffered reasoning before the edit goes to the synthetic `finish`.
    """
    messages, pending = [], []
    awaiting_responses = 0

    for msg in body:
        role = msg.get("role")
        if role == "assistant":
            prose = clean_assistant_prose(msg.get("content") or "")
            if prose:
                pending.append(prose)
            calls = []
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function", {})
                if fn.get("name") == "think":
                    thought = parse_args(tc).get("thought", "").strip()
                    if thought:
                        pending.append(thought)
                    continue
                arguments = fn.get("arguments", "{}")
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                calls.append({
                    "id": tc.get("id"),
                    "type": "function",
                    "function": {"name": fn.get("name"), "arguments": arguments},
                })
            if calls:
                messages.append(assistant_message(pending, calls))
                pending = []
                awaiting_responses = len(calls)
        elif role == "tool" and awaiting_responses:
            messages.append({"role": "tool", "content": msg.get("content") or ""})
            awaiting_responses -= 1
        # other user messages and `think` responses are dropped

    finish_call = {
        "id": "call_finish_explorer",
        "type": "function",
        "function": {"name": "finish", "arguments": "{}"},
    }
    messages.append(assistant_message(pending, [finish_call]))
    return messages


# =====================================================================
# 4. TOKENIZATION (clipping and metrics in Qwen tokens)
# =====================================================================
_tokenizer = None


def tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    return _tokenizer


def count_tokens(text: str) -> int:
    return len(tokenizer()(text, add_special_tokens=False)["input_ids"]) if text else 0


def clip_tool_outputs(messages: list) -> int:
    """Clips tool outputs in place to MAX_TOOL_TOKENS; returns the number clipped."""
    tool_msgs = [m for m in messages if m["role"] == "tool" and m["content"]]
    if not tool_msgs:
        return 0
    ids = tokenizer()([m["content"] for m in tool_msgs], add_special_tokens=False)["input_ids"]
    n_clipped = 0
    for msg, msg_ids in zip(tool_msgs, ids):
        if len(msg_ids) > MAX_TOOL_TOKENS:
            msg["content"] = tokenizer().decode(msg_ids[:MAX_TOOL_TOKENS]) + "\n<response clipped>"
            n_clipped += 1
    return n_clipped


def rendered_tokens(messages: list) -> int:
    """Length of the row as the model sees it: chat template + tool schemas.

    The template needs arguments as dicts; the stored rows keep JSON strings.
    """
    render = [
        {**m, "tool_calls": [
            {**tc, "function": {**tc["function"], "arguments": json.loads(tc["function"]["arguments"])}}
            for tc in m["tool_calls"]
        ]} if m.get("tool_calls") else m
        for m in messages
    ]
    text = tokenizer().apply_chat_template(render, tools=TOOLS, tokenize=False)
    return count_tokens(text)


def transform_trajectory(trajectory: list) -> dict:
    body, cut_kind = cut_at_first_edit(trajectory)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": transform_user_prompt(trajectory[1].get("content") or "")},
        *merge_thoughts(body),
    ]
    n_clipped = clip_tool_outputs(messages)
    return {
        "cut_kind": cut_kind,
        "had_edit": cut_kind != "none",
        "n_calls": sum(1 for m in messages if m["role"] == "assistant") - 1,
        "n_clipped": n_clipped,
        "finish_think_tokens": count_tokens(messages[-1]["content"]),
        "qwen_tokens": rendered_tokens(messages),
        "messages_json": json.dumps(messages, ensure_ascii=False),
    }


# =====================================================================
# 5. BATCH EXECUTION
# =====================================================================
def transform_shard(file_path: str) -> str:
    out_path = os.path.join(OUTPUT_DIR, os.path.basename(file_path))
    if os.path.exists(out_path):
        return f"skip {out_path}"

    df = pd.read_parquet(
        file_path, columns=["instance_id", "repo", "trajectory_id", "dataset", "trajectory"]
    )
    records = []
    for row in df.itertuples(index=False):
        records.append({
            "instance_id": row.instance_id,
            "repo": row.repo,
            "trajectory_id": row.trajectory_id,
            "dataset": row.dataset,
            **transform_trajectory(row.trajectory),
        })
    pd.DataFrame(records).to_parquet(out_path + ".tmp", index=False)
    os.replace(out_path + ".tmp", out_path)
    return f"done {out_path} ({len(records)} rows)"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = sorted(glob.glob(INPUT_GLOB))
    print(f"Found {len(files)} parquet shards to transform.")
    with ProcessPoolExecutor(N_WORKERS) as pool:
        for message in pool.map(transform_shard, files):
            print(message, flush=True)


if __name__ == "__main__":
    main()
