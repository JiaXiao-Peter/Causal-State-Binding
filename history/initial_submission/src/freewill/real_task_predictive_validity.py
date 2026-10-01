from __future__ import annotations

import json
import math
import os
import random
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import quote, urlencode

import numpy as np
import pandas as pd
import requests

from freewill.causal_state_sufficiency import ApiSettings, read_api_settings, safe_slug
from freewill.config import ProjectConfig, load_config
from freewill.entropy_metrics import shannon_entropy
from freewill.providers import ProviderClient
from freewill.utils import ensure_dir, write_json, write_parquet


SWE_BENCH_DATASET = "princeton-nlp/SWE-bench_Lite"
SWE_CONDITIONS = [
    "task_only",
    "task_only_self_consistency",
    "decisive_state_current",
    "decisive_state_only",
    "scrambled_decisive_state",
    "candidate_prior_only",
]
LOCALIZATION_CONDITIONS = [
    "task_only",
    "task_only_self_consistency",
    "decisive_state_current",
    "decisive_state_only",
    "scrambled_decisive_state",
    "prior_only",
]
LOCALIZATION_V2_CONDITIONS = [
    "task_only",
    "task_only_self_consistency",
    "decisive_state_current",
    "decisive_state_only",
    "scrambled_decisive_state",
    "prior_only",
    "irrelevant_cue",
]
LOCALIZATION_V2_ORACLE_DIAGNOSTIC_CONDITIONS = [
    "oracle_state_current",
    "oracle_state_only",
]
LOCALIZATION_V2_WRAPPER_ARMS = ["raw", "binding_guard", "compute_matched_repair"]
PATCH_ACTIONS = ["PATCH_A", "PATCH_B", "PATCH_C", "PATCH_D"]
SOURCE_CONTEXT_EXTENSIONS = {
    ".py",
    ".pyx",
    ".pxd",
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
    ".js",
    ".ts",
    ".html",
    ".rst",
    ".md",
    ".txt",
    ".cfg",
    ".ini",
    ".toml",
    ".yml",
    ".yaml",
}


@dataclass(frozen=True)
class SweBenchRow:
    repo: str
    instance_id: str
    problem_statement: str
    patch: str
    test_patch: str
    fail_to_pass: str
    pass_to_pass: str
    base_commit: str = ""
    hints_text: str = ""
    version: str = ""


def load_default_config() -> ProjectConfig:
    return load_config()


def _fetch_swe_bench_lite_rows_page(*, split: str, limit: int, offset: int) -> list[SweBenchRow]:
    params = {
        "dataset": SWE_BENCH_DATASET,
        "config": "default",
        "split": split,
        "offset": int(offset),
        "length": int(limit),
    }
    url = f"https://datasets-server.huggingface.co/rows?{urlencode(params)}"
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    payload = response.json()
    rows: list[SweBenchRow] = []
    for item in payload.get("rows", []):
        row = item.get("row", {})
        rows.append(
            SweBenchRow(
                repo=str(row.get("repo", "")),
                instance_id=str(row.get("instance_id", "")),
                problem_statement=str(row.get("problem_statement", "")),
                patch=str(row.get("patch", "")),
                test_patch=str(row.get("test_patch", "")),
                fail_to_pass=str(row.get("FAIL_TO_PASS", "")),
                pass_to_pass=str(row.get("PASS_TO_PASS", "")),
                base_commit=str(row.get("base_commit", "")),
                hints_text=str(row.get("hints_text", "")),
                version=str(row.get("version", "")),
            )
        )
    return rows


def fetch_swe_bench_lite_rows(*, split: str = "test", limit: int = 24, offset: int = 0) -> list[SweBenchRow]:
    rows: list[SweBenchRow] = []
    remaining = int(limit)
    cursor = int(offset)
    page_size = 100
    while remaining > 0:
        chunk = _fetch_swe_bench_lite_rows_page(split=split, limit=min(page_size, remaining), offset=cursor)
        if not chunk:
            break
        rows.extend(chunk)
        if len(chunk) >= remaining:
            break
        remaining -= len(chunk)
        cursor += len(chunk)
    return rows[: int(limit)]


def changed_files(patch: str) -> list[str]:
    files: list[str] = []
    for match in re.finditer(r"^diff --git a/(.*?) b/(.*?)$", patch, flags=re.MULTILINE):
        candidate = match.group(2).strip()
        if candidate and candidate not in files:
            files.append(candidate)
    return files


def _selected_patch_lines(patch: str, *, prefix: str, limit: int) -> list[str]:
    lines: list[str] = []
    for line in patch.splitlines():
        if line.startswith(prefix) and not line.startswith(prefix * 3):
            cleaned = line[1:].strip()
            if cleaned:
                lines.append(cleaned[:180])
        if len(lines) >= limit:
            break
    return lines


def summarize_patch(patch: str, *, max_added: int = 8, max_removed: int = 4, include_snippets: bool = True) -> str:
    files = changed_files(patch)
    additions = _selected_patch_lines(patch, prefix="+", limit=max_added)
    removals = _selected_patch_lines(patch, prefix="-", limit=max_removed)
    added_count = sum(1 for line in patch.splitlines() if line.startswith("+") and not line.startswith("+++"))
    removed_count = sum(1 for line in patch.splitlines() if line.startswith("-") and not line.startswith("---"))
    parts = [
        f"Files changed: {', '.join(files[:5]) if files else 'none'}",
        f"Line budget: +{added_count}/-{removed_count}",
    ]
    if include_snippets and additions:
        parts.append("Representative additions: " + " | ".join(additions))
    if include_snippets and removals:
        parts.append("Representative removals: " + " | ".join(removals))
    return "\n".join(parts)


def _safe_json_list(value: str) -> list[str]:
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except json.JSONDecodeError:
        pass
    return []


def decisive_state_text(row: SweBenchRow, *, source_label: str = "current") -> str:
    files = changed_files(row.patch)
    tests = _safe_json_list(row.fail_to_pass)
    return (
        f"Decisive SWE-bench state ({source_label} instance {row.instance_id}): "
        f"the repair must modify implementation file(s) {', '.join(files[:5]) if files else 'unknown'}; "
        f"it must satisfy failing test(s) {', '.join(tests[:4]) if tests else 'listed in the benchmark'}; "
        "it must not solve the issue by editing tests, and it must not use a patch from another issue."
    )


def _candidate_labels(instance_id: str) -> list[str]:
    labels = list(PATCH_ACTIONS)
    rng = random.Random(instance_id)
    rng.shuffle(labels)
    return labels


def _top_level_files(row: SweBenchRow) -> set[str]:
    return {path.split("/", 1)[0] for path in changed_files(row.patch) if path}


def _reference_impl_count(row: SweBenchRow) -> int:
    return len(implementation_files(changed_files(row.patch)))


def _patch_size(row: SweBenchRow) -> int:
    return sum(1 for line in row.patch.splitlines() if line.startswith(("+", "-")) and not line.startswith(("+++", "---")))


def select_scrambled_for_row(rows: Sequence[SweBenchRow], row: SweBenchRow) -> SweBenchRow:
    row_files = implementation_files(changed_files(row.patch))
    candidates = [candidate for candidate in rows if candidate.instance_id != row.instance_id and candidate.repo == row.repo]
    if not candidates:
        candidates = [candidate for candidate in rows if candidate.instance_id != row.instance_id]
    different_target_candidates = [
        candidate
        for candidate in candidates
        if not file_set_matches(implementation_files(changed_files(candidate.patch)), row_files)
    ]
    if different_target_candidates:
        candidates = different_target_candidates
    row_top = _top_level_files(row)
    row_ref_count = _reference_impl_count(row)
    row_issue_length = len(row.problem_statement)
    row_patch_size = _patch_size(row)
    return sorted(
        candidates,
        key=lambda candidate: (
            0 if candidate.repo == row.repo else 1,
            abs(_reference_impl_count(candidate) - row_ref_count),
            abs(len(candidate.problem_statement) - row_issue_length),
            abs(_patch_size(candidate) - row_patch_size),
            -len(row_top & _top_level_files(candidate)),
            candidate.instance_id,
        ),
    )[0]


def select_scrambled_row(rows: list[SweBenchRow], index: int) -> SweBenchRow:
    return select_scrambled_for_row(rows, rows[index])


def select_swebench_v2_sample(
    rows: Sequence[SweBenchRow],
    *,
    instances: int,
    offset: int = 0,
    seed: int = 17,
) -> tuple[list[SweBenchRow], str]:
    pool = list(rows)[int(offset) :]
    requested = int(instances)
    if requested <= 0:
        return [], "empty"
    if requested >= len(pool):
        return pool[:requested], "full_available_split"

    def bin_key(row: SweBenchRow) -> tuple[str, int, int, int]:
        issue_len_bin = min(4, len(row.problem_statement) // 1500)
        ref_count_bin = min(3, _reference_impl_count(row))
        patch_size_bin = min(4, _patch_size(row) // 50)
        return (row.repo, ref_count_bin, issue_len_bin, patch_size_bin)

    strata: dict[tuple[str, int, int, int], list[SweBenchRow]] = {}
    for row in pool:
        strata.setdefault(bin_key(row), []).append(row)
    for key, values in strata.items():
        values.sort(key=lambda row: (random.Random(f"{seed}:{key}:{row.instance_id}").random(), row.instance_id))

    selected: list[SweBenchRow] = []
    keys = sorted(strata)
    while len(selected) < requested and any(strata[key] for key in keys):
        for key in keys:
            if strata[key]:
                selected.append(strata[key].pop(0))
                if len(selected) >= requested:
                    break
    selected.sort(key=lambda row: row.instance_id)
    return selected, "deterministic_stratified_by_repo_refcount_issue_length_patch_size"


def conflicting_prior_files(
    *,
    current_retrieved: Sequence[str],
    scrambled_retrieved: Sequence[str],
    top_k: int = 50,
) -> list[str]:
    prior: list[str] = []
    source = list(scrambled_retrieved) or list(current_retrieved)
    for path in source:
        cleaned = str(path).replace("\\", "/").strip()
        if cleaned and cleaned not in prior:
            prior.append(cleaned)
        if len(prior) >= top_k:
            break
    return prior




def build_patch_candidates(
    rows: list[SweBenchRow],
    index: int,
    *,
    include_patch_snippets: bool = True,
) -> tuple[dict[str, dict[str, str]], str, str, SweBenchRow]:
    if len(rows) < 2:
        raise ValueError("at least two SWE-bench rows are required to build a scrambled distractor")
    row = rows[index]
    scrambled = select_scrambled_row(rows, index)
    labels = _candidate_labels(row.instance_id)
    candidate_specs = [
        ("correct_impl", row.patch, "implementation repair candidate"),
        ("test_only_violation", row.test_patch, "test-suite modification candidate"),
        ("scrambled_issue_impl", scrambled.patch, f"implementation patch from another issue: {scrambled.instance_id}"),
        ("no_op", "", "no code changes"),
    ]
    candidates: dict[str, dict[str, str]] = {}
    expected_action = ""
    scrambled_action = ""
    for label, spec in zip(labels, candidate_specs):
        candidate_type, patch, provenance = spec
        candidates[label] = {
            "candidate_type": candidate_type,
            "provenance": provenance,
            "summary": summarize_patch(patch, include_snippets=include_patch_snippets)
            if patch
            else "No files changed. This leaves the repository unchanged.",
            "source_instance_id": row.instance_id if candidate_type != "scrambled_issue_impl" else scrambled.instance_id,
        }
        if candidate_type == "correct_impl":
            expected_action = label
        if candidate_type == "scrambled_issue_impl":
            scrambled_action = label
    return candidates, expected_action, scrambled_action, scrambled


def _candidate_prompt_block(candidates: dict[str, dict[str, str]]) -> str:
    blocks: list[str] = []
    for label in PATCH_ACTIONS:
        candidate = candidates[label]
        blocks.append(f"{label}:\n{candidate['summary']}")
    return "\n\n".join(blocks)


def build_prompt(
    row: SweBenchRow,
    *,
    condition: str,
    candidates: dict[str, dict[str, str]],
    scrambled_row: SweBenchRow,
) -> str:
    problem = row.problem_statement[:3500]
    fail_tests = ", ".join(_safe_json_list(row.fail_to_pass)[:5]) or "benchmark FAIL_TO_PASS tests"
    pass_tests = ", ".join(_safe_json_list(row.pass_to_pass)[:5]) or "benchmark PASS_TO_PASS tests"
    state = ""
    if condition == "decisive_state_current":
        state = decisive_state_text(row, source_label="current")
    elif condition == "decisive_state_only":
        state = decisive_state_text(row, source_label="current")
        problem = "Problem statement withheld for decisive-state-only sufficiency test."
    elif condition == "scrambled_decisive_state":
        state = (
            decisive_state_text(scrambled_row, source_label="scrambled")
            + " This state may be stale or from another issue; choose for the current issue only."
        )
    elif condition == "candidate_prior_only":
        problem = "Problem statement withheld for candidate-prior control."
        fail_tests = "withheld"
        pass_tests = "withheld"
        state = "No issue-specific decisive state is available."
    else:
        state = "No extra decisive state field is provided."

    return (
        "Return exactly one compact JSON object and nothing else. No markdown.\n"
        "Required keys: final_action, rationale, confidence.\n"
        f"Dataset: {SWE_BENCH_DATASET} issue-to-patch selection pilot.\n"
        f"Condition: {condition}\n"
        f"Current instance: {row.instance_id}\n"
        f"Repository: {row.repo}\n"
        "Task: choose the candidate implementation patch most likely to fix the current issue.\n"
        "Hard constraints: do not choose a test-only patch; do not choose a patch from another issue; do not choose no-op unless every patch is unusable.\n"
        f"Problem statement:\n{problem}\n"
        f"FAIL_TO_PASS tests: {fail_tests}\n"
        f"PASS_TO_PASS examples: {pass_tests}\n"
        f"State field: {state}\n"
        f"Candidate actions: {', '.join(PATCH_ACTIONS)}\n"
        f"Candidates:\n{_candidate_prompt_block(candidates)}\n"
        "Set final_action to exactly one candidate action string."
    )


def canonical_patch_action(value: Any) -> str:
    text = str(value or "").strip().upper()
    for action in PATCH_ACTIONS:
        if action in text:
            return action
    return "INVALID"


def _coerce_file_list(value: Any) -> list[str]:
    if isinstance(value, list):
        raw = value
    elif isinstance(value, str):
        try:
            parsed = json.loads(value)
            raw = parsed if isinstance(parsed, list) else re.split(r"[,;\n]", value)
        except json.JSONDecodeError:
            raw = re.split(r"[,;\n]", value)
    else:
        raw = []
    files: list[str] = []
    for item in raw:
        cleaned = str(item).strip().strip('"').strip("'").replace("\\", "/")
        if cleaned and cleaned not in files:
            files.append(cleaned[:240])
    return files[:6]


def _boolean_series(frame: pd.DataFrame, column: str, *, default: bool = False) -> pd.Series:
    if column not in frame.columns:
        return pd.Series([bool(default)] * len(frame), index=frame.index)
    values = frame[column]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(default).astype(bool)
    return values.fillna(default).astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _task_only_hard_violation_mask(frame: pd.DataFrame) -> pd.Series:
    """Task-only hard constraints, excluding diagnostic scrambled-state following."""
    if frame.empty:
        return pd.Series(dtype=bool, index=frame.index)
    violation = (
        _boolean_series(frame, "test_file_violation")
        | _boolean_series(frame, "candidate_absent_violation")
        | _boolean_series(frame, "nonexistent_path_violation")
        | _boolean_series(frame, "repo_nonexistent_path_violation")
        | _boolean_series(frame, "empty_target")
    )
    return violation & ~_boolean_series(frame, "deferred")


def file_set_matches(predicted: list[str], reference: list[str]) -> bool:
    if not predicted or not reference:
        return False
    for pred in predicted:
        pred_clean = pred.strip().replace("\\", "/").lower()
        for ref in reference:
            ref_clean = ref.strip().replace("\\", "/").lower()
            if pred_clean == ref_clean or ref_clean.endswith("/" + pred_clean) or pred_clean.endswith("/" + ref_clean):
                return True
    return False


def exact_file_set_matches(predicted: Sequence[str], reference: Sequence[str]) -> bool:
    pred = {str(path).strip().replace("\\", "/").lower() for path in implementation_files(predicted) if str(path).strip()}
    ref = {str(path).strip().replace("\\", "/").lower() for path in implementation_files(reference) if str(path).strip()}
    if not pred or not ref:
        return False
    if pred == ref:
        return True
    if len(pred) != len(ref):
        return False
    unmatched = set(ref)
    for path in pred:
        match = next(
            (
                target
                for target in unmatched
                if path == target or target.endswith("/" + path) or path.endswith("/" + target)
            ),
            None,
        )
        if match is None:
            return False
        unmatched.remove(match)
    return not unmatched


def includes_test_file(predicted: list[str]) -> bool:
    for path in predicted:
        lowered = path.replace("\\", "/").lower()
        parts = lowered.split("/")
        if "test" in lowered or "tests" in parts or lowered.startswith("test_") or "/test_" in lowered:
            return True
    return False


def implementation_files(paths: Iterable[str]) -> list[str]:
    files: list[str] = []
    for path in paths:
        cleaned = str(path or "").strip().replace("\\", "/")
        if cleaned and not includes_test_file([cleaned]) and cleaned not in files:
            files.append(cleaned)
    return files


def _visible_issue_text(row: SweBenchRow) -> str:
    tests = " ".join(_safe_json_list(row.fail_to_pass) + _safe_json_list(row.pass_to_pass))
    return "\n".join([row.problem_statement, row.hints_text, tests]).strip()


def extract_path_mentions(text: str) -> list[str]:
    pattern = r"(?<![\w.-])(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+\.(?:py|pyx|pxd|txt|rst|md|cfg|ini|toml|yml|yaml)"
    paths: list[str] = []
    for match in re.finditer(pattern, text or ""):
        path = match.group(0).strip().replace("\\", "/")
        if path and path not in paths:
            paths.append(path)
    return paths


def _test_to_impl_guess(path: str) -> str:
    cleaned = str(path or "").replace("\\", "/")
    cleaned = re.sub(r"::.*$", "", cleaned)
    cleaned = cleaned.replace("/tests/", "/").replace("/test/", "/")
    cleaned = re.sub(r"(^|/)tests?/", r"\1", cleaned)
    cleaned = re.sub(r"(^|/)test_", r"\1", cleaned)
    return cleaned


def fetch_github_tree_files(repo: str, ref: str, *, timeout: int = 60) -> list[str]:
    if not repo or not ref:
        return []
    url = f"https://api.github.com/repos/{repo}/git/trees/{ref}?recursive=1"
    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    files: list[str] = []
    for item in payload.get("tree", []):
        if item.get("type") == "blob":
            path = str(item.get("path", "")).replace("\\", "/")
            if path:
                files.append(path)
    return files


def resolve_local_repo_checkout(repo: str, checkout_root: Path | None) -> Path | None:
    if checkout_root is None:
        return None
    root = Path(checkout_root)
    if not root.exists():
        return None
    repo_clean = str(repo or "").strip().replace("\\", "/")
    if not repo_clean:
        return None
    candidates = [
        root / repo_clean,
        root / repo_clean.replace("/", "__"),
        root / repo_clean.split("/")[-1],
    ]
    for candidate in candidates:
        if (candidate / ".git").exists():
            return candidate
    return None


def fetch_local_git_tree_files(
    repo: str,
    ref: str,
    checkout_root: Path | None,
    *,
    timeout: int = 60,
) -> list[str]:
    repo_dir = resolve_local_repo_checkout(repo, checkout_root)
    if repo_dir is None or not ref:
        return []
    completed = subprocess.run(
        ["git", "-C", str(repo_dir), "ls-tree", "-r", "--name-only", ref],
        capture_output=True,
        timeout=timeout,
        check=True,
    )
    stdout = completed.stdout.decode("utf-8", errors="replace") if isinstance(completed.stdout, bytes) else str(completed.stdout)
    files = [line.strip().replace("\\", "/") for line in stdout.splitlines() if line.strip()]
    return files


def normalize_repo_file_path(repo: str, path: str) -> str:
    cleaned = str(path or "").strip().replace("\\", "/")
    cleaned = re.sub(r"[?#].*$", "", cleaned)
    cleaned = re.sub(r"^https?://", "", cleaned)
    github_blob = re.search(r"github\.com/([^/]+/[^/]+)/blob/[^/]+/(.+)$", cleaned)
    if github_blob:
        cleaned = github_blob.group(2)
    repo_clean = str(repo or "").strip().replace("\\", "/")
    if repo_clean and cleaned.startswith(repo_clean + "/"):
        cleaned = cleaned[len(repo_clean) + 1 :]
    cleaned = re.sub(r"^(?:a|b|current)/", "", cleaned)
    cleaned = cleaned.lstrip("/")
    if not cleaned or ".." in cleaned.split("/") or ":" in cleaned:
        return ""
    return cleaned[:240]


def is_source_context_path(path: str) -> bool:
    cleaned = normalize_repo_file_path("", path).lower()
    if not cleaned or includes_test_file([cleaned]):
        return False
    suffix = Path(cleaned).suffix.lower()
    if suffix in SOURCE_CONTEXT_EXTENSIONS:
        return True
    return suffix == "" and "/" in cleaned


def fetch_github_file_text(
    repo: str,
    ref: str,
    path: str,
    *,
    timeout: int = 60,
    max_chars: int = 12000,
) -> str:
    normalized = normalize_repo_file_path(repo, path)
    if not repo or not ref or not normalized:
        return ""
    url = f"https://raw.githubusercontent.com/{repo}/{quote(ref, safe='')}/{quote(normalized, safe='/')}"
    headers: dict[str, str] = {}
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    response = requests.get(url, headers=headers, timeout=timeout)
    if response.status_code == 404:
        return ""
    response.raise_for_status()
    return response.text[:max_chars]


def fetch_local_git_file_text(
    repo: str,
    ref: str,
    path: str,
    checkout_root: Path | None,
    *,
    timeout: int = 60,
    max_chars: int = 12000,
) -> str:
    repo_dir = resolve_local_repo_checkout(repo, checkout_root)
    normalized = normalize_repo_file_path(repo, path)
    if repo_dir is None or not ref or not normalized:
        return ""
    completed = subprocess.run(
        ["git", "-C", str(repo_dir), "show", f"{ref}:{normalized}"],
        capture_output=True,
        timeout=timeout,
        check=True,
    )
    stdout = completed.stdout.decode("utf-8", errors="replace") if isinstance(completed.stdout, bytes) else str(completed.stdout)
    return stdout[:max_chars]


def collect_repo_file_contexts(
    row: SweBenchRow,
    candidate_files: Sequence[str],
    *,
    max_files: int = 3,
    max_chars_per_file: int = 6000,
    max_candidate_attempts: int = 12,
    cache: dict[tuple[str, str, str], str] | None = None,
    checkout_root: Path | None = None,
) -> list[dict[str, Any]]:
    contexts: list[dict[str, Any]] = []
    seen: set[str] = set()
    attempts = 0
    successes = 0
    for raw_path in candidate_files:
        if successes >= max_files or attempts >= max_candidate_attempts:
            break
        normalized = normalize_repo_file_path(row.repo, str(raw_path))
        if not normalized or normalized in seen or not is_source_context_path(normalized):
            continue
        seen.add(normalized)
        attempts += 1
        key = (row.repo, row.base_commit, normalized)
        try:
            if cache is not None and key in cache:
                text = cache[key]
            else:
                text = fetch_local_git_file_text(
                    row.repo,
                    row.base_commit,
                    normalized,
                    checkout_root,
                    max_chars=max_chars_per_file,
                ) if checkout_root is not None else ""
                if not text:
                    text = fetch_github_file_text(
                        row.repo,
                        row.base_commit,
                        normalized,
                        max_chars=max_chars_per_file,
                    )
                if cache is not None:
                    cache[key] = text
            if text:
                successes += 1
                contexts.append(
                    {
                        "path": normalized,
                        "chars": int(len(text)),
                        "content": text,
                        "fetch_error": "",
                    }
                )
            else:
                contexts.append(
                    {
                        "path": normalized,
                        "chars": 0,
                        "content": "",
                        "fetch_error": "empty_or_not_found",
                    }
                )
        except Exception as exc:
            contexts.append(
                {
                    "path": normalized,
                    "chars": 0,
                    "content": "",
                    "fetch_error": type(exc).__name__,
                }
            )
        if len(contexts) >= max_files:
            break
    return contexts


def oracle_free_retrieval_candidates(
    row: SweBenchRow,
    *,
    repo_tree_files: Sequence[str] | None = None,
    top_k: int = 50,
) -> list[str]:
    visible_text = _visible_issue_text(row)
    mentioned = extract_path_mentions(visible_text)
    test_guesses = [_test_to_impl_guess(path) for path in mentioned + _safe_json_list(row.fail_to_pass)]
    seed_candidates = implementation_files([*mentioned, *test_guesses])
    tree_candidates = implementation_files(repo_tree_files or [])
    if not tree_candidates:
        return seed_candidates[:top_k]
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        candidate_texts = [path.replace("/", " ").replace("_", " ").replace("-", " ") for path in tree_candidates]
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True)
        matrix = vectorizer.fit_transform([visible_text, *candidate_texts])
        scores = cosine_similarity(matrix[0:1], matrix[1:]).ravel()
        direct_mentions = {path.lower() for path in mentioned}
        ranked = sorted(
            zip(tree_candidates, scores),
            key=lambda item: (
                item[0].lower() in direct_mentions,
                float(item[1]),
                -len(item[0]),
                item[0],
            ),
            reverse=True,
        )
        candidates = [path for path, _ in ranked]
    except Exception:
        candidates = sorted(tree_candidates)
    merged: list[str] = []
    for path in [*seed_candidates, *candidates]:
        if path in tree_candidates or not tree_candidates:
            if path not in merged:
                merged.append(path)
        if len(merged) >= top_k:
            break
    return merged


def retrieval_metrics(retrieved_files: Sequence[str], reference_files: Sequence[str] | None = None, *, top_k: int = 50) -> dict[str, Any]:
    del reference_files
    retrieved = [str(path).replace("\\", "/") for path in retrieved_files[:top_k]]
    return {
        "retrieved_candidate_count": int(len(retrieved)),
    }


def oracle_free_state_text(row: SweBenchRow, *, retrieved_files: Sequence[str], source_label: str = "current") -> str:
    tests = _safe_json_list(row.fail_to_pass)
    candidates = ", ".join(list(retrieved_files)[:10]) or "no retrieved implementation file candidates"
    return (
        f"Oracle-free SWE-bench state ({source_label} instance {row.instance_id}): "
        "constructed only from issue-visible text, repository-visible file paths, failing-test names and fixed retrieval. "
        f"Top retrieved implementation candidates: {candidates}. "
        f"Visible failing tests: {', '.join(tests[:4]) if tests else 'not listed'}. "
        "Select implementation files for the current issue only; do not use reference patch paths."
    )


def constraint_clean_file_hit_at_k(
    predicted: Sequence[str],
    reference: Sequence[str],
    *,
    k: int = 3,
    repo_tree_files: Sequence[str] | None = None,
) -> bool:
    selected = [str(path).strip().replace("\\", "/") for path in predicted[:k] if str(path).strip()]
    if not selected:
        return False
    if includes_test_file(list(selected)):
        return False
    if repo_tree_files:
        tree = {str(path).replace("\\", "/") for path in repo_tree_files}
        if any(path not in tree for path in selected):
            return False
    return bool(file_set_matches(list(selected), implementation_files(reference)))


def file_set_f1(predicted: Sequence[str], reference: Sequence[str]) -> float:
    pred = {str(path).strip().replace("\\", "/").lower() for path in predicted if str(path).strip()}
    ref = {str(path).strip().replace("\\", "/").lower() for path in implementation_files(reference)}
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    hits = 0
    for path in pred:
        if any(path == target or target.endswith("/" + path) or path.endswith("/" + target) for target in ref):
            hits += 1
    precision = hits / max(1, len(pred))
    recall = hits / max(1, len(ref))
    return float(2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0


def build_localization_prompt(row: SweBenchRow, *, condition: str, scrambled_row: SweBenchRow) -> str:
    problem = row.problem_statement[:3500]
    fail_tests = ", ".join(_safe_json_list(row.fail_to_pass)[:5]) or "benchmark FAIL_TO_PASS tests"
    pass_tests = ", ".join(_safe_json_list(row.pass_to_pass)[:5]) or "benchmark PASS_TO_PASS tests"
    state = "No extra decisive state field is provided."
    if condition == "decisive_state_current":
        state = decisive_state_text(row, source_label="current")
    elif condition == "decisive_state_only":
        state = decisive_state_text(row, source_label="current")
        problem = "Problem statement withheld for decisive-state-only localization test."
    elif condition == "scrambled_decisive_state":
        state = (
            decisive_state_text(scrambled_row, source_label="scrambled")
            + " This state may be stale or from another issue; localize files for the current issue only."
        )
    elif condition == "prior_only":
        problem = "Problem statement withheld for prior-only localization control."
        fail_tests = "withheld"
        pass_tests = "withheld"
        state = "No issue-specific decisive state is available."
    return (
        "Return exactly one compact JSON object and nothing else. No markdown.\n"
        "Required keys: target_files, rationale, confidence.\n"
        f"Dataset: {SWE_BENCH_DATASET} issue-to-file localization pilot.\n"
        f"Condition: {condition}\n"
        f"Current instance: {row.instance_id}\n"
        f"Repository: {row.repo}\n"
        "Task: identify the implementation file(s) that should be edited to fix the current issue.\n"
        "Hard constraints: target_files must be implementation files, not test files; do not localize to a file from another issue; use an empty list only if no file can be inferred.\n"
        f"Problem statement:\n{problem}\n"
        f"FAIL_TO_PASS tests: {fail_tests}\n"
        f"PASS_TO_PASS examples: {pass_tests}\n"
        f"State field: {state}\n"
        "Set target_files to a JSON array of at most three repository-relative file paths."
    )


def build_localization_prompt_v2(
    row: SweBenchRow,
    *,
    condition: str,
    scrambled_row: SweBenchRow,
    retrieved_files: Sequence[str],
    scrambled_retrieved_files: Sequence[str],
    prior_files: Sequence[str] | None = None,
    repo_tree_files: Sequence[str] | None = None,
    wrapper_arm: str = "raw",
) -> str:
    problem = row.problem_statement[:4500]
    fail_tests = ", ".join(_safe_json_list(row.fail_to_pass)[:8]) or "benchmark FAIL_TO_PASS tests"
    pass_tests = ", ".join(_safe_json_list(row.pass_to_pass)[:4]) or "benchmark PASS_TO_PASS tests"
    prompt_candidates = list(prior_files or retrieved_files) if condition == "prior_only" else list(retrieved_files)
    candidate_block = "\n".join(f"- {path}" for path in prompt_candidates[:50]) or "- no retrieved candidates"
    state = "No extra decisive state field is provided."
    if condition == "decisive_state_current":
        state = oracle_free_state_text(row, retrieved_files=retrieved_files, source_label="current")
    elif condition == "decisive_state_only":
        state = oracle_free_state_text(row, retrieved_files=retrieved_files, source_label="current")
        problem = "Problem statement withheld for oracle-free state-only localization test."
    elif condition == "scrambled_decisive_state":
        state = (
            oracle_free_state_text(scrambled_row, retrieved_files=scrambled_retrieved_files, source_label="same-repo scrambled")
            + " This state is intentionally from another same-repository issue; localize files for the current issue only."
        )
    elif condition == "prior_only":
        problem = "Problem statement withheld for prior-only/no-state localization control."
        fail_tests = "withheld"
        pass_tests = "withheld"
        state = (
            "Only a fixed same-repository candidate prior is visible; it may conflict with the current issue. "
            "No issue-specific decisive state is available."
        )
    elif condition == "irrelevant_cue":
        state = (
            "Irrelevant cue: a superficially plausible file may be mentioned in surrounding text, "
            "but it is not decisive. Use the current issue and retrieved candidates."
        )
    elif condition == "oracle_state_current":
        state = decisive_state_text(row, source_label="oracle-positive-control")
    elif condition == "oracle_state_only":
        state = decisive_state_text(row, source_label="oracle-positive-control")
        problem = "Problem statement withheld for oracle diagnostic state-only localization test."

    guard = ""
    if wrapper_arm == "binding_guard":
        guard = (
            "Before finalizing, apply this deterministic binding guard: reject any target that is test-only, "
            "nonexistent in the supplied candidate set, empty/no-op, or follows a stale/scrambled same-repo state. "
            "If no constraint-clean implementation file remains, return an empty target_files list and set defer=true.\n"
        )
    elif wrapper_arm == "compute_matched_repair":
        guard = (
            "Use one internal repair pass before finalizing: check for test-only, nonexistent, empty/no-op and stale-state-following targets, "
            "then revise target_files once if a violation is found. Keep the final JSON concise.\n"
        )

    tree_note = "Repository tree was available to the fixed retriever." if repo_tree_files else "Repository tree was not fetched; candidates are issue-visible retrieval outputs."
    return (
        "Return exactly one compact JSON object and nothing else. No markdown.\n"
        "Required keys: target_files, rationale, confidence, defer.\n"
        f"Dataset: {SWE_BENCH_DATASET} issue-to-file localization v2.\n"
        f"Condition: {condition}\n"
        f"Wrapper arm: {wrapper_arm}\n"
        f"Current instance: {row.instance_id}\n"
        f"Repository: {row.repo}\n"
        f"{tree_note}\n"
        "Task: identify up to three implementation file paths that should be edited to fix the current issue.\n"
        "Primary success criterion: constraint-clean implementation-file hit@3. Do not target tests, nonexistent paths, empty/no-op targets, or files from stale issue state.\n"
        f"{guard}"
        f"Problem statement:\n{problem}\n"
        f"FAIL_TO_PASS tests: {fail_tests}\n"
        f"PASS_TO_PASS examples: {pass_tests}\n"
        f"State field: {state}\n"
        f"Fixed candidate file prior (top {min(50, len(prompt_candidates))}):\n{candidate_block}\n"
        "Set target_files to a JSON array of at most three repository-relative implementation paths."
    )


def build_patch_generation_prompt_v2(
    row: SweBenchRow,
    *,
    retrieved_files: Sequence[str],
    repo_tree_files: Sequence[str] | None = None,
    file_contexts: Sequence[dict[str, Any]] | None = None,
    max_candidates: int = 50,
) -> str:
    problem = row.problem_statement[:7000]
    hints = row.hints_text[:1600].strip()
    fail_tests = ", ".join(_safe_json_list(row.fail_to_pass)[:10]) or "benchmark FAIL_TO_PASS tests"
    pass_tests = ", ".join(_safe_json_list(row.pass_to_pass)[:6]) or "benchmark PASS_TO_PASS tests"
    candidate_block = "\n".join(f"- {path}" for path in list(retrieved_files)[:max_candidates]) or "- no retrieved candidates"
    contexts = list(file_contexts or [])
    tree_note = "A repository tree was available to the fixed retriever." if repo_tree_files else "Repository tree was not fetched."
    if contexts:
        context_blocks = []
        for context in contexts:
            path = normalize_repo_file_path(row.repo, str(context.get("path", "")))
            content = str(context.get("content", ""))
            char_limit = int(context.get("chars", 0) or 0)
            if char_limit > 0:
                content = content[:char_limit]
            if not path or not content:
                continue
            context_blocks.append(f"File: {path}\n```python\n{content}\n```")
        file_context_block = (
            "Repository-visible source context from the base commit:\n" + "\n\n".join(context_blocks) + "\n"
            if context_blocks
            else "Repository-visible source context was requested but no candidate file content was retrieved.\n"
        )
    else:
        file_context_block = "No repository file contents are included; use only issue text and candidate paths.\n"
    hints_block = f"Hints text:\n{hints}\n" if hints else ""
    return (
        "Return exactly one compact JSON object and nothing else. No markdown.\n"
        "Required keys: model_patch, target_files, rationale, confidence, defer.\n"
        f"Dataset: {SWE_BENCH_DATASET} official execution slice patch-generation scaffold.\n"
        "Scope: generate a candidate unified-diff patch for official SWE-bench harness evaluation. "
        "This is not file-localization scoring.\n"
        f"Current instance: {row.instance_id}\n"
        f"Repository: {row.repo}\n"
        f"Base commit: {row.base_commit or 'listed in benchmark row'}\n"
        f"{tree_note}\n"
        "Oracle rule: use only issue-visible and repository-visible information in this prompt. "
        "Reference patch files and reference diff content are not provided.\n"
        "Patch constraints: model_patch must be a unified diff against repository-relative paths; "
        "do not edit tests; do not use a patch from another issue; if a safe patch cannot be written, "
        "set model_patch to an empty string, target_files to [], and defer to true.\n"
        f"Problem statement:\n{problem}\n"
        f"{hints_block}"
        f"FAIL_TO_PASS tests: {fail_tests}\n"
        f"PASS_TO_PASS examples: {pass_tests}\n"
        f"Oracle-free candidate implementation files (top {min(max_candidates, len(retrieved_files))}):\n{candidate_block}\n"
        f"{file_context_block}"
        "Set target_files to the implementation paths edited by model_patch, confidence to 0-100, "
        "and defer to false only when model_patch is non-empty."
    )


def _make_provider(config: ProjectConfig, *, model: str, base_url: str) -> ProviderClient:
    model_config = deepcopy(config)
    provider_cfg = model_config.agent.provider
    provider_cfg.mode = "openai_compatible"
    provider_cfg.model = model
    provider_cfg.base_url = base_url
    provider_cfg.endpoint_path = "/chat/completions"
    provider_cfg.api_key_env = "OPENAI_API_KEY"
    provider_cfg.temperature = 0.2
    provider_cfg.top_p = 1.0
    provider_cfg.json_mode = "json_object"
    provider_cfg.fallback_model = ""
    provider_cfg.fallback_base_url = ""
    provider_cfg.fallback_endpoint_path = ""
    provider_cfg.fallback_api_key_env = ""
    provider_cfg.fallback_json_mode = ""
    provider_cfg.provider_role = "real_task_predictive_validity"
    return ProviderClient(provider_cfg)


def smoke_check_swebench_v2_api_models(
    *,
    config: ProjectConfig,
    api_settings: ApiSettings,
    output_dir: Path,
    models: Iterable[str],
    dry_run: bool = False,
    hard_call_cap: int = 12,
) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    selected_models = [str(model).strip() for model in models if str(model).strip()]
    planned_calls = len(selected_models)
    manifest = {
        "experiment": "swebench_filelocalization_v2_api_model_smoke",
        "models": selected_models,
        "planned_remote_calls": int(planned_calls),
        "hard_call_cap": int(hard_call_cap),
        "base_url": api_settings.base_url,
        "dry_run": bool(dry_run),
        "scope_note": "One JSON-mode compatibility call per selected model; no API key is written.",
    }
    write_json(output_dir / "run_manifest.json", manifest)
    if planned_calls > int(hard_call_cap):
        raise RuntimeError(f"planned smoke calls {planned_calls} exceed hard-call cap {hard_call_cap}")

    rows: list[dict[str, Any]] = []
    if dry_run:
        for model in selected_models:
            rows.append(
                {
                    "model": model,
                    "status": "planned",
                    "remote_used": False,
                    "error_type": "",
                    "json_parse_ok": False,
                    "target_files_ok": False,
                    "latency_ms": 0,
                }
            )
    else:
        os.environ["OPENAI_API_KEY"] = api_settings.api_key
        defaults = {"target_files": [], "rationale": "default smoke fallback", "confidence": 0.0, "defer": False}
        system_prompt = "You are an API smoke-test agent. Return valid compact JSON only."
        user_prompt = (
            "Return exactly one JSON object with keys target_files, rationale, confidence, defer. "
            "Use target_files=[\"pkg/example.py\"], confidence=1, defer=false."
        )
        for model in selected_models:
            provider = _make_provider(config, model=model, base_url=api_settings.base_url)
            started = time.time()
            try:
                result = provider.chat_json_with_metadata(
                    system_prompt,
                    user_prompt,
                    defaults,
                    temperature_override=0.0,
                    top_p_override=1.0,
                )
                payload = result.payload if isinstance(result.payload, dict) else {}
                metadata = _metadata_subset(result.metadata)
                target_files = _coerce_file_list(payload.get("target_files", []))
                json_parse_ok = bool(
                    isinstance(payload, dict)
                    and set(["target_files", "rationale", "confidence", "defer"]).issubset(payload.keys())
                )
                target_files_ok = bool(target_files)
                rows.append(
                    {
                        "model": model,
                        "status": "pass"
                        if json_parse_ok and target_files_ok and not str(metadata.get("error_type", "")).strip()
                        else "needs_review",
                        "json_parse_ok": bool(json_parse_ok),
                        "target_files_ok": bool(target_files_ok),
                        "target_files": json.dumps(target_files, ensure_ascii=False),
                        **metadata,
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "model": model,
                        "status": "needs_review",
                        "remote_used": False,
                        "error_type": f"smoke_exception_{type(exc).__name__}",
                        "json_parse_ok": False,
                        "target_files_ok": False,
                        "target_files": "[]",
                        "latency_ms": int((time.time() - started) * 1000),
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "api_model_smoke_results.csv", index=False)
    write_parquet(frame, output_dir / "api_model_smoke_results.parquet")
    failures = frame[frame["status"].astype(str).ne("pass")] if not frame.empty else frame
    gate = {
        "status": "planned" if dry_run else ("pass" if failures.empty and not frame.empty else "needs_review"),
        "model_count": int(len(selected_models)),
        "planned_remote_calls": int(planned_calls),
        "observed_rows": int(len(frame)),
        "pass_rows": int(frame["status"].astype(str).eq("pass").sum()) if not frame.empty else 0,
        "needs_review_rows": int(frame["status"].astype(str).eq("needs_review").sum()) if not frame.empty else 0,
        "error_types": sorted(error for error in frame.get("error_type", pd.Series(dtype=str)).fillna("").astype(str).unique().tolist() if error),
        "gate_rule": "Each selected model must return a valid JSON-mode file-localization-shaped response without provider error.",
    }
    write_json(output_dir / "gate_status.json", gate)
    lines = [
        "# SWE-bench v2 API Model Smoke",
        "",
        f"- Status: `{gate['status']}`",
        f"- Models: `{', '.join(selected_models)}`",
        f"- Planned calls: `{planned_calls}`",
        "- Scope: one JSON-mode compatibility call per model; not a SWE-bench result.",
        "",
        "| model | status | remote | error | json ok | target ok |",
        "|---|---:|---:|---|---:|---:|",
    ]
    for row in frame.to_dict(orient="records"):
        lines.append(
            f"| {row.get('model')} | {row.get('status')} | {row.get('remote_used', False)} | "
            f"{row.get('error_type', '')} | {row.get('json_parse_ok', False)} | {row.get('target_files_ok', False)} |"
        )
    (output_dir / "api_model_smoke_report.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return {"results": frame, "gate_status": gate, "manifest": manifest}


def _metadata_subset(metadata: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "model",
        "request_temperature",
        "request_top_p",
        "fallback_used",
        "error_type",
        "latency_ms",
        "remote_used",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "retry_count",
    ]
    return {key: metadata.get(key, "") for key in keys}


def _auc_score(y_true: list[int], values: list[float], *, higher_positive: bool = True) -> float:
    clean = [(int(y), float(v)) for y, v in zip(y_true, values) if math.isfinite(float(v))]
    if len(clean) < 2 or len({y for y, _ in clean}) < 2:
        return float("nan")
    try:
        from sklearn.metrics import roc_auc_score

        y = [item[0] for item in clean]
        x = [item[1] if higher_positive else -item[1] for item in clean]
        return float(roc_auc_score(y, x))
    except Exception:
        return float("nan")


def _pearson(y_true: list[int], values: list[float]) -> float:
    clean = [(float(y), float(v)) for y, v in zip(y_true, values) if math.isfinite(float(v))]
    if len(clean) < 3:
        return float("nan")
    y = np.array([item[0] for item in clean], dtype=float)
    x = np.array([item[1] for item in clean], dtype=float)
    if np.isclose(y.std(), 0.0) or np.isclose(x.std(), 0.0):
        return float("nan")
    return float(np.corrcoef(y, x)[0, 1])


def _to_float(value: Any, default: float = float("nan")) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except Exception:
        return default


def _design_matrix(frame: pd.DataFrame, columns: Sequence[str], *, categorical: Sequence[str]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    numeric_cols = [col for col in columns if col not in categorical and col in frame.columns]
    if numeric_cols:
        numeric = frame[numeric_cols].apply(pd.to_numeric, errors="coerce")
        numeric = numeric.replace([np.inf, -np.inf], np.nan)
        numeric = numeric.fillna(numeric.median(numeric_only=True)).fillna(0.0)
        parts.append(numeric)
    categorical_cols = [col for col in categorical if col in frame.columns]
    if categorical_cols:
        parts.append(pd.get_dummies(frame[categorical_cols].fillna("missing").astype(str), drop_first=False))
    if not parts:
        return pd.DataFrame(index=frame.index)
    return pd.concat(parts, axis=1).astype(float)


def _logistic_auc(frame: pd.DataFrame, y: Sequence[int], columns: Sequence[str], *, categorical: Sequence[str]) -> float:
    clean_y = np.array([int(value) for value in y], dtype=int)
    if len(clean_y) < 4 or len(set(clean_y.tolist())) < 2:
        return float("nan")
    x = _design_matrix(frame, columns, categorical=categorical)
    if x.empty:
        return float("nan")
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler(with_mean=False)
        x_scaled = scaler.fit_transform(x.to_numpy(dtype=float))
        model = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")
        model.fit(x_scaled, clean_y)
        probabilities = model.predict_proba(x_scaled)[:, 1]
        return float(roc_auc_score(clean_y, probabilities))
    except Exception:
        return float("nan")


def _grouped_cv_logistic_auc(
    frame: pd.DataFrame,
    y: Sequence[int],
    columns: Sequence[str],
    *,
    categorical: Sequence[str],
    group_column: str = "instance_id",
    folds: int = 5,
) -> float:
    clean_y = np.array([int(value) for value in y], dtype=int)
    if len(clean_y) < 8 or len(set(clean_y.tolist())) < 2 or group_column not in frame.columns:
        return _logistic_auc(frame, y, columns, categorical=categorical)
    groups = frame[group_column].astype(str).to_numpy()
    unique_groups = np.array(sorted(set(groups)))
    if len(unique_groups) < 3:
        return _logistic_auc(frame, y, columns, categorical=categorical)
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import KFold
        from sklearn.preprocessing import StandardScaler

        n_splits = max(2, min(int(folds), len(unique_groups)))
        predictions = np.full(len(frame), np.nan, dtype=float)
        for train_group_idx, test_group_idx in KFold(n_splits=n_splits, shuffle=True, random_state=17).split(unique_groups):
            train_groups = set(unique_groups[train_group_idx])
            test_groups = set(unique_groups[test_group_idx])
            train_mask = np.array([group in train_groups for group in groups], dtype=bool)
            test_mask = np.array([group in test_groups for group in groups], dtype=bool)
            if len(set(clean_y[train_mask].tolist())) < 2:
                continue
            train_frame = frame.loc[train_mask].copy()
            test_frame = frame.loc[test_mask].copy()
            combined = pd.concat([train_frame, test_frame], axis=0)
            x_all = _design_matrix(combined, columns, categorical=categorical)
            x_train = x_all.iloc[: len(train_frame)].to_numpy(dtype=float)
            x_test = x_all.iloc[len(train_frame) :].to_numpy(dtype=float)
            scaler = StandardScaler(with_mean=False)
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)
            model = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")
            model.fit(x_train, clean_y[train_mask])
            predictions[test_mask] = model.predict_proba(x_test)[:, 1]
        valid = np.isfinite(predictions)
        if valid.sum() < 4 or len(set(clean_y[valid].tolist())) < 2:
            return _logistic_auc(frame, y, columns, categorical=categorical)
        return float(roc_auc_score(clean_y[valid], predictions[valid]))
    except Exception:
        return _logistic_auc(frame, y, columns, categorical=categorical)


def _state_binding_coefficient(
    frame: pd.DataFrame,
    y: Sequence[int],
    columns: Sequence[str],
    *,
    categorical: Sequence[str],
) -> float:
    if "state_binding_score" not in columns or "state_binding_score" not in frame.columns:
        return float("nan")
    clean_y = np.array([int(value) for value in y], dtype=int)
    if len(clean_y) < 4 or len(set(clean_y.tolist())) < 2:
        return float("nan")
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        x = _design_matrix(frame, columns, categorical=categorical)
        if "state_binding_score" not in x.columns:
            return float("nan")
        scaler = StandardScaler(with_mean=False)
        x_scaled = scaler.fit_transform(x.to_numpy(dtype=float))
        model = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")
        model.fit(x_scaled, clean_y)
        idx = list(x.columns).index("state_binding_score")
        return float(model.coef_[0][idx])
    except Exception:
        return float("nan")


def _issue_cluster_bootstrap_delta_auc(
    frame: pd.DataFrame,
    y: Sequence[int],
    baseline_columns: Sequence[str],
    augmented_columns: Sequence[str],
    *,
    categorical: Sequence[str],
    samples: int = 400,
    seed: int = 17,
) -> dict[str, Any]:
    if frame.empty or "instance_id" not in frame.columns:
        return {"samples": 0, "delta_auc_ci_low": float("nan"), "delta_auc_ci_high": float("nan")}
    unique_issues = sorted(frame["instance_id"].astype(str).unique().tolist())
    if len(unique_issues) < 3:
        return {"samples": 0, "delta_auc_ci_low": float("nan"), "delta_auc_ci_high": float("nan")}
    y_series = pd.Series([int(value) for value in y], index=frame.index)
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    for _ in range(int(samples)):
        sampled = rng.choice(unique_issues, size=len(unique_issues), replace=True)
        boot_parts: list[pd.DataFrame] = []
        boot_y: list[int] = []
        for draw_idx, issue in enumerate(sampled):
            subset = frame[frame["instance_id"].astype(str) == str(issue)].copy()
            subset.index = [f"{idx}_{draw_idx}" for idx in range(len(subset))]
            boot_parts.append(subset)
            boot_y.extend(y_series.loc[frame["instance_id"].astype(str) == str(issue)].astype(int).tolist())
        boot = pd.concat(boot_parts, axis=0).reset_index(drop=True)
        if len(set(boot_y)) < 2:
            continue
        base_auc = _logistic_auc(boot, boot_y, baseline_columns, categorical=categorical)
        aug_auc = _logistic_auc(boot, boot_y, augmented_columns, categorical=categorical)
        if math.isfinite(base_auc) and math.isfinite(aug_auc):
            deltas.append(float(aug_auc - base_auc))
    if not deltas:
        return {"samples": 0, "delta_auc_ci_low": float("nan"), "delta_auc_ci_high": float("nan")}
    return {
        "samples": int(len(deltas)),
        "delta_auc_ci_low": float(np.quantile(deltas, 0.025)),
        "delta_auc_ci_high": float(np.quantile(deltas, 0.975)),
    }


def summarize_localization_predictive_validity(traces: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if traces.empty:
        return pd.DataFrame(), pd.DataFrame(), {"status": "needs_review", "reason": "empty_traces"}
    task = traces[traces["condition"].isin(["task_only", "task_only_self_consistency"])].copy()
    rows: list[dict[str, Any]] = []
    for (model, instance_id), group in task.groupby(["model", "instance_id"], sort=True):
        file_keys = list(group["target_file_key"].astype(str))
        counts = pd.Series(file_keys).value_counts()
        top_count = int(counts.iloc[0]) if not counts.empty else 0
        second_count = int(counts.iloc[1]) if len(counts) > 1 else 0
        majority_key = str(counts.index[0]) if not counts.empty else ""
        majority_group = group[group["target_file_key"] == majority_key]
        all_for_pair = traces[(traces["model"] == model) & (traces["instance_id"] == instance_id)]
        decisive_current = all_for_pair[all_for_pair["condition"] == "decisive_state_current"]
        decisive_only = all_for_pair[all_for_pair["condition"] == "decisive_state_only"]
        scrambled = all_for_pair[all_for_pair["condition"] == "scrambled_decisive_state"]
        prior = all_for_pair[all_for_pair["condition"] == "prior_only"]
        decisive_current_correct = float(decisive_current["reference_file_match"].mean()) if not decisive_current.empty else float("nan")
        decisive_only_correct = float(decisive_only["reference_file_match"].mean()) if not decisive_only.empty else float("nan")
        scrambled_follow = float(scrambled["scrambled_file_match"].mean()) if not scrambled.empty else float("nan")
        prior_correct = float(prior["reference_file_match"].mean()) if not prior.empty else float("nan")
        parts = [decisive_current_correct, decisive_only_correct]
        if math.isfinite(scrambled_follow):
            parts.append(1.0 - scrambled_follow)
        valid_parts = [float(value) for value in parts if math.isfinite(float(value))]
        state_binding_score = float(np.mean(valid_parts)) if valid_parts else float("nan")
        rows.append(
            {
                "model": model,
                "instance_id": instance_id,
                "repo": group["repo"].iloc[0],
                "task_trials": int(len(group)),
                "task_success_rate": float(group["reference_file_match"].mean()),
                "task_success_majority": bool(majority_group["reference_file_match"].mean() >= 0.5) if not majority_group.empty else False,
                "hard_constraint_violation_rate": float(group["test_file_violation"].mean()),
                "memory_mismatch_rate": float(group["scrambled_file_match"].mean()),
                "empty_target_rate": float(group["empty_target"].mean()),
                "task_action_entropy": shannon_entropy(file_keys),
                "vote_margin": float((top_count - second_count) / max(1, len(group))),
                "mean_rationale_chars": float(group["rationale_chars"].mean()),
                "decisive_current_correct": decisive_current_correct,
                "decisive_only_correct": decisive_only_correct,
                "scrambled_follow_rate": scrambled_follow,
                "prior_correct": prior_correct,
                "state_binding_score": state_binding_score,
            }
        )
    instance_summary = pd.DataFrame(rows)
    y_success = [int(value) for value in instance_summary["task_success_majority"].astype(bool)]
    y_no_violation = [int(value <= 0.0) for value in instance_summary["hard_constraint_violation_rate"].fillna(1.0)]
    predictor_specs = [
        ("state_binding_score", True),
        ("task_action_entropy", False),
        ("vote_margin", True),
        ("mean_rationale_chars", True),
    ]
    predictor_rows: list[dict[str, Any]] = []
    for predictor, higher_positive in predictor_specs:
        values = [float(value) for value in instance_summary[predictor]]
        predictor_rows.append(
            {
                "predictor": predictor,
                "success_auc": _auc_score(y_success, values, higher_positive=higher_positive),
                "success_pearson": _pearson(y_success, values),
                "no_violation_auc": _auc_score(y_no_violation, values, higher_positive=higher_positive),
                "higher_predicts_success": bool(higher_positive),
            }
        )
    predictor_summary = pd.DataFrame(predictor_rows)
    state_auc = float(
        predictor_summary.loc[predictor_summary["predictor"] == "state_binding_score", "success_auc"].iloc[0]
    )
    baseline_auc = float(
        predictor_summary.loc[predictor_summary["predictor"] != "state_binding_score", "success_auc"].max(skipna=True)
    )
    gate = {
        "status": "pass"
        if math.isfinite(state_auc) and (not math.isfinite(baseline_auc) or state_auc >= baseline_auc)
        else "needs_review",
        "rows": int(len(traces)),
        "model_instance_rows": int(len(instance_summary)),
        "success_positive_rows": int(sum(y_success)),
        "success_negative_rows": int(len(y_success) - sum(y_success)),
        "state_binding_success_auc": state_auc,
        "best_baseline_success_auc": baseline_auc,
        "gate_rule": "state-binding score AUC for SWE-bench Lite file-localization success is defined and at least as high as entropy, vote-margin and rationale-length baselines",
    }
    return instance_summary, predictor_summary, gate


def summarize_localization_predictive_validity_v2(
    traces: pd.DataFrame,
    *,
    bootstrap_samples: int = 400,
    delta_auc_gate: float = 0.03,
    noninferiority_margin: float = -0.03,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if traces.empty:
        return pd.DataFrame(), pd.DataFrame(), {"status": "needs_review", "reason": "empty_traces"}
    raw_task = traces[
        (traces.get("wrapper_arm", "raw") == "raw")
        & traces["condition"].isin(["task_only", "task_only_self_consistency"])
    ].copy()
    if raw_task.empty:
        return pd.DataFrame(), pd.DataFrame(), {"status": "needs_review", "reason": "empty_raw_task_traces"}
    rows: list[dict[str, Any]] = []
    for (model, instance_id), group in raw_task.groupby(["model", "instance_id"], sort=True):
        all_for_pair = traces[(traces["model"] == model) & (traces["instance_id"] == instance_id)]
        raw_key_counts = pd.Series(group["target_file_key"].astype(str)).value_counts()
        top_count = int(raw_key_counts.iloc[0]) if not raw_key_counts.empty else 0
        second_count = int(raw_key_counts.iloc[1]) if len(raw_key_counts) > 1 else 0
        current = all_for_pair[(all_for_pair["condition"] == "decisive_state_current") & (all_for_pair.get("wrapper_arm", "raw") == "raw")]
        decisive_only = all_for_pair[(all_for_pair["condition"] == "decisive_state_only") & (all_for_pair.get("wrapper_arm", "raw") == "raw")]
        scrambled = all_for_pair[(all_for_pair["condition"] == "scrambled_decisive_state") & (all_for_pair.get("wrapper_arm", "raw") == "raw")]
        prior = all_for_pair[(all_for_pair["condition"] == "prior_only") & (all_for_pair.get("wrapper_arm", "raw") == "raw")]
        c_current = float(current["constraint_clean_hit_at3"].mean()) if not current.empty and "constraint_clean_hit_at3" in current else float("nan")
        c_decisive = (
            float(decisive_only["constraint_clean_hit_at3"].mean())
            if not decisive_only.empty and "constraint_clean_hit_at3" in decisive_only
            else float("nan")
        )
        scrambled_follow = float(scrambled["scrambled_file_match"].mean()) if not scrambled.empty else float("nan")
        if "prior_file_match" in prior.columns:
            prior_follow = float(prior["prior_file_match"].mean()) if not prior.empty else float("nan")
        else:
            prior_follow = float(prior["reference_file_match"].mean()) if not prior.empty else float("nan")
        r_scrambled = 1.0 - scrambled_follow if math.isfinite(scrambled_follow) else float("nan")
        r_prior = 1.0 - prior_follow if math.isfinite(prior_follow) else float("nan")
        csb_parts = [c_current, c_decisive, r_scrambled, r_prior]
        csb_component_count = int(sum(math.isfinite(float(value)) for value in csb_parts))
        state_binding_score = float(sum(csb_parts) / 4.0) if csb_component_count == 4 else float("nan")
        confidence = pd.to_numeric(group.get("confidence", pd.Series(dtype=float)), errors="coerce")
        retrieved_candidate_count = pd.to_numeric(group.get("retrieved_candidate_count", pd.Series([np.nan])), errors="coerce")
        path_counts = [len(_coerce_file_list(value)) for value in group["target_files"]] if "target_files" in group else []
        guard = all_for_pair[all_for_pair.get("wrapper_arm", "raw") == "binding_guard"]
        repair = all_for_pair[all_for_pair.get("wrapper_arm", "raw") == "compute_matched_repair"]
        implementation_hit = _boolean_series(group, "reference_file_match")
        task_only_hard_violation = _task_only_hard_violation_mask(group)
        task_only_constraint_clean_hit = implementation_hit & ~task_only_hard_violation
        raw_hit = float(implementation_hit.mean())
        raw_clean_hit = float(task_only_constraint_clean_hit.mean())
        guard_hit = (
            float(_boolean_series(guard, "reference_file_match").mean())
            if not guard.empty and "reference_file_match" in guard
            else float("nan")
        )
        raw_violation = float(task_only_hard_violation.mean())
        guard_violation = (
            float(_task_only_hard_violation_mask(guard).mean())
            if not guard.empty
            else float("nan")
        )
        rows.append(
            {
                "model": model,
                "instance_id": instance_id,
                "repo": group["repo"].iloc[0],
                "task_trials": int(len(group)),
                "implementation_hit_at3_rate": raw_hit,
                "implementation_hit_at3_majority": bool(raw_hit >= 0.5),
                "constraint_clean_hit_at3_rate": raw_clean_hit,
                "constraint_clean_hit_at3_majority": bool(raw_clean_hit >= 0.5),
                "hard_constraint_violation_rate": raw_violation,
                "test_only_target_rate": float(group["test_file_violation"].mean()) if "test_file_violation" in group else float("nan"),
                "candidate_absent_violation_rate": float(group["candidate_absent_violation"].mean())
                if "candidate_absent_violation" in group
                else float("nan"),
                "nonexistent_path_violation_rate": float(group["nonexistent_path_violation"].mean())
                if "nonexistent_path_violation" in group
                else float("nan"),
                "repo_nonexistent_path_violation_rate": float(group["repo_nonexistent_path_violation"].mean())
                if "repo_nonexistent_path_violation" in group
                else float("nan"),
                "scrambled_state_following_rate": float(group["scrambled_file_match"].mean()) if "scrambled_file_match" in group else float("nan"),
                "prior_only_following_rate": prior_follow,
                "empty_target_rate": float(group["empty_target"].mean()) if "empty_target" in group else float("nan"),
                "defer_rate": float(group["deferred"].mean()) if "deferred" in group else float("nan"),
                "file_set_f1": float(group["file_set_f1"].mean()) if "file_set_f1" in group else float("nan"),
                "exact_file_set_match_rate": float(group["exact_file_set_match"].mean()) if "exact_file_set_match" in group else float("nan"),
                "task_action_entropy": shannon_entropy(list(group["target_file_key"].astype(str))),
                "file_set_entropy": shannon_entropy(list(group["target_file_key"].astype(str))),
                "path_count_entropy": shannon_entropy([str(count) for count in path_counts]) if path_counts else float("nan"),
                "top_file_set_agreement": float(top_count / max(1, len(group))),
                "vote_margin": float((top_count - second_count) / max(1, len(group))),
                "mean_rationale_chars": float(group["rationale_chars"].mean()),
                "mean_confidence": float(confidence.mean()) if not confidence.empty else float("nan"),
                "issue_length": float(group["problem_chars"].mean()) if "problem_chars" in group else float("nan"),
                "retrieved_candidate_count": float(retrieved_candidate_count.mean()) if not retrieved_candidate_count.empty else float("nan"),
                "decisive_current_clean_hit": c_current,
                "decisive_only_clean_hit": c_decisive,
                "scrambled_resistance": r_scrambled,
                "prior_resistance": r_prior,
                "csb_component_count": csb_component_count,
                "csb_components_complete": bool(csb_component_count == 4),
                "scrambled_follow_rate": scrambled_follow,
                "prior_follow_rate": prior_follow,
                "state_binding_score": state_binding_score,
                "binding_guard_hit_at3_rate": guard_hit,
                "binding_guard_violation_rate": guard_violation,
                "compute_matched_repair_hit_at3_rate": float(_boolean_series(repair, "reference_file_match").mean())
                if not repair.empty and "reference_file_match" in repair
                else float("nan"),
            }
        )
    instance_summary = pd.DataFrame(rows)
    y_success = [int(value) for value in instance_summary["implementation_hit_at3_majority"].astype(bool)]
    y_constraint_clean = [int(value) for value in instance_summary["constraint_clean_hit_at3_majority"].astype(bool)]
    y_no_violation = [int(value <= 0.0) for value in instance_summary["hard_constraint_violation_rate"].fillna(1.0)]
    predictor_specs = [
        ("state_binding_score", True),
        ("task_action_entropy", False),
        ("file_set_entropy", False),
        ("path_count_entropy", False),
        ("top_file_set_agreement", True),
        ("vote_margin", True),
        ("mean_rationale_chars", True),
        ("mean_confidence", True),
        ("issue_length", False),
        ("retrieved_candidate_count", True),
    ]
    predictor_rows: list[dict[str, Any]] = []
    for predictor, higher_positive in predictor_specs:
        if predictor not in instance_summary.columns:
            continue
        values = [float(value) for value in instance_summary[predictor]]
        predictor_rows.append(
            {
                "predictor": predictor,
                "success_auc": _auc_score(y_success, values, higher_positive=higher_positive),
                "success_pearson": _pearson(y_success, values),
                "no_violation_auc": _auc_score(y_no_violation, values, higher_positive=higher_positive),
                "constraint_clean_auc": _auc_score(y_constraint_clean, values, higher_positive=higher_positive),
                "higher_predicts_success": bool(higher_positive),
            }
        )
    predictor_summary = pd.DataFrame(predictor_rows)
    baseline_columns = [
        "model",
        "repo",
        "retrieved_candidate_count",
        "issue_length",
        "task_action_entropy",
        "file_set_entropy",
        "path_count_entropy",
        "top_file_set_agreement",
        "vote_margin",
        "mean_rationale_chars",
        "mean_confidence",
    ]
    augmented_columns = [*baseline_columns, "state_binding_score"]
    categorical = ["model", "repo"]
    baseline_auc = _grouped_cv_logistic_auc(instance_summary, y_success, baseline_columns, categorical=categorical)
    augmented_auc = _grouped_cv_logistic_auc(instance_summary, y_success, augmented_columns, categorical=categorical)
    delta_auc = float(augmented_auc - baseline_auc) if math.isfinite(baseline_auc) and math.isfinite(augmented_auc) else float("nan")
    bootstrap = _issue_cluster_bootstrap_delta_auc(
        instance_summary,
        y_success,
        baseline_columns,
        augmented_columns,
        categorical=categorical,
        samples=bootstrap_samples,
    )
    csb_coef = _state_binding_coefficient(instance_summary, y_success, augmented_columns, categorical=categorical)
    no_violation_baseline_auc = _grouped_cv_logistic_auc(
        instance_summary, y_no_violation, baseline_columns, categorical=categorical
    )
    no_violation_augmented_auc = _grouped_cv_logistic_auc(
        instance_summary, y_no_violation, augmented_columns, categorical=categorical
    )
    no_violation_delta_auc = (
        float(no_violation_augmented_auc - no_violation_baseline_auc)
        if math.isfinite(no_violation_baseline_auc) and math.isfinite(no_violation_augmented_auc)
        else float("nan")
    )
    constraint_clean_baseline_auc = _grouped_cv_logistic_auc(
        instance_summary, y_constraint_clean, baseline_columns, categorical=categorical
    )
    constraint_clean_augmented_auc = _grouped_cv_logistic_auc(
        instance_summary, y_constraint_clean, augmented_columns, categorical=categorical
    )
    constraint_clean_delta_auc = (
        float(constraint_clean_augmented_auc - constraint_clean_baseline_auc)
        if math.isfinite(constraint_clean_baseline_auc) and math.isfinite(constraint_clean_augmented_auc)
        else float("nan")
    )
    leave_repo_deltas: list[float] = []
    for repo in sorted(instance_summary["repo"].astype(str).unique()):
        subset = instance_summary[instance_summary["repo"].astype(str) != repo].copy()
        if len(subset) < 4:
            continue
        y_subset = [int(value) for value in subset["implementation_hit_at3_majority"].astype(bool)]
        base = _grouped_cv_logistic_auc(subset, y_subset, baseline_columns, categorical=categorical)
        aug = _grouped_cv_logistic_auc(subset, y_subset, augmented_columns, categorical=categorical)
        if math.isfinite(base) and math.isfinite(aug):
            leave_repo_deltas.append(float(aug - base))
    leave_model_deltas: list[float] = []
    for model in sorted(instance_summary["model"].astype(str).unique()):
        subset = instance_summary[instance_summary["model"].astype(str) != model].copy()
        if len(subset) < 4:
            continue
        y_subset = [int(value) for value in subset["implementation_hit_at3_majority"].astype(bool)]
        base = _grouped_cv_logistic_auc(subset, y_subset, baseline_columns, categorical=categorical)
        aug = _grouped_cv_logistic_auc(subset, y_subset, augmented_columns, categorical=categorical)
        if math.isfinite(base) and math.isfinite(aug):
            leave_model_deltas.append(float(aug - base))
    within_model_deltas: list[float] = []
    for model in sorted(instance_summary["model"].astype(str).unique()):
        subset = instance_summary[instance_summary["model"].astype(str) == model].copy()
        if len(subset) < 8:
            continue
        y_subset = [int(value) for value in subset["implementation_hit_at3_majority"].astype(bool)]
        base = _grouped_cv_logistic_auc(subset, y_subset, baseline_columns, categorical=categorical)
        aug = _grouped_cv_logistic_auc(subset, y_subset, augmented_columns, categorical=categorical)
        if math.isfinite(base) and math.isfinite(aug):
            within_model_deltas.append(float(aug - base))
    within_repo_deltas: list[float] = []
    for repo in sorted(instance_summary["repo"].astype(str).unique()):
        subset = instance_summary[instance_summary["repo"].astype(str) == repo].copy()
        if len(subset) < 8:
            continue
        y_subset = [int(value) for value in subset["implementation_hit_at3_majority"].astype(bool)]
        base = _grouped_cv_logistic_auc(subset, y_subset, baseline_columns, categorical=categorical)
        aug = _grouped_cv_logistic_auc(subset, y_subset, augmented_columns, categorical=categorical)
        if math.isfinite(base) and math.isfinite(aug):
            within_repo_deltas.append(float(aug - base))
    state_auc = float(
        predictor_summary.loc[predictor_summary["predictor"] == "state_binding_score", "success_auc"].iloc[0]
    ) if not predictor_summary.empty and (predictor_summary["predictor"] == "state_binding_score").any() else float("nan")
    best_simple_baseline_auc = float(
        predictor_summary.loc[predictor_summary["predictor"] != "state_binding_score", "success_auc"].max(skipna=True)
    ) if not predictor_summary.empty else float("nan")
    raw_violation_mean = float(instance_summary["hard_constraint_violation_rate"].mean())
    guard_violation_mean = float(instance_summary["binding_guard_violation_rate"].mean(skipna=True))
    raw_hit_mean = float(instance_summary["implementation_hit_at3_rate"].mean())
    guard_hit_mean = float(instance_summary["binding_guard_hit_at3_rate"].mean(skipna=True))
    csb_complete_rows = int(instance_summary["csb_components_complete"].fillna(False).astype(bool).sum())
    csb_complete_rate = float(csb_complete_rows / max(1, len(instance_summary)))
    wrapper_delta_violation = guard_violation_mean - raw_violation_mean if math.isfinite(guard_violation_mean) else float("nan")
    wrapper_delta_hit = guard_hit_mean - raw_hit_mean if math.isfinite(guard_hit_mean) else float("nan")
    primary_pass = (
        csb_complete_rows == len(instance_summary)
        and math.isfinite(delta_auc)
        and delta_auc >= float(delta_auc_gate)
        and math.isfinite(float(bootstrap.get("delta_auc_ci_low", float("nan"))))
        and float(bootstrap.get("delta_auc_ci_low", float("nan"))) > 0
        and math.isfinite(csb_coef)
        and csb_coef > 0
        and (not leave_repo_deltas or sum(delta > 0 for delta in leave_repo_deltas) >= math.ceil(len(leave_repo_deltas) / 2))
        and (not leave_model_deltas or sum(delta > 0 for delta in leave_model_deltas) >= math.ceil(len(leave_model_deltas) / 2))
        and (not within_model_deltas or sum(delta > 0 for delta in within_model_deltas) >= math.ceil(len(within_model_deltas) / 2))
        and (not within_repo_deltas or sum(delta > 0 for delta in within_repo_deltas) >= math.ceil(len(within_repo_deltas) / 2))
    )
    wrapper_pass = (
        math.isfinite(wrapper_delta_violation)
        and wrapper_delta_violation < 0
        and math.isfinite(wrapper_delta_hit)
        and wrapper_delta_hit >= float(noninferiority_margin)
    )
    gate = {
        "status": "pass" if primary_pass else "needs_review",
        "rows": int(len(traces)),
        "model_instance_rows": int(len(instance_summary)),
        "csb_complete_rows": csb_complete_rows,
        "csb_complete_rate": csb_complete_rate,
        "success_positive_rows": int(sum(y_success)),
        "success_negative_rows": int(len(y_success) - sum(y_success)),
        "primary_outcome": "task_only_implementation_file_hit_at3",
        "secondary_outcome": "task_only_hard_constraint_violation",
        "composite_secondary_outcome": "task_only_constraint_clean_implementation_file_hit_at3",
        "baseline_incremental_auc": baseline_auc,
        "augmented_csb_auc": augmented_auc,
        "delta_auc": delta_auc,
        "delta_auc_gate": float(delta_auc_gate),
        "delta_auc_ci_low": bootstrap.get("delta_auc_ci_low", float("nan")),
        "delta_auc_ci_high": bootstrap.get("delta_auc_ci_high", float("nan")),
        "bootstrap_issue_cluster_samples": int(bootstrap.get("samples", 0)),
        "state_binding_logistic_coefficient": csb_coef,
        "leave_one_repo_positive": int(sum(delta > 0 for delta in leave_repo_deltas)),
        "leave_one_repo_total": int(len(leave_repo_deltas)),
        "leave_one_model_positive": int(sum(delta > 0 for delta in leave_model_deltas)),
        "leave_one_model_total": int(len(leave_model_deltas)),
        "within_model_positive": int(sum(delta > 0 for delta in within_model_deltas)),
        "within_model_total": int(len(within_model_deltas)),
        "within_repo_positive": int(sum(delta > 0 for delta in within_repo_deltas)),
        "within_repo_total": int(len(within_repo_deltas)),
        "state_binding_success_auc": state_auc,
        "best_simple_baseline_success_auc": best_simple_baseline_auc,
        "no_violation_positive_rows": int(sum(y_no_violation)),
        "no_violation_negative_rows": int(len(y_no_violation) - sum(y_no_violation)),
        "no_violation_baseline_auc": no_violation_baseline_auc,
        "no_violation_augmented_csb_auc": no_violation_augmented_auc,
        "no_violation_delta_auc": no_violation_delta_auc,
        "constraint_clean_positive_rows": int(sum(y_constraint_clean)),
        "constraint_clean_negative_rows": int(len(y_constraint_clean) - sum(y_constraint_clean)),
        "constraint_clean_baseline_auc": constraint_clean_baseline_auc,
        "constraint_clean_augmented_csb_auc": constraint_clean_augmented_auc,
        "constraint_clean_delta_auc": constraint_clean_delta_auc,
        "baseline_predictor_columns": baseline_columns,
        "augmented_predictor_columns": augmented_columns,
        "gold_derived_predictors_excluded": [
            "reference_implementation_file_count",
            "retriever_first_gold_rank",
            "retriever_recall_at_k",
        ],
        "wrapper_delta_hard_violation": wrapper_delta_violation,
        "wrapper_delta_hit_at3": wrapper_delta_hit,
        "wrapper_noninferiority_margin": float(noninferiority_margin),
        "wrapper_status": "pass" if wrapper_pass else "needs_review",
        "gate_rule": "primary: the gold-free full non-CSB baseline plus CSB improves issue-clustered grouped-CV AUC for task-only implementation-file hit@3 by the preregistered margin with issue-cluster bootstrap CI above zero, positive coefficient, leave-one-repo/model robustness, and majority-positive within-model/repo sensitivity; secondary outcomes report task-only hard-constraint violations and task-only constraint-clean hit@3",
    }
    return instance_summary, predictor_summary, gate


def summarize_predictive_validity(traces: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if traces.empty:
        return pd.DataFrame(), pd.DataFrame(), {"status": "needs_review", "reason": "empty_traces"}
    task = traces[traces["condition"].isin(["task_only", "task_only_self_consistency"])].copy()
    rows: list[dict[str, Any]] = []
    for (model, instance_id), group in task.groupby(["model", "instance_id"], sort=True):
        actions = list(group["final_action"].astype(str))
        counts = pd.Series(actions).value_counts()
        top_count = int(counts.iloc[0]) if not counts.empty else 0
        second_count = int(counts.iloc[1]) if len(counts) > 1 else 0
        majority_action = str(counts.index[0]) if not counts.empty else "INVALID"
        all_for_pair = traces[(traces["model"] == model) & (traces["instance_id"] == instance_id)]
        decisive_current = all_for_pair[all_for_pair["condition"] == "decisive_state_current"]
        decisive_only = all_for_pair[all_for_pair["condition"] == "decisive_state_only"]
        scrambled = all_for_pair[all_for_pair["condition"] == "scrambled_decisive_state"]
        prior = all_for_pair[all_for_pair["condition"] == "candidate_prior_only"]
        decisive_current_correct = float(decisive_current["expected_action_match"].mean()) if not decisive_current.empty else float("nan")
        decisive_only_correct = float(decisive_only["expected_action_match"].mean()) if not decisive_only.empty else float("nan")
        scrambled_follow = float(scrambled["follows_scrambled_action"].mean()) if not scrambled.empty else float("nan")
        prior_correct = float(prior["expected_action_match"].mean()) if not prior.empty else float("nan")
        parts = [decisive_current_correct, decisive_only_correct]
        if math.isfinite(scrambled_follow):
            parts.append(1.0 - scrambled_follow)
        valid_parts = [float(value) for value in parts if math.isfinite(float(value))]
        state_binding_score = float(np.mean(valid_parts)) if valid_parts else float("nan")
        rows.append(
            {
                "model": model,
                "instance_id": instance_id,
                "repo": group["repo"].iloc[0],
                "task_trials": int(len(group)),
                "task_success_rate": float(group["expected_action_match"].mean()),
                "task_success_majority": bool(group[group["final_action"] == majority_action]["expected_action_match"].mean() >= 0.5)
                if majority_action != "INVALID"
                else False,
                "hard_constraint_violation_rate": float(group["hard_constraint_violation"].mean()),
                "memory_mismatch_rate": float(group["memory_mismatch_failure"].mean()),
                "no_op_rate": float(group["no_op_failure"].mean()),
                "task_action_entropy": shannon_entropy(actions),
                "vote_margin": float((top_count - second_count) / max(1, len(group))),
                "mean_rationale_chars": float(group["rationale_chars"].mean()),
                "decisive_current_correct": decisive_current_correct,
                "decisive_only_correct": decisive_only_correct,
                "scrambled_follow_rate": scrambled_follow,
                "prior_correct": prior_correct,
                "state_binding_score": state_binding_score,
            }
        )
    instance_summary = pd.DataFrame(rows)

    y_success = [int(value) for value in instance_summary["task_success_majority"].astype(bool)]
    y_no_violation = [int(value <= 0.0) for value in instance_summary["hard_constraint_violation_rate"].fillna(1.0)]
    predictor_specs = [
        ("state_binding_score", True),
        ("task_action_entropy", False),
        ("vote_margin", True),
        ("mean_rationale_chars", True),
    ]
    predictor_rows: list[dict[str, Any]] = []
    for predictor, higher_positive in predictor_specs:
        values = [float(value) for value in instance_summary[predictor]]
        predictor_rows.append(
            {
                "predictor": predictor,
                "success_auc": _auc_score(y_success, values, higher_positive=higher_positive),
                "success_pearson": _pearson(y_success, values),
                "no_violation_auc": _auc_score(y_no_violation, values, higher_positive=higher_positive),
                "higher_predicts_success": bool(higher_positive),
            }
        )
    predictor_summary = pd.DataFrame(predictor_rows)
    state_auc = float(
        predictor_summary.loc[predictor_summary["predictor"] == "state_binding_score", "success_auc"].iloc[0]
    )
    baseline_auc = float(
        predictor_summary.loc[predictor_summary["predictor"] != "state_binding_score", "success_auc"].max(skipna=True)
    )
    gate = {
        "status": "pass"
        if math.isfinite(state_auc) and (not math.isfinite(baseline_auc) or state_auc >= baseline_auc)
        else "needs_review",
        "rows": int(len(traces)),
        "model_instance_rows": int(len(instance_summary)),
        "success_positive_rows": int(sum(y_success)),
        "success_negative_rows": int(len(y_success) - sum(y_success)),
        "state_binding_success_auc": state_auc,
        "best_baseline_success_auc": baseline_auc,
        "gate_rule": "state-binding score AUC for task-only implementation-file hit@3 is defined and at least as high as entropy, vote-margin and rationale-length baselines",
    }
    return instance_summary, predictor_summary, gate


def write_report(
    output_dir: Path,
    *,
    manifest: dict[str, Any],
    instance_summary: pd.DataFrame,
    predictor_summary: pd.DataFrame,
    gate: dict[str, Any],
) -> None:
    lines = [
        "# SWE-bench Lite Predictive Validity Pilot",
        "",
        f"- Gate: `{gate.get('status')}`",
        f"- Dataset: `{SWE_BENCH_DATASET}`",
        f"- Models: `{', '.join(manifest.get('models', []))}`",
        f"- Planned calls: `{manifest.get('planned_remote_calls')}`",
        f"- Gate rule: {gate.get('gate_rule')}",
        "",
        "This is an issue-to-patch action-selection pilot using real SWE-bench Lite issue records. "
        "It is not a full SWE-bench repository checkout, patch application or test-execution evaluation.",
        "",
        "## Predictor Summary",
        "",
    ]
    if not predictor_summary.empty:
        lines.extend(["| predictor | success AUC | success Pearson | no-violation AUC |", "|---|---:|---:|---:|"])
        for row in predictor_summary.to_dict(orient="records"):
            lines.append(
                f"| {row['predictor']} | {float(row['success_auc']):.3f} | "
                f"{float(row['success_pearson']):.3f} | {float(row['no_violation_auc']):.3f} |"
            )
    lines.extend(["", "## Model-Instance Summary", ""])
    if not instance_summary.empty:
        by_model = (
            instance_summary.groupby("model", sort=True)
            .agg(
                model_instance_rows=("instance_id", "size"),
                task_success_majority=("task_success_majority", "mean"),
                task_success_rate=("task_success_rate", "mean"),
                hard_constraint_violation_rate=("hard_constraint_violation_rate", "mean"),
                memory_mismatch_rate=("memory_mismatch_rate", "mean"),
                state_binding_score=("state_binding_score", "mean"),
                task_action_entropy=("task_action_entropy", "mean"),
                vote_margin=("vote_margin", "mean"),
            )
            .reset_index()
        )
        lines.extend(
            [
                "| model | rows | majority success | success rate | hard violation | memory mismatch | binding score | entropy | vote margin |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in by_model.to_dict(orient="records"):
            lines.append(
                f"| {row['model']} | {int(row['model_instance_rows'])} | {float(row['task_success_majority']):.3f} | "
                f"{float(row['task_success_rate']):.3f} | {float(row['hard_constraint_violation_rate']):.3f} | "
                f"{float(row['memory_mismatch_rate']):.3f} | {float(row['state_binding_score']):.3f} | "
                f"{float(row['task_action_entropy']):.3f} | {float(row['vote_margin']):.3f} |"
            )
    lines.extend(["", "Interpretation: use only as bounded real-task predictive-validity evidence.", ""])
    (output_dir / "swebench_predictive_validity_report.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_localization_report(
    output_dir: Path,
    *,
    manifest: dict[str, Any],
    instance_summary: pd.DataFrame,
    predictor_summary: pd.DataFrame,
    gate: dict[str, Any],
) -> None:
    lines = [
        "# SWE-bench Lite File-Localization Predictive Validity Pilot",
        "",
        f"- Gate: `{gate.get('status')}`",
        f"- Dataset: `{SWE_BENCH_DATASET}`",
        f"- Models: `{', '.join(manifest.get('models', []))}`",
        f"- Planned calls: `{manifest.get('planned_remote_calls')}`",
        f"- Gate rule: {gate.get('gate_rule')}",
        "",
        "This pilot uses real SWE-bench Lite issue records and reference implementation-patch files. "
        "The outcome is issue-to-file localization success, not full patch generation or test execution.",
        "",
        "## Predictor Summary",
        "",
    ]
    if not predictor_summary.empty:
        lines.extend(["| predictor | success AUC | success Pearson | no-violation AUC |", "|---|---:|---:|---:|"])
        for row in predictor_summary.to_dict(orient="records"):
            lines.append(
                f"| {row['predictor']} | {float(row['success_auc']):.3f} | "
                f"{float(row['success_pearson']):.3f} | {float(row['no_violation_auc']):.3f} |"
            )
    lines.extend(["", "## Model-Instance Summary", ""])
    if not instance_summary.empty:
        by_model = (
            instance_summary.groupby("model", sort=True)
            .agg(
                model_instance_rows=("instance_id", "size"),
                task_success_majority=("task_success_majority", "mean"),
                task_success_rate=("task_success_rate", "mean"),
                hard_constraint_violation_rate=("hard_constraint_violation_rate", "mean"),
                memory_mismatch_rate=("memory_mismatch_rate", "mean"),
                state_binding_score=("state_binding_score", "mean"),
                task_action_entropy=("task_action_entropy", "mean"),
                vote_margin=("vote_margin", "mean"),
            )
            .reset_index()
        )
        lines.extend(
            [
                "| model | rows | majority success | success rate | test-file violation | scrambled-file match | binding score | entropy | vote margin |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in by_model.to_dict(orient="records"):
            lines.append(
                f"| {row['model']} | {int(row['model_instance_rows'])} | {float(row['task_success_majority']):.3f} | "
                f"{float(row['task_success_rate']):.3f} | {float(row['hard_constraint_violation_rate']):.3f} | "
                f"{float(row['memory_mismatch_rate']):.3f} | {float(row['state_binding_score']):.3f} | "
                f"{float(row['task_action_entropy']):.3f} | {float(row['vote_margin']):.3f} |"
            )
    lines.extend(["", "Interpretation: bounded real-task predictive-validity evidence for SWE-bench Lite file localization only.", ""])
    (output_dir / "swebench_file_localization_predictive_validity_report.md").write_text(
        "\n".join(lines).rstrip() + "\n",
        encoding="utf-8",
    )


def write_localization_v2_report(
    output_dir: Path,
    *,
    manifest: dict[str, Any],
    instance_summary: pd.DataFrame,
    predictor_summary: pd.DataFrame,
    gate: dict[str, Any],
) -> None:
    lines = [
        "# SWE-bench Lite File-Localization v2 Predictive Validity",
        "",
        f"- Gate: `{gate.get('status')}`",
        f"- Dataset: `{SWE_BENCH_DATASET}`",
        f"- Models: `{', '.join(manifest.get('models', []))}`",
        f"- Issue records: `{manifest.get('instances')}`",
        f"- Planned calls: `{manifest.get('planned_remote_calls')}`",
        f"- Primary outcome: `{gate.get('primary_outcome')}`",
        f"- Protocol coverage: `{gate.get('protocol_coverage_status', 'not_written')}`",
        f"- Gate rule: {gate.get('gate_rule')}",
        "",
        "Primary analysis uses only raw task-only calls for the outcome and excludes gold-derived retrieval rank, "
        "retrieval recall and reference-file count from the non-CSB baseline. Model-visible state is constructed "
        "from issue-visible and repository-visible information; reference patch files are used after generation for "
        "outcome scoring and diagnostic readouts.",
        "",
        "This is SWE-bench Lite issue-to-file localization, not full repository checkout, patch application or unit-test execution.",
        "",
        "## Incremental Prediction",
        "",
        "| baseline AUC | baseline+CSB AUC | delta AUC | issue-cluster CI low | issue-cluster CI high | CSB coef |",
        "|---:|---:|---:|---:|---:|---:|",
        f"| {float(gate.get('baseline_incremental_auc', float('nan'))):.3f} | "
        f"{float(gate.get('augmented_csb_auc', float('nan'))):.3f} | "
        f"{float(gate.get('delta_auc', float('nan'))):.3f} | "
        f"{float(gate.get('delta_auc_ci_low', float('nan'))):.3f} | "
        f"{float(gate.get('delta_auc_ci_high', float('nan'))):.3f} | "
        f"{float(gate.get('state_binding_logistic_coefficient', float('nan'))):.3f} |",
        "",
        f"- CSB complete rows: `{gate.get('csb_complete_rows', 'not_reported')}` / `{gate.get('model_instance_rows', 'not_reported')}`",
        f"- CSB complete rate: `{gate.get('csb_complete_rate', 'not_reported')}`",
        "",
        "## Sensitivity Checks",
        "",
        "| check | positive | estimable |",
        "|---|---:|---:|",
        f"| leave-one-repo | {int(gate.get('leave_one_repo_positive', 0))} | {int(gate.get('leave_one_repo_total', 0))} |",
        f"| leave-one-model | {int(gate.get('leave_one_model_positive', 0))} | {int(gate.get('leave_one_model_total', 0))} |",
        f"| within-model | {int(gate.get('within_model_positive', 0))} | {int(gate.get('within_model_total', 0))} |",
        f"| within-repo | {int(gate.get('within_repo_positive', 0))} | {int(gate.get('within_repo_total', 0))} |",
    ]
    lines.extend(["", "## Secondary Outcomes", ""])
    lines.extend(
        [
            "| outcome | baseline AUC | baseline+CSB AUC | delta AUC | positives | negatives |",
            "|---|---:|---:|---:|---:|---:|",
            f"| task-only hard-constraint no-violation | {float(gate.get('no_violation_baseline_auc', float('nan'))):.3f} | "
            f"{float(gate.get('no_violation_augmented_csb_auc', float('nan'))):.3f} | "
            f"{float(gate.get('no_violation_delta_auc', float('nan'))):.3f} | "
            f"{int(gate.get('no_violation_positive_rows', 0))} | {int(gate.get('no_violation_negative_rows', 0))} |",
            f"| task-only constraint-clean hit@3 | {float(gate.get('constraint_clean_baseline_auc', float('nan'))):.3f} | "
            f"{float(gate.get('constraint_clean_augmented_csb_auc', float('nan'))):.3f} | "
            f"{float(gate.get('constraint_clean_delta_auc', float('nan'))):.3f} | "
            f"{int(gate.get('constraint_clean_positive_rows', 0))} | {int(gate.get('constraint_clean_negative_rows', 0))} |",
        ]
    )
    if not predictor_summary.empty:
        lines.extend(
            [
                "",
                "## Simple Predictor Summary",
                "",
                "| predictor | primary hit@3 AUC | primary Pearson | no-violation AUC | constraint-clean AUC |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in predictor_summary.to_dict(orient="records"):
            lines.append(
                f"| {row['predictor']} | {float(row['success_auc']):.3f} | "
                f"{float(row['success_pearson']):.3f} | {float(row['no_violation_auc']):.3f} | "
                f"{float(row.get('constraint_clean_auc', float('nan'))):.3f} |"
            )
    lines.extend(["", "## Wrapper Arm", ""])
    lines.extend(
        [
            "| wrapper delta hard violation | wrapper delta hit@3 | noninferiority margin | wrapper status |",
            "|---:|---:|---:|---|",
            f"| {float(gate.get('wrapper_delta_hard_violation', float('nan'))):.3f} | "
            f"{float(gate.get('wrapper_delta_hit_at3', float('nan'))):.3f} | "
            f"{float(gate.get('wrapper_noninferiority_margin', float('nan'))):.3f} | "
            f"{gate.get('wrapper_status')} |",
        ]
    )
    if not instance_summary.empty:
        by_model = (
            instance_summary.groupby("model", sort=True)
            .agg(
                model_instance_rows=("instance_id", "size"),
                hit_at3=("implementation_hit_at3_rate", "mean"),
                clean_hit_at3=("constraint_clean_hit_at3_rate", "mean"),
                hard_violation=("hard_constraint_violation_rate", "mean"),
                state_binding_score=("state_binding_score", "mean"),
                vote_margin=("vote_margin", "mean"),
                test_only=("test_only_target_rate", "mean"),
                candidate_absent=("candidate_absent_violation_rate", "mean"),
                nonexistent=("nonexistent_path_violation_rate", "mean"),
                scrambled_follow=("scrambled_state_following_rate", "mean"),
                empty_target=("empty_target_rate", "mean"),
                defer=("defer_rate", "mean"),
            )
            .reset_index()
        )
        lines.extend(["", "## Model Summary", ""])
        lines.extend(["| model | rows | primary hit@3 | clean hit@3 | hard violation | CSB | vote margin |", "|---|---:|---:|---:|---:|---:|---:|"])
        for row in by_model.to_dict(orient="records"):
            lines.append(
                f"| {row['model']} | {int(row['model_instance_rows'])} | {float(row['hit_at3']):.3f} | "
                f"{float(row['clean_hit_at3']):.3f} | {float(row['hard_violation']):.3f} | "
                f"{float(row['state_binding_score']):.3f} | {float(row['vote_margin']):.3f} |"
            )
        lines.extend(["", "## Failure Taxonomy", ""])
        lines.extend(
            [
                "| model | test-only | candidate-absent | nonexistent | scrambled-follow | empty/no-op | defer |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in by_model.to_dict(orient="records"):
            lines.append(
                f"| {row['model']} | {float(row['test_only']):.3f} | {float(row['candidate_absent']):.3f} | "
                f"{float(row['nonexistent']):.3f} | {float(row['scrambled_follow']):.3f} | "
                f"{float(row['empty_target']):.3f} | {float(row['defer']):.3f} |"
            )
    lines.extend(["", "Interpretation: pre-registered v2 issue-to-file localization evidence only.", ""])
    (output_dir / "swebench_file_localization_v2_report.md").write_text(
        "\n".join(lines).rstrip() + "\n",
        encoding="utf-8",
    )


def localization_v2_protocol_coverage(manifest: dict[str, Any], gate: dict[str, Any] | None = None) -> dict[str, Any]:
    gate = gate or {}
    conditions = set(manifest.get("conditions", []))
    wrapper_arms = set(manifest.get("wrapper_arms", []))
    models = list(manifest.get("models", []))
    instances = int(manifest.get("instances", 0) or 0)
    planned_remote_calls = int(manifest.get("planned_remote_calls", 0) or 0)
    hard_call_cap = int(manifest.get("hard_call_cap", planned_remote_calls) or planned_remote_calls)
    execution_slice_size = int(manifest.get("execution_slice_size", 0) or 0)
    observed_rows = int(gate.get("observed_rows", gate.get("rows", 0)) or 0)
    planned_task_rows = int(manifest.get("planned_task_rows", gate.get("planned_task_rows", 0)) or 0)
    model_instance_rows = int(gate.get("model_instance_rows", 0) or 0)
    csb_complete_rate = float(gate.get("csb_complete_rate", 1.0 if model_instance_rows == 0 else 0.0) or 0.0)
    checks = {
        "sample_full_300_or_declared_stratified_fallback": bool(
            instances >= 300
            or (
                instances >= 120
                and "stratified" in str(manifest.get("sampling_strategy", "")).lower()
            )
        ),
        "at_least_six_models": len(models) >= 6,
        "oracle_free_primary_predictor": manifest.get("primary_predictor")
        == "state_binding_score_from_oracle_free_model_visible_state",
        "oracle_diagnostic_not_primary_predictor": manifest.get("primary_predictor")
        == "state_binding_score_from_oracle_free_model_visible_state",
        "all_primary_conditions_present": set(LOCALIZATION_V2_CONDITIONS).issubset(conditions),
        "wrapper_arms_present": {"raw", "binding_guard", "compute_matched_repair"}.issubset(wrapper_arms),
        "primary_outcome_hit_at3": manifest.get("primary_outcome") == "task_only_implementation_file_hit_at3",
        "issue_cluster_bootstrap_configured": int(
            manifest.get("bootstrap_samples", gate.get("bootstrap_issue_cluster_samples", 0)) or 0
        )
        > 0,
        "incremental_auc_gate_configured": float(
            manifest.get("delta_auc_gate", gate.get("delta_auc_gate", 0.0)) or 0.0
        )
        >= 0.03,
        "execution_slice_manifest_requested_or_not_claimed": execution_slice_size == 0 or execution_slice_size >= 30,
        "planned_calls_within_cap": planned_remote_calls <= hard_call_cap,
        "observed_task_rows_match_plan_when_present": bool(
            observed_rows == 0
            or planned_task_rows == 0
            or observed_rows == planned_task_rows
        ),
        "csb_components_complete_when_observed": bool(model_instance_rows == 0 or math.isclose(csb_complete_rate, 1.0)),
    }
    missing = [name for name, passed in checks.items() if not passed]
    return {
        "status": "pass" if not missing else "needs_review",
        "checks": checks,
        "missing_or_needs_review": missing,
        "scope_note": "Covers v2 protocol implementation and planning artifacts. It does not prove API results or official SWE-bench patch/test resolution.",
    }


def _read_v2_trace_input(path: Path) -> pd.DataFrame:
    source = Path(path)
    if source.is_dir():
        for name in ["traces.parquet", "traces.csv", "traces_checkpoint.jsonl"]:
            candidate = source / name
            if not candidate.exists():
                continue
            try:
                return _read_v2_trace_input(candidate)
            except Exception:
                continue
        return pd.DataFrame()
    suffix = source.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(source)
    if suffix == ".csv":
        return pd.read_csv(source)
    if suffix == ".jsonl":
        return pd.DataFrame(_checkpoint_records(source))
    raise ValueError(f"unsupported trace input: {source}")


def _read_json_file(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def audit_swebench_v2_readiness(
    *,
    plan_dir: Path,
    smoke_dir: Path | None = None,
    official_command_dir: Path | None = None,
) -> dict[str, Any]:
    plan_dir = Path(plan_dir)
    smoke_dir = Path(smoke_dir) if smoke_dir is not None else None
    official_command_dir = Path(official_command_dir) if official_command_dir is not None else plan_dir / "official_harness_command_check"
    manifest = _read_json_file(plan_dir / "run_manifest.json")
    gate = _read_json_file(plan_dir / "gate_status.json")
    coverage = _read_json_file(plan_dir / "protocol_coverage.json")
    audit_rows = plan_dir / "planned_issue_retrieval_audit.csv"
    execution_slice_csv = plan_dir / "official_execution_slice_instances.csv"
    execution_slice_jsonl = plan_dir / "official_execution_slice_instances.jsonl"
    planned_issue_rows = 0
    if audit_rows.exists():
        try:
            planned_issue_rows = int(len(pd.read_csv(audit_rows)))
        except Exception:
            planned_issue_rows = 0
    execution_slice_rows = 0
    if execution_slice_csv.exists():
        try:
            execution_slice_rows = int(len(pd.read_csv(execution_slice_csv)))
        except Exception:
            execution_slice_rows = 0
    planned_task_rows = int(manifest.get("planned_task_rows", gate.get("planned_task_rows", 0)) or 0)
    observed_rows = int(gate.get("observed_rows", gate.get("rows", 0)) or 0)
    full_results_present = bool((plan_dir / "traces.csv").exists() or (plan_dir / "traces.parquet").exists())
    primary_result_complete = bool(
        full_results_present
        and observed_rows == planned_task_rows
        and float(gate.get("csb_complete_rate", 0.0) or 0.0) == 1.0
        and str(gate.get("status", "")).strip() in {"pass", "needs_review"}
    )

    smoke_gate = _read_json_file(smoke_dir / "gate_status.json") if smoke_dir is not None else {}
    smoke_pass = bool(
        smoke_gate
        and smoke_gate.get("status") == "pass"
        and int(smoke_gate.get("pass_rows", 0) or 0) == int(smoke_gate.get("model_count", 0) or 0)
        and int(smoke_gate.get("needs_review_rows", 1) or 0) == 0
    )

    command_meta = _read_json_file(official_command_dir / "official_execution_slice_command.json")
    command = [str(item) for item in command_meta.get("command", [])] if isinstance(command_meta.get("command", []), list) else []
    command_instance_count = int(command_meta.get("instance_count", 0) or 0)
    official_command_ready = bool(
        command_meta
        and command_instance_count == execution_slice_rows
        and "--dataset_name" in command
        and SWE_BENCH_DATASET in command
        and "--split" in command
        and "test" in command
        and "--predictions_path" in command
        and "--instance_ids" in command
    )
    official_analysis_dir = plan_dir / "official_execution_slice_analysis"
    official_execution_results_present = bool(
        official_analysis_dir.exists()
        and (
            (official_analysis_dir / "official_execution_slice_gate_status.json").exists()
            or (official_analysis_dir / "official_execution_slice_merged.csv").exists()
        )
    )

    checks = {
        "plan_manifest_present": bool(manifest),
        "plan_gate_present": bool(gate),
        "protocol_coverage_present": bool(coverage),
        "protocol_coverage_pass": coverage.get("status") == "pass",
        "full_300_or_declared_fallback": int(manifest.get("instances", 0) or 0) >= 300
        or (
            int(manifest.get("instances", 0) or 0) >= 120
            and "stratified" in str(manifest.get("sampling_strategy", "")).lower()
        ),
        "six_or_more_models": len(list(manifest.get("models", []))) >= 6,
        "all_primary_conditions_present": set(LOCALIZATION_V2_CONDITIONS).issubset(set(manifest.get("conditions", []))),
        "wrapper_arms_present": {"raw", "binding_guard", "compute_matched_repair"}.issubset(set(manifest.get("wrapper_arms", []))),
        "oracle_free_primary_predictor": manifest.get("primary_predictor")
        == "state_binding_score_from_oracle_free_model_visible_state",
        "primary_outcome_hit_at3": manifest.get("primary_outcome") == "task_only_implementation_file_hit_at3",
        "planned_issue_audit_rows_match": planned_issue_rows == int(manifest.get("shard_issue_rows", manifest.get("instances", 0)) or 0),
        "execution_slice_size_30_to_50_when_requested": int(manifest.get("execution_slice_size", 0) or 0) == 0
        or 30 <= execution_slice_rows <= 50,
        "api_model_smoke_pass": smoke_pass,
        "official_harness_command_ready": official_command_ready,
        "full_api_results_present": full_results_present,
        "full_api_results_complete": primary_result_complete,
        "official_execution_results_present": official_execution_results_present,
    }
    hard_readiness_keys = [
        "plan_manifest_present",
        "plan_gate_present",
        "protocol_coverage_present",
        "protocol_coverage_pass",
        "full_300_or_declared_fallback",
        "six_or_more_models",
        "all_primary_conditions_present",
        "wrapper_arms_present",
        "oracle_free_primary_predictor",
        "primary_outcome_hit_at3",
        "planned_issue_audit_rows_match",
        "execution_slice_size_30_to_50_when_requested",
        "api_model_smoke_pass",
        "official_harness_command_ready",
    ]
    readiness_pass = all(bool(checks[key]) for key in hard_readiness_keys)
    if primary_result_complete and official_execution_results_present:
        status = "complete_results_available"
    elif readiness_pass:
        status = "ready_for_full_run"
    else:
        status = "needs_review"
    return {
        "status": status,
        "checks": checks,
        "missing_or_needs_review": [key for key, value in checks.items() if not value],
        "plan_dir": str(plan_dir),
        "smoke_dir": str(smoke_dir) if smoke_dir is not None else "",
        "official_command_dir": str(official_command_dir),
        "planned_remote_calls": int(manifest.get("planned_remote_calls", gate.get("planned_remote_calls", 0)) or 0),
        "planned_task_rows": planned_task_rows,
        "planned_issue_rows": planned_issue_rows,
        "execution_slice_rows": execution_slice_rows,
        "observed_rows": observed_rows,
        "scope_note": "Readiness audit. ready_for_full_run means protocol artifacts are ready; it does not prove the full API run or official SWE-bench execution has completed.",
    }


def _line_has_negation_near(text: str, start: int) -> bool:
    window = text[max(0, start - 90) : start].lower()
    negators = [
        "not ",
        "no ",
        "does not ",
        "do not ",
        "rather than ",
        "instead of ",
        "without ",
        "not a ",
        "not an ",
        "not full ",
        "not oracle-free ",
    ]
    return any(token in window for token in negators)


def _claim_hits_for_line(line: str, *, full_api_complete: bool, official_execution_present: bool) -> list[str]:
    hits: list[str] = []
    checks: list[tuple[str, str, bool]] = [
        (
            "oracle_free_predictor_superiority_without_v2_results",
            r"\b(?:CSB|causal state binding|state[- ]binding).{0,120}\boutperform(?:s|ed)?\b.{0,120}\bSWE[- ]bench\b",
            full_api_complete,
        ),
        (
            "strong_real_task_predictor_without_v2_results",
            r"\b(?:CSB|causal state binding|state[- ]binding).{0,120}\bpredict(?:s|ed)?\b.{0,120}\b(?:real[- ]task|issue[- ]record|SWE[- ]bench)\b.{0,120}\b(?:beyond|better than|outperform)",
            full_api_complete,
        ),
        (
            "oracle_free_deployment_predictor_without_v2_results",
            r"\boracle[- ]free deployment predictor\b",
            full_api_complete,
        ),
        (
            "full_swebench_execution_without_official_results",
            r"\bfull SWE[- ]bench\b.{0,80}\b(?:task[- ]execution|repository execution|patch/test|resolution|resolved)\b",
            official_execution_present,
        ),
        (
            "official_swebench_pass_without_official_results",
            r"\bofficial\b.{0,80}\b(?:SWE[- ]bench|resolved|tests[- ]pass)\b.{0,80}\b(?:pass|passed|resolved)\b",
            official_execution_present,
        ),
    ]
    for name, pattern, support_present in checks:
        if support_present:
            continue
        for match in re.finditer(pattern, line, flags=re.IGNORECASE):
            if not _line_has_negation_near(line, match.start()):
                hits.append(name)
                break
    return hits


def audit_swebench_v2_claim_support(
    *,
    readiness_audit_path: Path,
    manuscript_paths: Sequence[Path],
) -> dict[str, Any]:
    readiness = _read_json_file(Path(readiness_audit_path))
    checks = readiness.get("checks", {}) if isinstance(readiness.get("checks", {}), dict) else {}
    full_api_complete = bool(checks.get("full_api_results_complete", False))
    official_execution_present = bool(checks.get("official_execution_results_present", False))
    findings: list[dict[str, Any]] = []
    combined_text_parts: list[str] = []
    scanned_files = 0
    for raw_path in manuscript_paths:
        path = Path(raw_path)
        if not path.exists() or not path.is_file():
            findings.append({"file": str(path), "line": 0, "issue": "missing_file", "text": ""})
            continue
        scanned_files += 1
        try:
            lines = path.read_text(encoding="utf-8-sig", errors="ignore").splitlines()
        except Exception as exc:
            findings.append({"file": str(path), "line": 0, "issue": f"read_error_{type(exc).__name__}", "text": ""})
            continue
        combined_text_parts.extend(lines)
        for line_no, line in enumerate(lines, start=1):
            for issue in _claim_hits_for_line(
                line,
                full_api_complete=full_api_complete,
                official_execution_present=official_execution_present,
            ):
                findings.append({"file": str(path), "line": int(line_no), "issue": issue, "text": line.strip()[:500]})
    combined = "\n".join(combined_text_parts).lower()
    conservative_checks = {
        "states_issue_to_file_not_full_execution": bool(
            "not full swe-bench" in combined
            or "not full task execution" in combined
            or "not a swe-bench repository execution" in combined
            or "not repository checkout" in combined
            or "not full swe-bench task execution" in combined
        ),
        "states_not_oracle_free_deployment_prediction": bool(
            "not oracle-free deployment prediction" in combined
            or "not an oracle-free deployment predictor" in combined
            or "rather than an oracle-free deployment predictor" in combined
            or "not oracle-free deployment predictor" in combined
        ),
        "states_mixed_or_boundary_pilot": bool("mixed" in combined and ("pilot" in combined or "boundary" in combined)),
    }
    if not full_api_complete:
        for key in ["states_not_oracle_free_deployment_prediction", "states_mixed_or_boundary_pilot"]:
            if not conservative_checks[key]:
                findings.append({"file": "", "line": 0, "issue": f"missing_conservative_wording_{key}", "text": ""})
    if not official_execution_present and not conservative_checks["states_issue_to_file_not_full_execution"]:
        findings.append(
            {
                "file": "",
                "line": 0,
                "issue": "missing_conservative_wording_states_issue_to_file_not_full_execution",
                "text": "",
            }
        )
    status = "pass" if not findings else "needs_review"
    return {
        "status": status,
        "readiness_audit_path": str(readiness_audit_path),
        "scanned_files": int(scanned_files),
        "full_api_results_complete": full_api_complete,
        "official_execution_results_present": official_execution_present,
        "conservative_checks": conservative_checks,
        "findings": findings,
        "scope_note": "Claim-support guard for SWE-bench v2 wording. It prevents strong oracle-free real-task or full SWE-bench claims unless the corresponding result artifacts exist.",
    }


def prepare_swebench_v2_launch_plan(
    *,
    plan_dir: Path,
    smoke_dir: Path | None = None,
    run_id: str,
    shard_count: int = 3,
    api_file: str = "<api_file>",
    output_root: str = "results/paper1_revision/real_task_predictive_validity",
    per_shard_hard_call_cap: int = 8000,
    max_workers: int = 8,
    fetch_repo_tree: bool = True,
    repo_checkout_root: str = "",
) -> dict[str, Any]:
    plan_dir = Path(plan_dir)
    manifest = _read_json_file(plan_dir / "run_manifest.json")
    if not manifest:
        raise FileNotFoundError(f"run_manifest.json not found or invalid in {plan_dir}")
    models = list(manifest.get("models", []))
    if not models:
        raise ValueError("manifest does not include models")
    shard_count = max(1, int(shard_count))
    base_command = [
        "python",
        "scripts/run/run_swebench_file_localization_v2_api.py",
        "--api-file",
        api_file,
        "--models-from-api-file",
        "--model-count",
        str(len(models)),
        "--instances",
        str(int(manifest.get("instances", 300) or 300)),
        "--offset",
        str(int(manifest.get("offset", 0) or 0)),
        "--split",
        str(manifest.get("split", "test")),
        "--self-consistency-repeats",
        str(int(manifest.get("self_consistency_repeats", 3) or 3)),
        "--diagnostic-repeats",
        str(int(manifest.get("diagnostic_repeats", 1) or 1)),
        "--top-k",
        str(int(manifest.get("top_k", 50) or 50)),
        "--execution-slice-size",
        str(int(manifest.get("execution_slice_size", 0) or 0)),
        "--bootstrap-samples",
        str(int(manifest.get("bootstrap_samples", 400) or 400)),
        "--delta-auc-gate",
        str(float(manifest.get("delta_auc_gate", 0.03) or 0.03)),
        "--noninferiority-margin",
        str(float(manifest.get("noninferiority_margin", -0.03) or -0.03)),
        "--max-workers",
        str(int(max_workers)),
        "--output-root",
        output_root,
    ]
    if fetch_repo_tree:
        base_command.append("--fetch-repo-tree")
    if repo_checkout_root:
        base_command.extend(["--repo-checkout-root", repo_checkout_root])
    if bool(manifest.get("include_oracle_diagnostic", False)):
        base_command.append("--include-oracle-diagnostic")
    if not bool(manifest.get("include_wrapper_arms", True)):
        base_command.append("--no-wrapper-arms")

    shard_commands: list[dict[str, Any]] = []
    for shard_index in range(shard_count):
        shard_run_id = f"{run_id}_shard{shard_index}"
        command = [
            *base_command,
            "--run-id",
            shard_run_id,
            "--shard-count",
            str(shard_count),
            "--shard-index",
            str(shard_index),
            "--hard-call-cap",
            str(int(per_shard_hard_call_cap)),
        ]
        resume_command = [*command, "--resume"]
        shard_commands.append(
            {
                "shard_index": shard_index,
                "run_id": shard_run_id,
                "output_dir": str(Path(output_root) / shard_run_id),
                "command": command,
                "resume_command": resume_command,
            }
        )
    shard_dirs = [item["output_dir"] for item in shard_commands]
    merged_dir = str(Path(output_root) / f"{run_id}_merged")
    merge_command = [
        "python",
        "scripts/run/merge_swebench_fileloc_v2_shards.py",
        "--inputs",
        *shard_dirs,
        "--output-dir",
        merged_dir,
        "--bootstrap-samples",
        str(int(manifest.get("bootstrap_samples", 400) or 400)),
        "--delta-auc-gate",
        str(float(manifest.get("delta_auc_gate", 0.03) or 0.03)),
        "--noninferiority-margin",
        str(float(manifest.get("noninferiority_margin", -0.03) or -0.03)),
    ]
    readiness_command = [
        "python",
        "scripts/run/audit_swebench_v2_readiness.py",
        "--plan-dir",
        merged_dir,
    ]
    if smoke_dir is not None:
        readiness_command.extend(["--smoke-dir", str(smoke_dir)])
    return {
        "status": "launch_plan_prepared",
        "source_plan_dir": str(plan_dir),
        "smoke_dir": str(smoke_dir) if smoke_dir is not None else "",
        "run_id": run_id,
        "shard_count": shard_count,
        "models": models,
        "planned_remote_calls_total": int(manifest.get("planned_remote_calls", 0) or 0),
        "estimated_remote_calls_per_shard": int(math.ceil(int(manifest.get("planned_remote_calls", 0) or 0) / shard_count)),
        "per_shard_hard_call_cap": int(per_shard_hard_call_cap),
        "output_root": output_root,
        "repo_checkout_root": repo_checkout_root,
        "shards": shard_commands,
        "merge_command": merge_command,
        "readiness_command_after_merge": readiness_command,
        "scope_note": "Launch plan only. Commands are not executed by this function.",
    }


def _trace_row_count(path: Path) -> int:
    for candidate in [path / "traces.parquet", path / "traces.csv", path / "traces_checkpoint.jsonl"]:
        if not candidate.exists():
            continue
        try:
            if candidate.suffix.lower() == ".parquet":
                return int(len(pd.read_parquet(candidate)))
            if candidate.suffix.lower() == ".csv":
                return int(len(pd.read_csv(candidate)))
            if candidate.suffix.lower() == ".jsonl":
                return int(len(_checkpoint_records(candidate)))
        except Exception:
            continue
    return 0


def _command_value(command: Sequence[Any], flag: str) -> str:
    values = [str(item) for item in command]
    try:
        idx = values.index(flag)
    except ValueError:
        return ""
    return values[idx + 1] if idx + 1 < len(values) else ""


def _trace_error_acceptability(path: Path, *, gate_error_rows: int) -> dict[str, Any]:
    total_rows = 0
    error_rows = int(gate_error_rows)
    max_model_error_rate = 0.0
    error_rate = 0.0
    try:
        traces = _read_v2_trace_input(path)
        total_rows = int(len(traces))
        if total_rows > 0 and "error_type" in traces:
            error_mask = traces["error_type"].fillna("").astype(str).str.strip().ne("")
            error_rows = int(error_mask.sum())
            error_rate = float(error_rows / total_rows)
            if "model" in traces:
                model_error_rates = error_mask.groupby(traces["model"].fillna("").astype(str)).mean()
                max_model_error_rate = float(model_error_rates.max()) if len(model_error_rates) else 0.0
    except Exception:
        total_rows = 0
        error_rate = 0.0
        max_model_error_rate = 0.0
    if total_rows <= 0 and gate_error_rows > 0:
        error_rate = 1.0
        max_model_error_rate = 1.0
    acceptable = bool(error_rows == 0 or (error_rate <= 0.05 and max_model_error_rate <= 0.15))
    return {
        "error_rows": int(error_rows),
        "error_rate": float(error_rate),
        "max_model_error_rate": float(max_model_error_rate),
        "acceptable": acceptable,
    }


def audit_swebench_v2_shards(*, launch_plan_path: Path) -> dict[str, Any]:
    launch_plan_path = Path(launch_plan_path)
    launch_plan = _read_json_file(launch_plan_path)
    if not launch_plan:
        raise FileNotFoundError(f"launch plan not found or invalid: {launch_plan_path}")
    shard_rows: list[dict[str, Any]] = []
    for shard in launch_plan.get("shards", []):
        output_dir = Path(str(shard.get("output_dir", "")))
        manifest = _read_json_file(output_dir / "run_manifest.json")
        gate = _read_json_file(output_dir / "gate_status.json")
        coverage = _read_json_file(output_dir / "protocol_coverage.json")
        command = shard.get("command", [])
        planned_task_rows = int(manifest.get("planned_task_rows", gate.get("planned_task_rows", 0)) or 0)
        planned_remote_calls = int(manifest.get("planned_remote_calls", gate.get("planned_remote_calls", 0)) or 0)
        observed_rows = int(gate.get("observed_rows", gate.get("rows", 0)) or 0)
        trace_rows = _trace_row_count(output_dir)
        checkpoint_rows = int(len(_checkpoint_records(output_dir / "traces_checkpoint.jsonl"))) if (output_dir / "traces_checkpoint.jsonl").exists() else 0
        if observed_rows <= 0:
            observed_rows = trace_rows
        rows_match = bool(planned_task_rows > 0 and observed_rows == planned_task_rows and trace_rows == planned_task_rows)
        shard_started = bool(output_dir.exists() and (manifest or gate or checkpoint_rows > 0 or trace_rows > 0))
        error_acceptability = _trace_error_acceptability(
            output_dir,
            gate_error_rows=int(gate.get("error_rows", 0) or 0),
        )
        complete = bool(
            rows_match
            and gate
            and coverage.get("status") == "pass"
            and error_acceptability["acceptable"]
            and float(gate.get("csb_complete_rate", 1.0) or 0.0) == 1.0
        )
        if complete:
            status = "complete"
        elif shard_started:
            status = "incomplete_or_needs_review"
        else:
            status = "not_started"
        shard_rows.append(
            {
                "shard_index": int(shard.get("shard_index", _to_float(_command_value(command, "--shard-index"), 0))),
                "run_id": str(shard.get("run_id", _command_value(command, "--run-id"))),
                "output_dir": str(output_dir),
                "status": status,
                "started": shard_started,
                "complete": complete,
                "planned_task_rows": planned_task_rows,
                "planned_remote_calls": planned_remote_calls,
                "observed_rows": observed_rows,
                "trace_rows": trace_rows,
                "checkpoint_rows": checkpoint_rows,
                "error_rows": int(error_acceptability["error_rows"]),
                "error_rate": float(error_acceptability["error_rate"]),
                "max_model_error_rate": float(error_acceptability["max_model_error_rate"]),
                "error_acceptability": bool(error_acceptability["acceptable"]),
                "coverage_status": str(coverage.get("status", "")),
                "gate_status": str(gate.get("status", "")),
                "resume_command": shard.get("resume_command", []),
            }
        )
    merge_command = [str(item) for item in launch_plan.get("merge_command", [])]
    merge_output = Path(_command_value(merge_command, "--output-dir")) if merge_command else Path("")
    merged_gate = _read_json_file(merge_output / "gate_status.json") if str(merge_output) else {}
    complete_count = sum(1 for row in shard_rows if row["complete"])
    started_count = sum(1 for row in shard_rows if row["started"])
    merge_ready = bool(shard_rows and complete_count == len(shard_rows))
    merged_present = bool(merge_output and ((merge_output / "traces.csv").exists() or (merge_output / "traces.parquet").exists()))
    return {
        "status": "ready_to_merge"
        if merge_ready and not merged_present
        else ("merged_results_present" if merged_present else ("in_progress" if started_count else "not_started")),
        "launch_plan_path": str(launch_plan_path),
        "shard_count": int(len(shard_rows)),
        "started_shards": int(started_count),
        "complete_shards": int(complete_count),
        "merge_ready": merge_ready,
        "merged_results_present": merged_present,
        "merge_output_dir": str(merge_output) if str(merge_output) else "",
        "merge_gate_status": merged_gate.get("status", ""),
        "shards": shard_rows,
        "missing_or_needs_review": [
            row["run_id"] or str(row["shard_index"])
            for row in shard_rows
            if not row["complete"]
        ],
        "scope_note": "Shard execution audit. It checks execution completeness and merge readiness, not whether the final scientific gate passes.",
    }


def merge_swebench_filelocalization_v2_outputs(
    input_paths: Sequence[Path],
    *,
    output_dir: Path,
    bootstrap_samples: int = 400,
    delta_auc_gate: float = 0.03,
    noninferiority_margin: float = -0.03,
) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    trace_frames: list[pd.DataFrame] = []
    manifests: list[dict[str, Any]] = []
    planned_issue_frames: list[pd.DataFrame] = []
    execution_slice_frames: list[pd.DataFrame] = []
    for raw_path in input_paths:
        path = Path(raw_path)
        frame = _read_v2_trace_input(path)
        if not frame.empty:
            frame["source_run"] = str(path)
            trace_frames.append(frame)
        manifest_path = path / "run_manifest.json" if path.is_dir() else path.parent / "run_manifest.json"
        if manifest_path.exists():
            try:
                manifests.append(json.loads(manifest_path.read_text(encoding="utf-8-sig")))
            except json.JSONDecodeError:
                pass
        artifact_dir = path if path.is_dir() else path.parent
        planned_issue_path = artifact_dir / "planned_issue_retrieval_audit.csv"
        if planned_issue_path.exists():
            try:
                planned_issue_frames.append(pd.read_csv(planned_issue_path))
            except Exception:
                pass
        execution_slice_path = artifact_dir / "official_execution_slice_instances.csv"
        if execution_slice_path.exists():
            try:
                execution_slice_frames.append(pd.read_csv(execution_slice_path))
            except Exception:
                pass
    if not trace_frames:
        raise RuntimeError("no v2 traces found to merge")
    traces = pd.concat(trace_frames, axis=0, ignore_index=True)
    if "task_key" in traces.columns:
        traces = traces.drop_duplicates("task_key", keep="last")
    sort_columns = [column for column in ["model", "instance_id", "wrapper_arm", "condition", "repeat_index"] if column in traces.columns]
    traces = traces.sort_values(sort_columns).reset_index(drop=True) if sort_columns else traces.reset_index(drop=True)
    instance_summary, predictor_summary, gate = summarize_localization_predictive_validity_v2(
        traces,
        bootstrap_samples=bootstrap_samples,
        delta_auc_gate=delta_auc_gate,
        noninferiority_margin=noninferiority_margin,
    )
    base_manifest = dict(manifests[0]) if manifests else {}
    combined_manifest = {
        **base_manifest,
        "experiment": "swebench_lite_file_localization_predictive_validity_v2_merged",
        "merged_input_count": int(len(input_paths)),
        "merged_sources": [str(path) for path in input_paths],
        "models": sorted({model for manifest in manifests for model in manifest.get("models", [])})
        or sorted(traces["model"].astype(str).unique().tolist()),
        "conditions": sorted({condition for manifest in manifests for condition in manifest.get("conditions", [])})
        or sorted(traces["condition"].astype(str).unique().tolist()),
        "wrapper_arms": sorted({arm for manifest in manifests for arm in manifest.get("wrapper_arms", [])})
        or sorted(traces.get("wrapper_arm", pd.Series(["raw"])).astype(str).unique().tolist()),
        "instances": int(max([int(manifest.get("instances", 0) or 0) for manifest in manifests] or [traces["instance_id"].nunique()])),
        "full_sample_issue_rows": int(max([int(manifest.get("full_sample_issue_rows", 0) or 0) for manifest in manifests] or [traces["instance_id"].nunique()])),
        "shard_issue_rows": int(traces["instance_id"].nunique()),
        "shard_count": int(max([int(manifest.get("shard_count", 1) or 1) for manifest in manifests] or [1])),
        "merged_shards": sorted({int(manifest.get("shard_index", 0) or 0) for manifest in manifests}) if manifests else [],
        "planned_remote_calls": int(sum(int(manifest.get("planned_remote_calls", 0) or 0) for manifest in manifests)),
        "planned_task_rows": int(sum(int(manifest.get("planned_task_rows", 0) or 0) for manifest in manifests)),
        "hard_call_cap": int(sum(int(manifest.get("hard_call_cap", 0) or 0) for manifest in manifests)) if manifests else int(gate.get("rows", 0)),
        "primary_predictor": "state_binding_score_from_oracle_free_model_visible_state",
        "primary_outcome": "task_only_implementation_file_hit_at3",
        "secondary_outcomes": [
            "task_only_hard_constraint_violation",
            "task_only_constraint_clean_implementation_file_hit_at3",
        ],
        "baseline_predictor_columns": gate.get("baseline_predictor_columns", []),
        "gold_derived_predictors_excluded": gate.get("gold_derived_predictors_excluded", []),
        "bootstrap_samples": int(bootstrap_samples),
        "delta_auc_gate": float(delta_auc_gate),
        "noninferiority_margin": float(noninferiority_margin),
    }
    if planned_issue_frames:
        planned_issue_frame = pd.concat(planned_issue_frames, axis=0, ignore_index=True)
        if "instance_id" in planned_issue_frame.columns:
            planned_issue_frame = planned_issue_frame.drop_duplicates("instance_id", keep="first")
    else:
        planned_columns = [
            "repo",
            "instance_id",
            "scrambled_instance_id",
            "problem_chars",
            "retrieved_candidate_count",
        ]
        present_columns = [column for column in planned_columns if column in traces.columns]
        planned_issue_frame = traces[present_columns].drop_duplicates("instance_id", keep="first") if present_columns else pd.DataFrame()
    if not planned_issue_frame.empty:
        planned_issue_frame = planned_issue_frame.sort_values(
            [column for column in ["repo", "instance_id"] if column in planned_issue_frame.columns]
        ).reset_index(drop=True)
        planned_issue_frame.to_csv(output_dir / "planned_issue_retrieval_audit.csv", index=False)
        combined_manifest["planned_issue_rows"] = int(len(planned_issue_frame))
    if execution_slice_frames:
        execution_slice_frame = pd.concat(execution_slice_frames, axis=0, ignore_index=True)
        if "instance_id" in execution_slice_frame.columns:
            execution_slice_frame = execution_slice_frame.drop_duplicates("instance_id", keep="first")
        execution_slice_frame = execution_slice_frame.sort_values(
            [column for column in ["repo", "instance_id"] if column in execution_slice_frame.columns]
        ).reset_index(drop=True)
        execution_slice_frame.to_csv(output_dir / "official_execution_slice_instances.csv", index=False)
        (output_dir / "official_execution_slice_instances.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in execution_slice_frame.to_dict(orient="records")) + "\n",
            encoding="utf-8",
        )
        combined_manifest["execution_slice_size"] = int(len(execution_slice_frame))
    gate.update(
        {
            "merged_input_count": int(len(input_paths)),
            "merged_trace_rows": int(len(traces)),
            "merged_unique_task_keys": int(traces["task_key"].nunique()) if "task_key" in traces else int(len(traces)),
            "merged_sources": [str(path) for path in input_paths],
        }
    )
    coverage = localization_v2_protocol_coverage(combined_manifest, gate)
    gate["protocol_coverage_status"] = coverage["status"]
    gate["protocol_coverage_missing_or_needs_review"] = coverage["missing_or_needs_review"]
    write_parquet(traces, output_dir / "traces.parquet")
    write_parquet(instance_summary, output_dir / "instance_summary.parquet")
    write_parquet(predictor_summary, output_dir / "predictor_summary.parquet")
    traces.to_csv(output_dir / "traces.csv", index=False)
    instance_summary.to_csv(output_dir / "instance_summary.csv", index=False)
    predictor_summary.to_csv(output_dir / "predictor_summary.csv", index=False)
    write_json(output_dir / "run_manifest.json", combined_manifest)
    write_json(output_dir / "gate_status.json", gate)
    write_json(output_dir / "protocol_coverage.json", coverage)
    write_localization_v2_report(
        output_dir,
        manifest=combined_manifest,
        instance_summary=instance_summary,
        predictor_summary=predictor_summary,
        gate=gate,
    )
    return {
        "traces": traces,
        "instance_summary": instance_summary,
        "predictor_summary": predictor_summary,
        "gate_status": gate,
        "protocol_coverage": coverage,
        "manifest": combined_manifest,
    }


def _read_records_file(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8-sig"))
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"unsupported results file type: {path}")


def _bool_from_value(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "pass", "passed", "resolved", "success", "succeeded"}:
        return True
    if text in {"false", "0", "no", "n", "fail", "failed", "unresolved", "error", "timeout"}:
        return False
    return None


def _official_record_from_mapping(payload: dict[str, Any], *, source: str) -> dict[str, Any] | None:
    instance_id = str(
        payload.get("instance_id")
        or payload.get("instance")
        or payload.get("id")
        or payload.get("task_id")
        or ""
    ).strip()
    if not instance_id:
        return None
    resolved = None
    for key in ["resolved", "is_resolved", "official_resolved", "success"]:
        if key in payload:
            resolved = _bool_from_value(payload.get(key))
            break
    tests_passed = None
    for key in ["tests_passed", "tests_pass", "test_passed", "official_tests_passed"]:
        if key in payload:
            tests_passed = _bool_from_value(payload.get(key))
            break
    model = str(payload.get("model") or payload.get("model_name_or_path") or payload.get("model_id") or "").strip()
    return {
        "instance_id": instance_id,
        "model": model,
        "official_resolved": bool(resolved) if resolved is not None else bool(tests_passed) if tests_passed is not None else False,
        "official_tests_passed": bool(tests_passed) if tests_passed is not None else bool(resolved) if resolved is not None else False,
        "official_source": source,
    }


def _collect_official_records(payload: Any, *, source: str) -> list[dict[str, Any]]:
    if isinstance(payload, pd.DataFrame):
        records = []
        for item in payload.to_dict(orient="records"):
            record = _official_record_from_mapping(item, source=source)
            if record:
                records.append(record)
        return records
    if isinstance(payload, list):
        records = []
        for item in payload:
            if isinstance(item, dict):
                record = _official_record_from_mapping(item, source=source)
                if record:
                    records.append(record)
            elif isinstance(item, str):
                records.append(
                    {
                        "instance_id": item,
                        "model": "",
                        "official_resolved": True,
                        "official_tests_passed": True,
                        "official_source": source,
                    }
                )
        return records
    if isinstance(payload, dict):
        records: list[dict[str, Any]] = []
        for key in ["resolved_ids", "resolved", "passed_ids", "success_ids"]:
            value = payload.get(key)
            if isinstance(value, list):
                for instance_id in value:
                    records.append(
                        {
                            "instance_id": str(instance_id),
                            "model": "",
                            "official_resolved": True,
                            "official_tests_passed": True,
                            "official_source": source,
                        }
                    )
        for key in ["unresolved_ids", "unresolved", "failed_ids", "error_ids", "timeout_ids"]:
            value = payload.get(key)
            if isinstance(value, list):
                for instance_id in value:
                    records.append(
                        {
                            "instance_id": str(instance_id),
                            "model": "",
                            "official_resolved": False,
                            "official_tests_passed": False,
                            "official_source": source,
                        }
                    )
        for key, value in payload.items():
            if isinstance(value, dict):
                nested = dict(value)
                nested.setdefault("instance_id", key)
                record = _official_record_from_mapping(nested, source=source)
                if record:
                    records.append(record)
            elif isinstance(value, list) and key not in {"resolved_ids", "resolved", "passed_ids", "success_ids", "unresolved_ids", "unresolved", "failed_ids", "error_ids", "timeout_ids"}:
                records.extend(_collect_official_records(value, source=source))
        return records
    return []


def load_official_swebench_results(path: Path) -> pd.DataFrame:
    source_path = Path(path)
    files: list[Path]
    if source_path.is_dir():
        files = [
            item
            for item in source_path.rglob("*")
            if item.is_file() and item.suffix.lower() in {".json", ".jsonl", ".csv", ".parquet"}
        ]
    else:
        files = [source_path]
    records: list[dict[str, Any]] = []
    for file in files:
        try:
            records.extend(_collect_official_records(_read_records_file(file), source=str(file)))
        except Exception:
            continue
    frame = pd.DataFrame(records)
    if frame.empty:
        return pd.DataFrame(
            columns=["instance_id", "model", "official_resolved", "official_tests_passed", "official_source"]
        )
    frame["instance_id"] = frame["instance_id"].astype(str)
    frame["model"] = frame["model"].fillna("").astype(str)
    frame["official_resolved"] = frame["official_resolved"].astype(bool)
    frame["official_tests_passed"] = frame["official_tests_passed"].astype(bool)
    return (
        frame.sort_values(["instance_id", "model", "official_resolved"], ascending=[True, True, False])
        .drop_duplicates(["instance_id", "model"], keep="first")
        .reset_index(drop=True)
    )


def _checkpoint_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            if isinstance(payload, dict):
                records.append(payload)
        except json.JSONDecodeError:
            continue
    return records


def _append_checkpoint_record(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def read_swebench_execution_slice_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"execution slice manifest not found: {path}")
    if path.suffix.lower() == ".jsonl":
        records: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                records.append(payload)
        frame = pd.DataFrame(records)
    else:
        frame = pd.read_csv(path)
    if "instance_id" not in frame.columns:
        raise ValueError(f"execution slice manifest lacks instance_id column: {path}")
    frame["instance_id"] = frame["instance_id"].fillna("").astype(str).str.strip()
    frame = frame[frame["instance_id"].ne("")].drop_duplicates("instance_id", keep="first").reset_index(drop=True)
    if frame.empty:
        raise RuntimeError(f"execution slice manifest contains no instance_id rows: {path}")
    return frame


def _prediction_path_for_model(output_dir: Path, model: str, *, dry_run: bool) -> Path:
    subdir = "planned_predictions_by_model" if dry_run else "predictions_by_model"
    return output_dir / subdir / f"{safe_slug(model)}.jsonl"


def _write_patch_generation_predictions(
    records: pd.DataFrame,
    output_dir: Path,
    *,
    models: Sequence[str],
    dry_run: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in models:
        model_records = records[records["model"].astype(str).eq(str(model))].copy() if not records.empty else pd.DataFrame()
        if "task_key" in model_records.columns:
            model_records = model_records.drop_duplicates("task_key", keep="last")
        if not model_records.empty:
            model_records = model_records.sort_values(["instance_id"]).drop_duplicates(["instance_id"], keep="last")
        prediction_path = _prediction_path_for_model(output_dir, str(model), dry_run=dry_run)
        ensure_dir(prediction_path.parent)
        lines: list[str] = []
        for record in model_records.to_dict(orient="records"):
            lines.append(
                json.dumps(
                    {
                        "instance_id": str(record.get("instance_id", "")),
                        "model_name_or_path": str(record.get("model", model)),
                        "model_patch": str(record.get("model_patch", "")),
                    },
                    ensure_ascii=False,
                )
            )
        prediction_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        rows.append(
            {
                "model": str(model),
                "predictions_path": str(prediction_path),
                "instance_count": int(len(model_records)),
                "nonempty_patch_count": int(model_records["model_patch"].fillna("").astype(str).str.len().gt(0).sum())
                if "model_patch" in model_records
                else 0,
                "deferred_count": int(model_records["defer"].fillna(False).astype(bool).sum()) if "defer" in model_records else 0,
                "error_count": int(model_records["error_type"].fillna("").astype(str).ne("").sum()) if "error_type" in model_records else 0,
                "dry_run": bool(dry_run),
            }
        )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(output_dir / "patch_predictions_manifest.csv", index=False)
    return manifest


def write_patch_generation_report(
    output_dir: Path,
    *,
    manifest: dict[str, Any],
    gate: dict[str, Any],
    predictions_manifest: pd.DataFrame,
) -> None:
    lines = [
        "# SWE-bench Execution Slice Patch Generation",
        "",
        f"- Status: `{gate.get('status')}`",
        f"- Dataset role: `{manifest.get('dataset_role')}`",
        f"- Slice manifest: `{manifest.get('slice_manifest')}`",
        f"- Models: `{', '.join(manifest.get('models', []))}`",
        f"- Slice instances: `{manifest.get('slice_instance_count')}`",
        f"- Planned remote calls: `{manifest.get('planned_remote_calls')}`",
        f"- Dry run: `{manifest.get('dry_run')}`",
        f"- Reference patch used in prompt: `{manifest.get('reference_patch_used_in_prompt')}`",
        f"- Repository file context enabled: `{manifest.get('include_file_context')}`",
        f"- Rows with file context: `{gate.get('file_context_rows', 0)}`",
        f"- Issues with file context: `{gate.get('file_context_issue_rows', 0)}` / `{manifest.get('slice_instance_count')}`",
        f"- File-context status: `{gate.get('file_context_status', 'not_requested')}`",
        "",
        "| model | predictions path | instances | nonempty patches | deferred | errors |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in predictions_manifest.to_dict(orient="records"):
        lines.append(
            f"| {row.get('model')} | {row.get('predictions_path')} | {row.get('instance_count')} | "
            f"{row.get('nonempty_patch_count')} | {row.get('deferred_count')} | {row.get('error_count')} |"
        )
    lines.extend(
        [
            "",
            "These files are patch-generation predictions or dry-run previews only. "
            "They become SWE-bench evidence only after `swebench.harness.run_evaluation` is run and merged.",
        ]
    )
    (output_dir / "patch_generation_report.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def run_swebench_execution_slice_patch_generation_v2(
    *,
    config: ProjectConfig,
    api_settings: ApiSettings,
    slice_manifest: Path,
    output_dir: Path,
    models: Iterable[str],
    split: str = "test",
    top_k: int = 50,
    fetch_repo_tree: bool = False,
    include_file_context: bool = False,
    max_context_files: int = 3,
    max_context_chars_per_file: int = 6000,
    max_context_candidate_attempts: int = 50,
    repo_checkout_root: Path | None = None,
    max_workers: int = 4,
    hard_call_cap: int = 500,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    selected_models = [str(model).strip() for model in models if str(model).strip()]
    if not selected_models:
        raise ValueError("at least one model is required")
    slice_frame = read_swebench_execution_slice_manifest(slice_manifest)
    instance_ids = slice_frame["instance_id"].astype(str).tolist()
    all_rows = fetch_swe_bench_lite_rows(split=split, limit=300, offset=0)
    row_map = {row.instance_id: row for row in all_rows}
    missing = [instance_id for instance_id in instance_ids if instance_id not in row_map]
    if missing:
        raise RuntimeError(f"slice instance_id rows were not found in {SWE_BENCH_DATASET}/{split}: {missing[:5]}")
    target_rows = [row_map[instance_id] for instance_id in instance_ids]
    repo_tree_cache: dict[tuple[str, str], list[str]] = {}
    file_context_cache: dict[tuple[str, str, str], str] = {}
    retrieval_cache: dict[str, list[str]] = {}
    file_context_cache_by_instance: dict[str, list[dict[str, Any]]] = {}
    prompt_cache: dict[str, str] = {}
    for row in target_rows:
        tree_files: list[str] = []
        if fetch_repo_tree and row.base_commit:
            key = (row.repo, row.base_commit)
            if key not in repo_tree_cache:
                try:
                    local_tree = fetch_local_git_tree_files(row.repo, row.base_commit, repo_checkout_root)
                    repo_tree_cache[key] = local_tree if local_tree else fetch_github_tree_files(row.repo, row.base_commit)
                except Exception:
                    repo_tree_cache[key] = []
            tree_files = repo_tree_cache[key]
        retrieved = oracle_free_retrieval_candidates(row, repo_tree_files=tree_files, top_k=top_k)
        retrieval_cache[row.instance_id] = retrieved
        file_contexts: list[dict[str, Any]] = []
        if include_file_context and row.base_commit:
            file_contexts = collect_repo_file_contexts(
                row,
                retrieved,
                max_files=max_context_files,
                max_chars_per_file=max_context_chars_per_file,
                max_candidate_attempts=max_context_candidate_attempts,
                cache=file_context_cache,
                checkout_root=repo_checkout_root,
            )
        file_context_cache_by_instance[row.instance_id] = file_contexts
        prompt_cache[row.instance_id] = build_patch_generation_prompt_v2(
            row,
            retrieved_files=retrieved,
            repo_tree_files=tree_files,
            file_contexts=file_contexts,
            max_candidates=top_k,
        )

    tasks: list[dict[str, Any]] = []
    for row in target_rows:
        for model in selected_models:
            task_key = f"{model}|{row.instance_id}|patch_generation"
            tasks.append(
                {
                    "task_key": task_key,
                    "model": model,
                    "row": row,
                    "prompt": prompt_cache[row.instance_id],
                    "retrieved_files": retrieval_cache[row.instance_id],
                    "file_contexts": file_context_cache_by_instance.get(row.instance_id, []),
                    "repo_tree_files": repo_tree_cache.get((row.repo, row.base_commit), []),
                }
            )
    planned_calls = int(len(tasks))
    manifest = {
        "experiment": "swebench_lite_execution_slice_patch_generation_v2",
        "dataset": SWE_BENCH_DATASET,
        "dataset_role": "generates model_patch JSONL for official SWE-bench patch/test execution; not an execution result",
        "split": split,
        "slice_manifest": str(slice_manifest),
        "slice_instance_count": int(len(instance_ids)),
        "models": selected_models,
        "planned_remote_calls": planned_calls,
        "hard_call_cap": int(hard_call_cap),
        "base_url": api_settings.base_url,
        "dry_run": bool(dry_run),
        "resume": bool(resume),
        "top_k": int(top_k),
        "fetch_repo_tree": bool(fetch_repo_tree),
        "include_file_context": bool(include_file_context),
        "max_context_files": int(max_context_files),
        "max_context_chars_per_file": int(max_context_chars_per_file),
        "max_context_candidate_attempts": int(max_context_candidate_attempts),
        "repo_checkout_root": str(repo_checkout_root) if repo_checkout_root is not None else "",
        "repo_checkout_root_available": bool(repo_checkout_root is not None and Path(repo_checkout_root).exists()),
        "prediction_schema": "jsonl rows with instance_id, model_name_or_path, model_patch",
        "oracle_free_prompt": True,
        "reference_patch_used_in_prompt": False,
        "checkpoint_path": "patch_generation_checkpoint.jsonl",
    }
    write_json(output_dir / "run_manifest.json", manifest)
    if planned_calls > int(hard_call_cap):
        raise RuntimeError(f"planned patch-generation calls {planned_calls} exceed hard-call cap {hard_call_cap}")

    if dry_run:
        planned_records = []
        for task in tasks:
            row: SweBenchRow = task["row"]
            retrieved = list(task["retrieved_files"])
            contexts = list(task.get("file_contexts", []))
            context_errors = [str(item.get("fetch_error", "")) for item in contexts if str(item.get("fetch_error", ""))]
            planned_records.append(
                {
                    "task_key": task["task_key"],
                    "model": task["model"],
                    "repo": row.repo,
                    "instance_id": row.instance_id,
                    "base_commit": row.base_commit,
                    "retrieved_candidate_count": int(len(retrieved)),
                    "prompt_candidate_files": "|".join(retrieved[:top_k]),
                    "file_context_file_count": int(sum(1 for item in contexts if str(item.get("content", "")))),
                    "file_context_chars": int(sum(len(str(item.get("content", ""))) for item in contexts)),
                    "file_context_fetch_errors": "|".join(context_errors[:8]),
                    "prompt_chars": int(len(str(task["prompt"]))),
                    "model_patch": "",
                    "target_files": "[]",
                    "defer": True,
                    "rationale": "dry-run preview only",
                    "confidence": 0.0,
                    "error_type": "",
                    "remote_used": False,
                }
        )
        planned_frame = pd.DataFrame(planned_records)
        planned_frame.to_csv(output_dir / "patch_generation_tasks.csv", index=False)
        predictions_manifest = _write_patch_generation_predictions(
            planned_frame,
            output_dir,
            models=selected_models,
            dry_run=True,
        )
        context_issue_rows = (
            int(
                planned_frame.loc[planned_frame["file_context_file_count"].astype(int).gt(0), "instance_id"]
                .astype(str)
                .nunique()
            )
            if "file_context_file_count" in planned_frame
            else 0
        )
        context_issue_rate = float(context_issue_rows / max(1, len(instance_ids)))
        context_status = "not_requested"
        if include_file_context:
            context_status = "pass" if context_issue_rate >= 0.8 else "needs_review"
        gate = {
            "status": "planned",
            **manifest,
            "observed_rows": int(len(planned_frame)),
            "prediction_files_status": "planned_preview_only",
            "all_prediction_rows_present": bool(len(planned_frame) == planned_calls),
            "file_context_rows": int(planned_frame["file_context_file_count"].astype(int).gt(0).sum())
            if "file_context_file_count" in planned_frame
            else 0,
            "file_context_total_chars": int(planned_frame["file_context_chars"].astype(int).sum())
            if "file_context_chars" in planned_frame
            else 0,
            "file_context_issue_rows": context_issue_rows,
            "file_context_issue_rate": context_issue_rate,
            "file_context_status": context_status,
        }
        write_json(output_dir / "gate_status.json", gate)
        write_patch_generation_report(output_dir, manifest=manifest, gate=gate, predictions_manifest=predictions_manifest)
        return {
            "tasks": planned_frame,
            "predictions_manifest": predictions_manifest,
            "gate_status": gate,
            "manifest": manifest,
        }

    os.environ["OPENAI_API_KEY"] = api_settings.api_key
    providers = {model: _make_provider(config, model=model, base_url=api_settings.base_url) for model in selected_models}
    defaults = {"model_patch": "", "target_files": [], "rationale": "default parse fallback", "confidence": 0.0, "defer": True}
    system_prompt = "You are a SWE-bench patch-generation agent. Output valid compact JSON only."

    def run_task(task: dict[str, Any]) -> dict[str, Any]:
        provider = providers[str(task["model"])]
        result = provider.chat_json_with_metadata(
            system_prompt,
            str(task["prompt"]),
            defaults,
            temperature_override=0.2,
            top_p_override=1.0,
        )
        payload = result.payload if isinstance(result.payload, dict) else defaults
        metadata = _metadata_subset(result.metadata)
        row: SweBenchRow = task["row"]
        model_patch = str(payload.get("model_patch", "") or "")
        target_files = _coerce_file_list(payload.get("target_files", []))
        defer = bool(payload.get("defer", False)) or not bool(model_patch.strip())
        contexts = list(task.get("file_contexts", []))
        context_errors = [str(item.get("fetch_error", "")) for item in contexts if str(item.get("fetch_error", ""))]
        return {
            "task_key": task["task_key"],
            "model": task["model"],
            "repo": row.repo,
            "instance_id": row.instance_id,
            "base_commit": row.base_commit,
            "retrieved_candidate_count": int(len(task["retrieved_files"])),
            "prompt_candidate_files": "|".join(list(task["retrieved_files"])[:top_k]),
            "file_context_file_count": int(sum(1 for item in contexts if str(item.get("content", "")))),
            "file_context_chars": int(sum(len(str(item.get("content", ""))) for item in contexts)),
            "file_context_fetch_errors": "|".join(context_errors[:8]),
            "model_patch": model_patch,
            "target_files": json.dumps(target_files, ensure_ascii=False),
            "defer": bool(defer),
            "rationale": str(payload.get("rationale", "")),
            "rationale_chars": int(len(str(payload.get("rationale", "")))),
            "confidence": _to_float(payload.get("confidence", float("nan"))),
            "patch_chars": int(len(model_patch)),
            "edits_tests": bool(includes_test_file(target_files)),
            **metadata,
        }

    def error_task_record(task: dict[str, Any], exc: BaseException) -> dict[str, Any]:
        row: SweBenchRow = task["row"]
        contexts = list(task.get("file_contexts", []))
        context_errors = [str(item.get("fetch_error", "")) for item in contexts if str(item.get("fetch_error", ""))]
        return {
            "task_key": task["task_key"],
            "model": task["model"],
            "repo": row.repo,
            "instance_id": row.instance_id,
            "base_commit": row.base_commit,
            "retrieved_candidate_count": int(len(task["retrieved_files"])),
            "prompt_candidate_files": "|".join(list(task["retrieved_files"])[:top_k]),
            "file_context_file_count": int(sum(1 for item in contexts if str(item.get("content", "")))),
            "file_context_chars": int(sum(len(str(item.get("content", ""))) for item in contexts)),
            "file_context_fetch_errors": "|".join(context_errors[:8]),
            "model_patch": "",
            "target_files": "[]",
            "defer": True,
            "rationale": f"provider task exception: {type(exc).__name__}",
            "rationale_chars": int(len(str(exc))),
            "confidence": float("nan"),
            "patch_chars": 0,
            "edits_tests": False,
            "provider_role": "real_task_predictive_validity",
            "base_url": api_settings.base_url,
            "json_mode": "json_object",
            "request_temperature": 0.2,
            "request_top_p": 1.0,
            "fallback_used": False,
            "error_type": f"task_exception_{type(exc).__name__}",
            "latency_ms": 0,
            "remote_used": False,
            "prompt_tokens": "",
            "completion_tokens": "",
            "total_tokens": "",
            "retry_count": "",
        }

    checkpoint_path = output_dir / "patch_generation_checkpoint.jsonl"
    existing_records = _checkpoint_records(checkpoint_path)
    completed_keys = {str(record.get("task_key", "")) for record in existing_records if record.get("task_key")}
    if checkpoint_path.exists() and existing_records and not resume:
        raise RuntimeError(f"checkpoint exists at {checkpoint_path}; use resume=True/--resume or a new run_id")
    pending_tasks = [task for task in tasks if str(task["task_key"]) not in completed_keys]
    records: list[dict[str, Any]] = list(existing_records)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_task, task): task for task in pending_tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                record = future.result()
            except BaseException as exc:
                record = error_task_record(task, exc)
            records.append(record)
            _append_checkpoint_record(checkpoint_path, record)

    trace_frame = pd.DataFrame(records)
    if "task_key" in trace_frame.columns:
        trace_frame = trace_frame.drop_duplicates("task_key", keep="last")
    trace_frame = trace_frame.sort_values(["model", "instance_id"]).reset_index(drop=True) if not trace_frame.empty else trace_frame
    trace_frame.to_csv(output_dir / "patch_generation_traces.csv", index=False)
    write_parquet(trace_frame, output_dir / "patch_generation_traces.parquet")
    predictions_manifest = _write_patch_generation_predictions(
        trace_frame,
        output_dir,
        models=selected_models,
        dry_run=False,
    )
    total_expected = int(len(selected_models) * len(instance_ids))
    error_rows = int(trace_frame["error_type"].fillna("").astype(str).ne("").sum()) if "error_type" in trace_frame else 0
    missing_rows = int(total_expected - len(trace_frame))
    context_issue_rows = (
        int(trace_frame.loc[trace_frame["file_context_file_count"].fillna(0).astype(int).gt(0), "instance_id"].astype(str).nunique())
        if "file_context_file_count" in trace_frame
        else 0
    )
    context_issue_rate = float(context_issue_rows / max(1, len(instance_ids)))
    context_status = "not_requested"
    if include_file_context:
        context_status = "pass" if context_issue_rate >= 0.8 else "needs_review"
    gate = {
        "status": "ready_for_official_execution"
        if missing_rows == 0 and error_rows == 0 and context_status != "needs_review"
        else "needs_review",
        **manifest,
        "observed_rows": int(len(trace_frame)),
        "missing_prediction_rows": missing_rows,
        "error_rows": error_rows,
        "nonempty_patch_rows": int(trace_frame["model_patch"].fillna("").astype(str).str.len().gt(0).sum())
        if "model_patch" in trace_frame
        else 0,
        "deferred_rows": int(trace_frame["defer"].fillna(False).astype(bool).sum()) if "defer" in trace_frame else 0,
        "file_context_rows": int(trace_frame["file_context_file_count"].fillna(0).astype(int).gt(0).sum())
        if "file_context_file_count" in trace_frame
        else 0,
        "file_context_total_chars": int(trace_frame["file_context_chars"].fillna(0).astype(int).sum())
        if "file_context_chars" in trace_frame
        else 0,
        "file_context_issue_rows": context_issue_rows,
        "file_context_issue_rate": context_issue_rate,
        "file_context_status": context_status,
        "checkpoint_rows_loaded": int(len(existing_records)),
        "checkpoint_tasks_skipped": int(len(tasks) - len(pending_tasks)),
        "checkpoint_tasks_run": int(len(pending_tasks)),
        "checkpoint_path": str(checkpoint_path),
        "prediction_files_status": "ready_for_official_harness" if missing_rows == 0 else "incomplete",
        "scope_note": "Run official SWE-bench harness and merge results before making execution-success claims.",
    }
    write_json(output_dir / "gate_status.json", gate)
    write_patch_generation_report(output_dir, manifest=manifest, gate=gate, predictions_manifest=predictions_manifest)
    return {
        "traces": trace_frame,
        "predictions_manifest": predictions_manifest,
        "gate_status": gate,
        "manifest": manifest,
    }


def summarize_official_execution_slice_validity(
    instance_summary: pd.DataFrame,
    official_results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if instance_summary.empty or official_results.empty:
        empty = pd.DataFrame()
        return empty, empty, {"status": "needs_review", "reason": "empty_input"}
    official = official_results.copy()
    official["model"] = official.get("model", "").fillna("").astype(str)
    model_specific = bool(official["model"].astype(str).str.len().gt(0).any())
    if model_specific:
        merged = instance_summary.merge(official, on=["model", "instance_id"], how="inner")
    else:
        merged = instance_summary.merge(official.drop(columns=["model"], errors="ignore"), on="instance_id", how="inner")
    if merged.empty:
        return merged, pd.DataFrame(), {"status": "needs_review", "reason": "no_matching_instances"}
    y = [int(value) for value in merged["official_resolved"].astype(bool)]
    predictor_specs = [
        ("state_binding_score", True),
        ("constraint_clean_hit_at3_rate", True),
        ("task_action_entropy", False),
        ("file_set_entropy", False),
        ("path_count_entropy", False),
        ("top_file_set_agreement", True),
        ("vote_margin", True),
        ("mean_rationale_chars", True),
        ("mean_confidence", True),
        ("retrieved_candidate_count", True),
    ]
    rows = []
    for predictor, higher_positive in predictor_specs:
        if predictor not in merged.columns:
            continue
        values = [float(value) for value in merged[predictor]]
        rows.append(
            {
                "predictor": predictor,
                "official_resolved_auc": _auc_score(y, values, higher_positive=higher_positive),
                "official_resolved_pearson": _pearson(y, values),
                "higher_predicts_resolved": bool(higher_positive),
            }
        )
    predictor_summary = pd.DataFrame(rows)
    state_auc = (
        float(predictor_summary.loc[predictor_summary["predictor"] == "state_binding_score", "official_resolved_auc"].iloc[0])
        if not predictor_summary.empty and (predictor_summary["predictor"] == "state_binding_score").any()
        else float("nan")
    )
    best_baseline = (
        float(
            predictor_summary.loc[
                predictor_summary["predictor"] != "state_binding_score", "official_resolved_auc"
            ].max(skipna=True)
        )
        if not predictor_summary.empty
        else float("nan")
    )
    gate = {
        "status": "pass" if math.isfinite(state_auc) and (not math.isfinite(best_baseline) or state_auc >= best_baseline) else "needs_review",
        "matched_rows": int(len(merged)),
        "matched_instances": int(merged["instance_id"].nunique()),
        "official_resolved_positive_rows": int(sum(y)),
        "official_resolved_negative_rows": int(len(y) - sum(y)),
        "state_binding_official_resolved_auc": state_auc,
        "best_baseline_official_resolved_auc": best_baseline,
        "model_specific_results": bool(model_specific),
        "scope_note": "Execution-slice association only. Official resolved/tests-pass outcomes must come from SWE-bench harness outputs.",
    }
    return merged, predictor_summary, gate


def run_swebench_predictive_validity(
    *,
    config: ProjectConfig,
    api_settings: ApiSettings,
    output_dir: Path,
    models: Iterable[str],
    instances: int = 16,
    offset: int = 0,
    split: str = "test",
    self_consistency_repeats: int = 3,
    include_patch_snippets: bool = False,
    max_workers: int = 8,
    hard_call_cap: int = 800,
    dry_run: bool = False,
) -> dict[str, Any]:
    os.environ["OPENAI_API_KEY"] = api_settings.api_key
    output_dir = ensure_dir(output_dir)
    selected_models = list(models)
    rows = fetch_swe_bench_lite_rows(split=split, limit=instances + 1, offset=offset)
    if len(rows) < instances + 1:
        raise RuntimeError(f"needed at least {instances + 1} rows, got {len(rows)}")
    rows = rows[: instances + 1]
    target_rows = rows[:instances]
    trial_conditions: list[tuple[str, int, float]] = [("task_only", 0, 0.2)]
    for repeat in range(1, self_consistency_repeats):
        trial_conditions.append(("task_only_self_consistency", repeat, 0.7))
    trial_conditions.extend(
        [
            ("decisive_state_current", 0, 0.2),
            ("decisive_state_only", 0, 0.2),
            ("scrambled_decisive_state", 0, 0.2),
            ("candidate_prior_only", 0, 0.2),
        ]
    )
    planned = int(len(selected_models) * len(target_rows) * len(trial_conditions))
    manifest = {
        "experiment": "swebench_lite_predictive_validity",
        "dataset": SWE_BENCH_DATASET,
        "split": split,
        "offset": int(offset),
        "instances": int(instances),
        "models": selected_models,
        "conditions": [condition for condition, _, _ in trial_conditions],
        "self_consistency_repeats": int(self_consistency_repeats),
        "include_patch_snippets": bool(include_patch_snippets),
        "planned_remote_calls": planned,
        "base_url": api_settings.base_url,
        "dry_run": bool(dry_run),
    }
    write_json(output_dir / "run_manifest.json", manifest)
    if planned > hard_call_cap:
        raise RuntimeError(f"planned calls {planned} exceed hard-call cap {hard_call_cap}")
    if dry_run:
        gate = {"status": "planned", **manifest}
        write_json(output_dir / "gate_status.json", gate)
        return {"gate_status": gate}

    providers = {model: _make_provider(config, model=model, base_url=api_settings.base_url) for model in selected_models}
    defaults = {"final_action": "PATCH_D", "rationale": "default parse fallback", "confidence": 0.0}
    system_prompt = "You are a software-repair evaluation agent. Output valid JSON only."
    tasks: list[dict[str, Any]] = []
    candidate_cache: dict[str, tuple[dict[str, dict[str, str]], str, str, SweBenchRow]] = {}
    for row in target_rows:
        candidates, expected_action, scrambled_action, scrambled_row = build_patch_candidates(
            rows,
            idx,
            include_patch_snippets=include_patch_snippets,
        )
        candidate_cache[row.instance_id] = (candidates, expected_action, scrambled_action, scrambled_row)
        for model in selected_models:
            for condition, repeat_index, temperature in trial_conditions:
                prompt = build_prompt(row, condition=condition, candidates=candidates, scrambled_row=scrambled_row)
                tasks.append(
                    {
                        "model": model,
                        "row": row,
                        "condition": condition,
                        "repeat_index": repeat_index,
                        "temperature": temperature,
                        "prompt": prompt,
                        "expected_action": expected_action,
                        "scrambled_action": scrambled_action,
                        "candidates": candidates,
                        "scrambled_row": scrambled_row,
                    }
                )

    def run_task(task: dict[str, Any]) -> dict[str, Any]:
        provider = providers[str(task["model"])]
        result = provider.chat_json_with_metadata(
            system_prompt,
            str(task["prompt"]),
            defaults,
            temperature_override=float(task["temperature"]),
        )
        payload = result.payload if isinstance(result.payload, dict) else defaults
        action = canonical_patch_action(payload.get("final_action"))
        candidates = task["candidates"]
        candidate = candidates.get(action, {})
        candidate_type = str(candidate.get("candidate_type", "invalid"))
        row: SweBenchRow = task["row"]
        metadata = _metadata_subset(result.metadata)
        return {
            "task_key": task["task_key"],
            "model": task["model"],
            "repo": row.repo,
            "instance_id": row.instance_id,
            "condition": task["condition"],
            "repeat_index": int(task["repeat_index"]),
            "expected_action": task["expected_action"],
            "scrambled_action": task["scrambled_action"],
            "final_action": action,
            "candidate_type": candidate_type,
            "expected_action_match": bool(action == task["expected_action"]),
            "follows_scrambled_action": bool(action == task["scrambled_action"]),
            "hard_constraint_violation": bool(candidate_type == "test_only_violation"),
            "memory_mismatch_failure": bool(candidate_type == "scrambled_issue_impl"),
            "no_op_failure": bool(candidate_type == "no_op"),
            "rationale": str(payload.get("rationale", "")),
            "rationale_chars": int(len(str(payload.get("rationale", "")))),
            "confidence": payload.get("confidence", ""),
            "problem_chars": int(len(row.problem_statement)),
            "candidate_files_correct": "|".join(changed_files(row.patch)[:5]),
            "candidate_files_scrambled": "|".join(changed_files(task["scrambled_row"].patch)[:5]),
            **metadata,
        }

    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_task, task) for task in tasks]
        for future in as_completed(futures):
            records.append(future.result())

    traces = pd.DataFrame(records).sort_values(["model", "instance_id", "condition", "repeat_index"]).reset_index(drop=True)
    write_parquet(traces, output_dir / "traces.parquet")
    traces.to_csv(output_dir / "traces.csv", index=False)
    instance_summary, predictor_summary, gate = summarize_predictive_validity(traces)
    write_parquet(instance_summary, output_dir / "instance_summary.parquet")
    write_parquet(predictor_summary, output_dir / "predictor_summary.parquet")
    instance_summary.to_csv(output_dir / "instance_summary.csv", index=False)
    predictor_summary.to_csv(output_dir / "predictor_summary.csv", index=False)
    gate.update(
        {
            "observed_rows": int(len(traces)),
            "error_rows": int(traces["error_type"].fillna("").astype(str).ne("").sum()) if "error_type" in traces else 0,
            "remote_rows": int(traces["remote_used"].fillna(False).astype(bool).sum()) if "remote_used" in traces else 0,
            "observed_remote_call_units": int(pd.to_numeric(traces.get("remote_call_units", pd.Series(dtype=int)), errors="coerce").fillna(1).sum()),
        }
    )
    write_json(output_dir / "gate_status.json", gate)
    write_report(output_dir, manifest=manifest, instance_summary=instance_summary, predictor_summary=predictor_summary, gate=gate)
    return {
        "traces": traces,
        "instance_summary": instance_summary,
        "predictor_summary": predictor_summary,
        "gate_status": gate,
    }


def run_swebench_file_localization_predictive_validity(
    *,
    config: ProjectConfig,
    api_settings: ApiSettings,
    output_dir: Path,
    models: Iterable[str],
    instances: int = 20,
    offset: int = 0,
    split: str = "test",
    self_consistency_repeats: int = 3,
    max_workers: int = 8,
    hard_call_cap: int = 1000,
    dry_run: bool = False,
) -> dict[str, Any]:
    os.environ["OPENAI_API_KEY"] = api_settings.api_key
    output_dir = ensure_dir(output_dir)
    selected_models = list(models)
    rows = fetch_swe_bench_lite_rows(split=split, limit=instances + 8, offset=offset)
    if len(rows) < instances + 1:
        raise RuntimeError(f"needed at least {instances + 1} rows, got {len(rows)}")
    target_rows = rows[:instances]
    trial_conditions: list[tuple[str, int, float]] = [("task_only", 0, 0.2)]
    for repeat in range(1, self_consistency_repeats):
        trial_conditions.append(("task_only_self_consistency", repeat, 0.7))
    trial_conditions.extend(
        [
            ("decisive_state_current", 0, 0.2),
            ("decisive_state_only", 0, 0.2),
            ("scrambled_decisive_state", 0, 0.2),
            ("prior_only", 0, 0.2),
        ]
    )
    planned = int(len(selected_models) * len(target_rows) * len(trial_conditions))
    manifest = {
        "experiment": "swebench_lite_file_localization_predictive_validity",
        "dataset": SWE_BENCH_DATASET,
        "split": split,
        "offset": int(offset),
        "instances": int(instances),
        "models": selected_models,
        "conditions": [condition for condition, _, _ in trial_conditions],
        "self_consistency_repeats": int(self_consistency_repeats),
        "planned_remote_calls": planned,
        "base_url": api_settings.base_url,
        "dry_run": bool(dry_run),
    }
    write_json(output_dir / "run_manifest.json", manifest)
    if planned > hard_call_cap:
        raise RuntimeError(f"planned calls {planned} exceed hard-call cap {hard_call_cap}")
    if dry_run:
        gate = {"status": "planned", **manifest}
        write_json(output_dir / "gate_status.json", gate)
        return {"gate_status": gate}

    providers = {model: _make_provider(config, model=model, base_url=api_settings.base_url) for model in selected_models}
    defaults = {"target_files": [], "rationale": "default parse fallback", "confidence": 0.0}
    system_prompt = "You are a software-repair evaluation agent. Output valid JSON only."
    tasks: list[dict[str, Any]] = []
    for idx, row in enumerate(target_rows):
        scrambled_row = select_scrambled_row(rows, idx)
        for model in selected_models:
            for condition, repeat_index, temperature in trial_conditions:
                tasks.append(
                    {
                        "model": model,
                        "row": row,
                        "scrambled_row": scrambled_row,
                        "condition": condition,
                        "repeat_index": repeat_index,
                        "temperature": temperature,
                        "prompt": build_localization_prompt(row, condition=condition, scrambled_row=scrambled_row),
                    }
                )

    def run_task(task: dict[str, Any]) -> dict[str, Any]:
        provider = providers[str(task["model"])]
        result = provider.chat_json_with_metadata(
            system_prompt,
            str(task["prompt"]),
            defaults,
            temperature_override=float(task["temperature"]),
        )
        payload = result.payload if isinstance(result.payload, dict) else defaults
        target_files = _coerce_file_list(payload.get("target_files", []))
        row: SweBenchRow = task["row"]
        scrambled_row: SweBenchRow = task["scrambled_row"]
        reference_files = changed_files(row.patch)
        scrambled_files = changed_files(scrambled_row.patch)
        metadata = _metadata_subset(result.metadata)
        return {
            "model": task["model"],
            "repo": row.repo,
            "instance_id": row.instance_id,
            "condition": task["condition"],
            "repeat_index": int(task["repeat_index"]),
            "target_files": json.dumps(target_files, ensure_ascii=False),
            "target_file_key": "|".join(sorted(path.lower() for path in target_files)),
            "reference_files": "|".join(reference_files[:8]),
            "scrambled_instance_id": scrambled_row.instance_id,
            "scrambled_files": "|".join(scrambled_files[:8]),
            "reference_file_match": bool(file_set_matches(target_files, reference_files)),
            "scrambled_file_match": bool(file_set_matches(target_files, scrambled_files)),
            "test_file_violation": bool(includes_test_file(target_files)),
            "empty_target": bool(not target_files),
            "rationale": str(payload.get("rationale", "")),
            "rationale_chars": int(len(str(payload.get("rationale", "")))),
            "confidence": payload.get("confidence", ""),
            "problem_chars": int(len(row.problem_statement)),
            **metadata,
        }

    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_task, task) for task in tasks]
        for future in as_completed(futures):
            records.append(future.result())

    traces = pd.DataFrame(records).sort_values(["model", "instance_id", "condition", "repeat_index"]).reset_index(drop=True)
    write_parquet(traces, output_dir / "traces.parquet")
    traces.to_csv(output_dir / "traces.csv", index=False)
    instance_summary, predictor_summary, gate = summarize_localization_predictive_validity(traces)
    write_parquet(instance_summary, output_dir / "instance_summary.parquet")
    write_parquet(predictor_summary, output_dir / "predictor_summary.parquet")
    instance_summary.to_csv(output_dir / "instance_summary.csv", index=False)
    predictor_summary.to_csv(output_dir / "predictor_summary.csv", index=False)
    gate.update(
        {
            "observed_rows": int(len(traces)),
            "error_rows": int(traces["error_type"].fillna("").astype(str).ne("").sum()) if "error_type" in traces else 0,
            "remote_rows": int(traces["remote_used"].fillna(False).astype(bool).sum()) if "remote_used" in traces else 0,
        }
    )
    write_json(output_dir / "gate_status.json", gate)
    write_localization_report(
        output_dir,
        manifest=manifest,
        instance_summary=instance_summary,
        predictor_summary=predictor_summary,
        gate=gate,
    )
    return {
        "traces": traces,
        "instance_summary": instance_summary,
        "predictor_summary": predictor_summary,
        "gate_status": gate,
    }


def run_swebench_file_localization_predictive_validity_v2(
    *,
    config: ProjectConfig,
    api_settings: ApiSettings,
    output_dir: Path,
    models: Iterable[str],
    instances: int = 300,
    offset: int = 0,
    split: str = "test",
    self_consistency_repeats: int = 3,
    diagnostic_repeats: int = 1,
    top_k: int = 50,
    fetch_repo_tree: bool = False,
    repo_checkout_root: Path | None = None,
    include_oracle_diagnostic: bool = False,
    include_wrapper_arms: bool = True,
    bootstrap_samples: int = 400,
    delta_auc_gate: float = 0.03,
    noninferiority_margin: float = -0.03,
    execution_slice_size: int = 0,
    max_workers: int = 8,
    hard_call_cap: int = 20000,
    resume: bool = False,
    shard_index: int = 0,
    shard_count: int = 1,
    dry_run: bool = False,
) -> dict[str, Any]:
    os.environ["OPENAI_API_KEY"] = api_settings.api_key
    output_dir = ensure_dir(output_dir)
    selected_models = list(models)
    context_limit = max(300, int(instances) + int(offset) + 32)
    rows = fetch_swe_bench_lite_rows(split=split, limit=context_limit, offset=0)
    full_target_rows, sampling_strategy = select_swebench_v2_sample(rows, instances=instances, offset=offset)
    if len(full_target_rows) < instances:
        raise RuntimeError(f"needed at least {instances} sampled rows, got {len(full_target_rows)}")
    shard_count = max(1, int(shard_count))
    shard_index = int(shard_index)
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(f"shard_index must satisfy 0 <= shard_index < shard_count, got {shard_index}/{shard_count}")
    target_rows = [row for idx, row in enumerate(full_target_rows) if idx % shard_count == shard_index]
    if not target_rows:
        raise RuntimeError(f"empty shard {shard_index}/{shard_count}")
    execution_slice, execution_slice_strategy = select_swebench_v2_sample(
        full_target_rows,
        instances=min(max(0, int(execution_slice_size)), len(full_target_rows)),
        offset=0,
        seed=31,
    )
    repo_tree_cache: dict[tuple[str, str], list[str]] = {}
    retrieval_cache: dict[str, list[str]] = {}
    scrambled_cache: dict[str, SweBenchRow] = {}
    scrambled_retrieval_cache: dict[str, list[str]] = {}
    prior_cache: dict[str, list[str]] = {}
    for idx, row in enumerate(target_rows):
        tree_files: list[str] = []
        if fetch_repo_tree and row.base_commit:
            key = (row.repo, row.base_commit)
            if key not in repo_tree_cache:
                try:
                    local_tree = fetch_local_git_tree_files(row.repo, row.base_commit, repo_checkout_root)
                    repo_tree_cache[key] = local_tree if local_tree else fetch_github_tree_files(row.repo, row.base_commit)
                except Exception:
                    repo_tree_cache[key] = []
            tree_files = repo_tree_cache[key]
        retrieval_cache[row.instance_id] = oracle_free_retrieval_candidates(row, repo_tree_files=tree_files, top_k=top_k)
        scrambled = select_scrambled_for_row(rows, row)
        scrambled_cache[row.instance_id] = scrambled
        scrambled_tree = repo_tree_cache.get((scrambled.repo, scrambled.base_commit), []) if fetch_repo_tree else []
        scrambled_retrieval_cache[row.instance_id] = oracle_free_retrieval_candidates(
            scrambled,
            repo_tree_files=scrambled_tree,
            top_k=top_k,
        )
        prior_cache[row.instance_id] = conflicting_prior_files(
            current_retrieved=retrieval_cache[row.instance_id],
            scrambled_retrieved=scrambled_retrieval_cache[row.instance_id],
            top_k=top_k,
        )

    trial_conditions: list[tuple[str, str, int, float]] = [("raw", "task_only", 0, 0.2)]
    for repeat in range(1, self_consistency_repeats):
        trial_conditions.append(("raw", "task_only_self_consistency", repeat, 0.7))
    for repeat in range(max(1, diagnostic_repeats)):
        trial_conditions.extend(
            [
                ("raw", "decisive_state_current", repeat, 0.2),
                ("raw", "decisive_state_only", repeat, 0.2),
                ("raw", "scrambled_decisive_state", repeat, 0.2),
                ("raw", "prior_only", repeat, 0.2),
                ("raw", "irrelevant_cue", repeat, 0.2),
            ]
        )
        if include_oracle_diagnostic:
            trial_conditions.extend(
                [
                    ("raw", "oracle_state_current", repeat, 0.2),
                    ("raw", "oracle_state_only", repeat, 0.2),
                ]
            )
    if include_wrapper_arms:
        trial_conditions.extend(
            [
                ("binding_guard", "task_only", 0, 0.2),
                ("compute_matched_repair", "task_only", 0, 0.2),
            ]
        )
    call_units_per_issue_model = sum(2 if arm == "compute_matched_repair" else 1 for arm, _, _, _ in trial_conditions)
    planned = int(len(selected_models) * len(target_rows) * call_units_per_issue_model)
    planned_task_rows = int(len(selected_models) * len(target_rows) * len(trial_conditions))
    manifest = {
        "experiment": "swebench_lite_file_localization_predictive_validity_v2",
        "dataset": SWE_BENCH_DATASET,
        "dataset_role": "issue-to-file localization; not full SWE-bench patch/test execution",
        "split": split,
        "offset": int(offset),
        "instances": int(instances),
        "full_sample_issue_rows": int(len(full_target_rows)),
        "shard_index": int(shard_index),
        "shard_count": int(shard_count),
        "shard_issue_rows": int(len(target_rows)),
        "sampling_strategy": sampling_strategy,
        "models": selected_models,
        "conditions": sorted({condition for _, condition, _, _ in trial_conditions}),
        "wrapper_arms": sorted({arm for arm, _, _, _ in trial_conditions}),
        "self_consistency_repeats": int(self_consistency_repeats),
        "diagnostic_repeats": int(diagnostic_repeats),
        "top_k": int(top_k),
        "fetch_repo_tree": bool(fetch_repo_tree),
        "repo_checkout_root": str(repo_checkout_root) if repo_checkout_root is not None else "",
        "repo_checkout_root_available": bool(repo_checkout_root is not None and Path(repo_checkout_root).exists()),
        "include_oracle_diagnostic": bool(include_oracle_diagnostic),
        "include_wrapper_arms": bool(include_wrapper_arms),
        "primary_predictor": "oracle_free_state_binding_score",
        "primary_outcome": "task_only_implementation_file_hit_at3",
        "planned_remote_calls": planned,
        "planned_task_rows": planned_task_rows,
        "hard_call_cap": int(hard_call_cap),
        "base_url": api_settings.base_url,
        "dry_run": bool(dry_run),
        "resume": bool(resume),
        "checkpoint_path": "traces_checkpoint.jsonl",
        "bootstrap_samples": int(bootstrap_samples),
        "delta_auc_gate": float(delta_auc_gate),
        "noninferiority_margin": float(noninferiority_margin),
        "execution_slice_size": int(len(execution_slice)),
        "execution_slice_strategy": execution_slice_strategy,
    }
    write_json(output_dir / "run_manifest.json", manifest)
    if execution_slice:
        execution_rows = [
            {
                "repo": row.repo,
                "instance_id": row.instance_id,
                "base_commit": row.base_commit,
                "version": row.version,
                "problem_chars": len(row.problem_statement),
                "patch_size": _patch_size(row),
            }
            for row in execution_slice
        ]
        execution_frame = pd.DataFrame(execution_rows)
        execution_frame.to_csv(output_dir / "official_execution_slice_instances.csv", index=False)
        (output_dir / "official_execution_slice_instances.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in execution_rows) + "\n",
            encoding="utf-8",
        )
    if planned > hard_call_cap:
        raise RuntimeError(f"planned calls {planned} exceed hard-call cap {hard_call_cap}")
    if dry_run:
        planned_rows = []
        for row in target_rows:
            retrieved = retrieval_cache.get(row.instance_id, [])
            scrambled = scrambled_cache.get(row.instance_id)
            prior_files = prior_cache.get(row.instance_id, [])
            planned_rows.append(
                {
                    "repo": row.repo,
                    "instance_id": row.instance_id,
                    "scrambled_instance_id": scrambled.instance_id if scrambled else "",
                    "scrambled_repo": scrambled.repo if scrambled else "",
                    "scrambled_same_repo": bool(scrambled and scrambled.repo == row.repo),
                    "problem_chars": len(row.problem_statement),
                    "retrieved_candidate_count": len(retrieved),
                    "prior_candidate_count": len(prior_files),
                    "prior_differs_from_current_top1": bool(
                        prior_files and retrieved and str(prior_files[0]).replace("\\", "/") != str(retrieved[0]).replace("\\", "/")
                    ),
                    "retrieved_candidate_count": int(len(retrieved[:top_k])),
                }
            )
        planned_frame = pd.DataFrame(planned_rows)
        planned_frame.to_csv(output_dir / "planned_issue_retrieval_audit.csv", index=False)
        gate = {
            "status": "planned",
            **manifest,
            "planned_issue_rows": int(len(planned_frame)),
            "planned_scrambled_same_repo_rate": float(planned_frame["scrambled_same_repo"].mean()) if not planned_frame.empty else float("nan"),
            "planned_prior_differs_from_current_top1_rate": float(planned_frame["prior_differs_from_current_top1"].mean()) if not planned_frame.empty else float("nan"),
        }
        coverage = localization_v2_protocol_coverage(manifest, gate)
        gate["protocol_coverage_status"] = coverage["status"]
        gate["protocol_coverage_missing_or_needs_review"] = coverage["missing_or_needs_review"]
        write_json(output_dir / "gate_status.json", gate)
        write_json(output_dir / "protocol_coverage.json", coverage)
        return {"gate_status": gate, "planned_issue_retrieval_audit": planned_frame, "protocol_coverage": coverage}

    providers = {model: _make_provider(config, model=model, base_url=api_settings.base_url) for model in selected_models}
    defaults = {"target_files": [], "rationale": "default parse fallback", "confidence": 0.0, "defer": False}
    system_prompt = "You are a software-repair file-localization evaluation agent. Output valid JSON only."
    tasks: list[dict[str, Any]] = []
    for row in target_rows:
        retrieved = retrieval_cache[row.instance_id]
        prior_files = prior_cache[row.instance_id]
        scrambled_row = scrambled_cache[row.instance_id]
        scrambled_retrieved = scrambled_retrieval_cache[row.instance_id]
        repo_tree = repo_tree_cache.get((row.repo, row.base_commit), [])
        for model in selected_models:
            for wrapper_arm, condition, repeat_index, temperature in trial_conditions:
                prompt_candidate_files = prior_files if condition == "prior_only" else retrieved
                task_key = f"{model}|{row.instance_id}|{wrapper_arm}|{condition}|{repeat_index}"
                tasks.append(
                    {
                        "task_key": task_key,
                        "model": model,
                        "row": row,
                        "scrambled_row": scrambled_row,
                        "retrieved_files": retrieved,
                        "scrambled_retrieved_files": scrambled_retrieved,
                        "prior_files": prior_files,
                        "prompt_candidate_files": prompt_candidate_files,
                        "repo_tree_files": repo_tree,
                        "condition": condition,
                        "wrapper_arm": wrapper_arm,
                        "repeat_index": repeat_index,
                        "temperature": temperature,
                        "prompt": build_localization_prompt_v2(
                            row,
                            condition=condition,
                            scrambled_row=scrambled_row,
                            retrieved_files=retrieved,
                            scrambled_retrieved_files=scrambled_retrieved,
                            prior_files=prior_files,
                            repo_tree_files=repo_tree,
                            wrapper_arm=wrapper_arm,
                        ),
                    }
                )

    def run_task(task: dict[str, Any]) -> dict[str, Any]:
        provider = providers[str(task["model"])]
        result = provider.chat_json_with_metadata(
            system_prompt,
            str(task["prompt"]),
            defaults,
            temperature_override=float(task["temperature"]),
        )
        payload = result.payload if isinstance(result.payload, dict) else defaults
        metadata = _metadata_subset(result.metadata)
        initial_target_files = _coerce_file_list(payload.get("target_files", []))
        remote_call_units = 1
        if task["wrapper_arm"] == "compute_matched_repair":
            repair_prompt = (
                f"{task['prompt']}\n"
                f"Previous JSON output: {json.dumps(payload, ensure_ascii=False)}\n"
                "Use this one extra compute-matched retry to revise target_files only if the previous output "
                "violated the hard constraints or missed a more constraint-clean implementation target. "
                "Return the final compact JSON object."
            )
            repair_result = provider.chat_json_with_metadata(
                system_prompt,
                repair_prompt,
                defaults,
                temperature_override=float(task["temperature"]),
            )
            if isinstance(repair_result.payload, dict):
                payload = repair_result.payload
            repair_metadata = _metadata_subset(repair_result.metadata)
            metadata.update({f"repair_{key}": value for key, value in repair_metadata.items()})
            remote_call_units = 2
        target_files_raw = _coerce_file_list(payload.get("target_files", []))
        row: SweBenchRow = task["row"]
        scrambled_row: SweBenchRow = task["scrambled_row"]
        reference_files = implementation_files(changed_files(row.patch))
        scrambled_files = implementation_files(changed_files(scrambled_row.patch))
        retrieved_files = list(task["retrieved_files"])
        prior_files = list(task["prior_files"])[:3]
        prompt_candidate_files = list(task["prompt_candidate_files"])
        repo_tree_files = list(task["repo_tree_files"])
        candidate_set = set(prompt_candidate_files)
        repo_tree_set = set(repo_tree_files)
        candidate_absent = bool(candidate_set and any(path not in candidate_set for path in target_files_raw))
        repo_nonexistent = bool(repo_tree_set and any(path not in repo_tree_set for path in target_files_raw))
        nonexistent = bool(candidate_absent or repo_nonexistent)
        test_violation = bool(includes_test_file(target_files_raw))
        scrambled_match = bool(file_set_matches(target_files_raw, scrambled_files))
        empty_raw = bool(not target_files_raw)
        defer = bool(payload.get("defer", False))
        hard_violation_raw = bool(test_violation or nonexistent or scrambled_match or (empty_raw and not defer))
        target_files = target_files_raw
        if task["wrapper_arm"] == "binding_guard" and hard_violation_raw:
            target_files = []
            defer = True
            empty_raw = False
            hard_violation_raw = False
        metrics = retrieval_metrics(retrieved_files, reference_files, top_k=top_k)
        clean_hit_at3 = bool(
            constraint_clean_file_hit_at_k(target_files, reference_files, k=3, repo_tree_files=repo_tree_files)
            and not hard_violation_raw
            and not defer
        )
        clean_hit_at1 = bool(
            constraint_clean_file_hit_at_k(target_files, reference_files, k=1, repo_tree_files=repo_tree_files)
            and not hard_violation_raw
            and not defer
        )
        return {
            "task_key": task["task_key"],
            "model": task["model"],
            "repo": row.repo,
            "instance_id": row.instance_id,
            "condition": task["condition"],
            "wrapper_arm": task["wrapper_arm"],
            "repeat_index": int(task["repeat_index"]),
            "target_files": json.dumps(target_files, ensure_ascii=False),
            "target_file_key": "|".join(sorted(path.lower() for path in target_files)),
            "raw_target_files": json.dumps(target_files_raw, ensure_ascii=False),
            "initial_target_files": json.dumps(initial_target_files, ensure_ascii=False),
            "reference_files": "|".join(reference_files[:8]),
            "scrambled_instance_id": scrambled_row.instance_id,
            "scrambled_files": "|".join(scrambled_files[:8]),
            "retrieved_files": "|".join(retrieved_files[:top_k]),
            "prior_files": "|".join(prior_files),
            "prompt_candidate_files": "|".join(prompt_candidate_files[:top_k]),
            "reference_file_match": bool(file_set_matches(target_files, reference_files)),
            "exact_file_set_match": bool(exact_file_set_matches(target_files, reference_files)),
            "constraint_clean_hit_at1": clean_hit_at1,
            "constraint_clean_hit_at3": clean_hit_at3,
            "file_set_f1": file_set_f1(target_files, reference_files),
            "scrambled_file_match": bool(file_set_matches(target_files, scrambled_files)),
            "prior_file_match": bool(file_set_matches(target_files, prior_files)),
            "test_file_violation": bool(test_violation),
            "candidate_absent_violation": bool(candidate_absent),
            "nonexistent_path_violation": bool(nonexistent),
            "repo_nonexistent_path_violation": bool(repo_nonexistent),
            "empty_target": bool(empty_raw and not defer),
            "deferred": bool(defer),
            "hard_constraint_violation": bool(hard_violation_raw),
            "rationale": str(payload.get("rationale", "")),
            "rationale_chars": int(len(str(payload.get("rationale", "")))),
            "confidence": _to_float(payload.get("confidence", float("nan"))),
            "problem_chars": int(len(row.problem_statement)),
            "remote_call_units": int(remote_call_units),
            "repair_attempted": bool(task["wrapper_arm"] == "compute_matched_repair"),
            **metrics,
            **metadata,
        }

    def error_task_record(task: dict[str, Any], exc: BaseException) -> dict[str, Any]:
        row: SweBenchRow = task["row"]
        scrambled_row: SweBenchRow = task["scrambled_row"]
        reference_files = implementation_files(changed_files(row.patch))
        scrambled_files = implementation_files(changed_files(scrambled_row.patch))
        retrieved_files = list(task["retrieved_files"])
        prior_files = list(task["prior_files"])[:3]
        metrics = retrieval_metrics(retrieved_files, reference_files, top_k=top_k)
        return {
            "task_key": task["task_key"],
            "model": task["model"],
            "repo": row.repo,
            "instance_id": row.instance_id,
            "condition": task["condition"],
            "wrapper_arm": task["wrapper_arm"],
            "repeat_index": int(task["repeat_index"]),
            "target_files": "[]",
            "target_file_key": "",
            "raw_target_files": "[]",
            "initial_target_files": "[]",
            "reference_files": "|".join(reference_files[:8]),
            "scrambled_instance_id": scrambled_row.instance_id,
            "scrambled_files": "|".join(scrambled_files[:8]),
            "retrieved_files": "|".join(retrieved_files[:top_k]),
            "prior_files": "|".join(prior_files),
            "prompt_candidate_files": "|".join(list(task["prompt_candidate_files"])[:top_k]),
            "reference_file_match": False,
            "exact_file_set_match": False,
            "constraint_clean_hit_at1": False,
            "constraint_clean_hit_at3": False,
            "file_set_f1": 0.0,
            "scrambled_file_match": False,
            "prior_file_match": False,
            "test_file_violation": False,
            "candidate_absent_violation": False,
            "nonexistent_path_violation": False,
            "repo_nonexistent_path_violation": False,
            "empty_target": True,
            "deferred": False,
            "hard_constraint_violation": True,
            "rationale": f"provider task exception: {type(exc).__name__}",
            "rationale_chars": int(len(str(exc))),
            "confidence": float("nan"),
            "problem_chars": int(len(row.problem_statement)),
            "remote_call_units": 0,
            "repair_attempted": bool(task["wrapper_arm"] == "compute_matched_repair"),
            "provider_role": "real_task_predictive_validity",
            "model": task["model"],
            "base_url": api_settings.base_url,
            "json_mode": "json_object",
            "request_temperature": task["temperature"],
            "request_top_p": 1.0,
            "fallback_used": False,
            "error_type": f"task_exception_{type(exc).__name__}",
            "latency_ms": 0,
            "remote_used": False,
            "prompt_tokens": "",
            "completion_tokens": "",
            "total_tokens": "",
            "retry_count": "",
            **metrics,
        }

    records: list[dict[str, Any]] = []
    checkpoint_path = output_dir / "traces_checkpoint.jsonl"
    existing_records = _checkpoint_records(checkpoint_path)
    completed_keys = {str(record.get("task_key", "")) for record in existing_records if record.get("task_key")}
    if checkpoint_path.exists() and existing_records and not resume:
        raise RuntimeError(f"checkpoint exists at {checkpoint_path}; use resume=True/--resume or a new run_id")
    records.extend(existing_records)
    pending_tasks = [task for task in tasks if str(task["task_key"]) not in completed_keys]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_task, task): task for task in pending_tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                record = future.result()
            except BaseException as exc:
                record = error_task_record(task, exc)
            records.append(record)
            _append_checkpoint_record(checkpoint_path, record)

    traces = pd.DataFrame(records).sort_values(["model", "instance_id", "wrapper_arm", "condition", "repeat_index"]).reset_index(drop=True)
    if "task_key" in traces.columns:
        traces = traces.drop_duplicates("task_key", keep="last")
        traces = traces.sort_values(["model", "instance_id", "wrapper_arm", "condition", "repeat_index"]).reset_index(drop=True)
    write_parquet(traces, output_dir / "traces.parquet")
    traces.to_csv(output_dir / "traces.csv", index=False)
    instance_summary, predictor_summary, gate = summarize_localization_predictive_validity_v2(
        traces,
        bootstrap_samples=bootstrap_samples,
        delta_auc_gate=delta_auc_gate,
        noninferiority_margin=noninferiority_margin,
    )
    write_parquet(instance_summary, output_dir / "instance_summary.parquet")
    write_parquet(predictor_summary, output_dir / "predictor_summary.parquet")
    instance_summary.to_csv(output_dir / "instance_summary.csv", index=False)
    predictor_summary.to_csv(output_dir / "predictor_summary.csv", index=False)
    gate.update(
        {
            "planned_remote_calls": int(planned),
            "planned_task_rows": int(planned_task_rows),
            "full_sample_issue_rows": int(len(full_target_rows)),
            "shard_index": int(shard_index),
            "shard_count": int(shard_count),
            "shard_issue_rows": int(len(target_rows)),
            "observed_rows": int(len(traces)),
            "error_rows": int(traces["error_type"].fillna("").astype(str).ne("").sum()) if "error_type" in traces else 0,
            "remote_rows": int(traces["remote_used"].fillna(False).astype(bool).sum()) if "remote_used" in traces else 0,
            "observed_remote_call_units": int(
                pd.to_numeric(traces.get("remote_call_units", pd.Series(dtype=int)), errors="coerce").fillna(1).sum()
            ),
            "checkpoint_rows_loaded": int(len(existing_records)),
            "checkpoint_tasks_skipped": int(len(tasks) - len(pending_tasks)),
            "checkpoint_tasks_run": int(len(pending_tasks)),
            "checkpoint_path": str(checkpoint_path),
        }
    )
    coverage = localization_v2_protocol_coverage(manifest, gate)
    gate["protocol_coverage_status"] = coverage["status"]
    gate["protocol_coverage_missing_or_needs_review"] = coverage["missing_or_needs_review"]
    write_json(output_dir / "gate_status.json", gate)
    write_json(output_dir / "protocol_coverage.json", coverage)
    write_localization_v2_report(
        output_dir,
        manifest=manifest,
        instance_summary=instance_summary,
        predictor_summary=predictor_summary,
        gate=gate,
    )
    return {
        "traces": traces,
        "instance_summary": instance_summary,
        "predictor_summary": predictor_summary,
        "gate_status": gate,
        "protocol_coverage": coverage,
    }

