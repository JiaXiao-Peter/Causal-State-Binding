from __future__ import annotations

import json
import math
import platform
import re
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from freewill.causal_state_sufficiency import decisive_field_text
from freewill.entropy_metrics import shannon_entropy
from freewill.structured_control_validation import (
    FINITE_ACTIONS,
    CSB_TASK_FAMILIES,
    _offline_distribution_policy,
    build_CSB_pilot_events,
    filter_CSB_event_indices,
)
from freewill.utils import ensure_dir, write_json, write_parquet


STRICT_MATCHING_CONDITIONS = ["structured_state", "entropy_prior_no_state"]


@dataclass(frozen=True)
class LocalGenerationConfig:
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    max_new_tokens: int = 32
    temperature: float = 0.7
    top_p: float = 0.9
    prompt_pad_token: str = " neutral"


def canonical_action(value: Any) -> str:
    text = str(value or "").upper()
    for action in FINITE_ACTIONS:
        if action in text:
            return action
    return "INVALID"


def build_strict_prompt(row: dict[str, Any], *, condition: str, prior_action: str) -> str:
    expected = str(row.get("expected_action", "")).strip()
    if condition == "structured_state":
        state = decisive_field_text(row, expected)
        condition_text = "Use the event-specific decisive state."
    elif condition == "entropy_prior_no_state":
        state = (
            f"No event-specific reason, memory, self-state, or veto field is available. "
            f"The only permitted control signal is ACTION_PRIOR={prior_action}. "
            "For this entropy-prior control, final_action must equal ACTION_PRIOR. "
            "This prior is distribution-matched but not tied to this event."
        )
        condition_text = "Use only the visible distribution-matched action prior."
    else:
        raise ValueError(f"unknown condition: {condition}")
    return (
        "Return one compact line with final_action=<ACTION> and no explanation.\n"
        f"Condition: {condition}\n"
        f"Instruction: {condition_text}\n"
        f"Event id: {row.get('event_id')}\n"
        f"Task family: {row.get('task_family')}\n"
        f"Candidate actions: {', '.join(FINITE_ACTIONS)}\n"
        f"State: {state}\n"
        "Final:"
    )


def token_len(tokenizer: Any, text: str) -> int:
    return int(len(tokenizer(text, add_special_tokens=False)["input_ids"]))


def pad_prompt_to_tokens(tokenizer: Any, prompt: str, target_tokens: int, *, pad_token: str = " neutral") -> str:
    final_marker = "\nFinal:"
    if final_marker in prompt:
        prefix, suffix = prompt.rsplit(final_marker, 1)
        working = prefix
        final_suffix = final_marker + suffix
    else:
        working = prompt
        final_suffix = ""
    suffix_tokens = token_len(tokenizer, final_suffix) if final_suffix else 0
    target_prefix_tokens = target_tokens - suffix_tokens
    if target_prefix_tokens < 0:
        raise ValueError(f"target {target_tokens} is shorter than final suffix length {suffix_tokens}")
    current = token_len(tokenizer, working)
    if current > target_prefix_tokens:
        raise ValueError(f"prompt already has {current} prefix tokens > target prefix {target_prefix_tokens}")
    while current < target_prefix_tokens:
        candidate = working + pad_token
        candidate_len = token_len(tokenizer, candidate)
        if candidate_len > target_prefix_tokens:
            break
        working = candidate
        current = candidate_len
    if current != target_prefix_tokens:
        # Fall back to a small search over printable fillers.
        fillers = [" x", " a", " 0", ".", "\n"]
        while current < target_prefix_tokens:
            added = False
            for filler in fillers:
                candidate = working + filler
                candidate_len = token_len(tokenizer, candidate)
                if candidate_len <= target_prefix_tokens:
                    working = candidate
                    current = candidate_len
                    added = True
                    break
            if not added:
                raise ValueError(f"could not pad prompt prefix to exactly {target_prefix_tokens} tokens; stopped at {current}")
    padded = working + final_suffix
    if token_len(tokenizer, padded) != target_tokens:
        raise ValueError(f"internal padding error: got {token_len(tokenizer, padded)} tokens, expected {target_tokens}")
    return padded


def _generate_one(
    *,
    model: Any,
    tokenizer: Any,
    prompt: str,
    generation_config: LocalGenerationConfig,
) -> tuple[str, dict[str, Any]]:
    encoded = tokenizer(prompt, return_tensors="pt")
    prompt_tokens = int(encoded["input_ids"].shape[1])
    tracemalloc.start()
    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            do_sample=True,
            temperature=float(generation_config.temperature),
            top_p=float(generation_config.top_p),
            min_new_tokens=int(generation_config.max_new_tokens),
            max_new_tokens=int(generation_config.max_new_tokens),
            pad_token_id=tokenizer.eos_token_id,
        )
    latency_ms = int((time.perf_counter() - start) * 1000)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    generated_ids = output[0, prompt_tokens:]
    completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
    metadata = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": int(generated_ids.shape[0]),
        "total_tokens": int(prompt_tokens + generated_ids.shape[0]),
        "latency_ms": latency_ms,
        "cpu_peak_tracemalloc_bytes": int(peak_bytes),
    }
    return completion, metadata


def _entropy_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for condition, group in frame.groupby("condition", sort=True):
        actions = list(group["final_action"].astype(str))
        rows.append(
            {
                "condition": condition,
                "rows": int(len(group)),
                "accuracy": float(group["expected_action_match"].mean()),
                "invalid_rate": float((group["final_action"] == "INVALID").mean()),
                "H_canonical_action": shannon_entropy(actions),
                "mean_prompt_tokens": float(group["prompt_tokens"].mean()),
                "mean_completion_tokens": float(group["completion_tokens"].mean()),
                "mean_total_tokens": float(group["total_tokens"].mean()),
                "mean_latency_ms": float(group["latency_ms"].mean()),
                "mean_cpu_peak_tracemalloc_bytes": float(group["cpu_peak_tracemalloc_bytes"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _matching_gate(summary: pd.DataFrame, traces: pd.DataFrame) -> dict[str, Any]:
    values = {row["condition"]: row for row in summary.to_dict(orient="records")}
    structured = values.get("structured_state", {})
    control = values.get("entropy_prior_no_state", {})
    prompt_match_rate = float(
        traces.groupby("event_id")["prompt_tokens"].nunique().eq(1).mean()
    ) if not traces.empty else float("nan")
    completion_match_rate = float(
        traces.groupby("event_id")["completion_tokens"].nunique().eq(1).mean()
    ) if not traces.empty else float("nan")
    total_ratio = float(control.get("mean_total_tokens", float("nan")) / structured.get("mean_total_tokens", float("nan")))
    latency_ratio = float(control.get("mean_latency_ms", float("nan")) / structured.get("mean_latency_ms", float("nan")))
    entropy_gap = abs(float(control.get("H_canonical_action", float("nan"))) - float(structured.get("H_canonical_action", float("nan"))))
    accuracy_delta = float(structured.get("accuracy", float("nan")) - control.get("accuracy", float("nan")))
    strict_pass = bool(
        math.isfinite(entropy_gap)
        and entropy_gap <= 0.15
        and prompt_match_rate >= 1.0
        and completion_match_rate >= 1.0
        and 0.95 <= total_ratio <= 1.05
        and 0.50 <= latency_ratio <= 1.50
    )
    return {
        "status": "pass" if strict_pass else "needs_review",
        "prompt_match_rate": prompt_match_rate,
        "completion_match_rate": completion_match_rate,
        "mean_total_token_ratio_control_over_structured": total_ratio,
        "mean_latency_ratio_control_over_structured": latency_ratio,
        "canonical_entropy_gap_bits": entropy_gap,
        "structured_accuracy": float(structured.get("accuracy", float("nan"))),
        "control_accuracy": float(control.get("accuracy", float("nan"))),
        "accuracy_delta": accuracy_delta,
        "gate_rule": "prompt and completion tokens exactly matched by event; total-token ratio within [0.95,1.05]; latency ratio within [0.50,1.50]; canonical action entropy gap <=0.15 bits",
    }


def run_local_strict_matching_audit(
    *,
    output_dir: Path,
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
    events_per_family: int = 8,
    event_index_start: int = 0,
    event_index_end: int = 2,
    task_families: Iterable[str] | None = None,
    max_new_tokens: int = 32,
    temperature: float = 0.7,
    top_p: float = 0.9,
    dry_run: bool = False,
) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    generation_config = LocalGenerationConfig(
        model_id=model_id,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    events = build_CSB_pilot_events(events_per_family=events_per_family, task_families=task_families or CSB_TASK_FAMILIES)
    events = filter_CSB_event_indices(events, start=event_index_start, end=event_index_end).reset_index(drop=True)
    seeds = [1]
    offline_actions, offline_policy = _offline_distribution_policy(events, ["strict_matching"], seeds)
    planned = int(len(events) * len(STRICT_MATCHING_CONDITIONS))
    manifest = {
        "experiment": "local_open_weight_strict_entropy_token_matching",
        "model_id": model_id,
        "events_per_family": int(events_per_family),
        "event_index_start": int(event_index_start),
        "event_index_end": int(event_index_end),
        "task_family_count": int(events["task_family"].nunique()) if not events.empty else 0,
        "event_count": int(len(events)),
        "conditions": STRICT_MATCHING_CONDITIONS,
        "planned_generations": planned,
        "max_new_tokens": int(max_new_tokens),
        "min_new_tokens": int(max_new_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "offline_action_prior_policy": offline_policy,
        "hardware": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
        },
        "dry_run": bool(dry_run),
    }
    write_json(output_dir / "run_manifest.json", manifest)
    if dry_run:
        gate = {"status": "planned", **manifest}
        write_json(output_dir / "gate_status.json", gate)
        return {"gate_status": gate}

    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, local_files_only=True)
    model.eval()
    records: list[dict[str, Any]] = []
    for row in events.to_dict(orient="records"):
        event_id = str(row.get("event_id", ""))
        prior_action = offline_actions[("strict_matching", 1, event_id)]
        base_prompts = {
            condition: build_strict_prompt(row, condition=condition, prior_action=prior_action)
            for condition in STRICT_MATCHING_CONDITIONS
        }
        max_prompt_tokens = max(token_len(tokenizer, prompt) for prompt in base_prompts.values()) + 8
        prompts = {
            condition: pad_prompt_to_tokens(
                tokenizer,
                prompt,
                max_prompt_tokens,
                pad_token=generation_config.prompt_pad_token,
            )
            for condition, prompt in base_prompts.items()
        }
        for condition in STRICT_MATCHING_CONDITIONS:
            completion, metadata = _generate_one(
                model=model,
                tokenizer=tokenizer,
                prompt=prompts[condition],
                generation_config=generation_config,
            )
            action = canonical_action(completion)
            records.append(
                {
                    "model_id": model_id,
                    "event_id": event_id,
                    "task_family": row.get("task_family", ""),
                    "condition": condition,
                    "expected_action": row.get("expected_action", ""),
                    "prior_action": prior_action,
                    "completion": completion,
                    "final_action": action,
                    "expected_action_match": bool(action == str(row.get("expected_action", ""))),
                    **metadata,
                }
            )
    traces = pd.DataFrame(records)
    write_parquet(traces, output_dir / "traces.parquet")
    traces.to_csv(output_dir / "traces.csv", index=False)
    summary = _entropy_summary(traces)
    write_parquet(summary, output_dir / "condition_summary.parquet")
    summary.to_csv(output_dir / "condition_summary.csv", index=False)
    gate = _matching_gate(summary, traces)
    gate.update({"observed_rows": int(len(traces)), "planned_generations": planned})
    write_json(output_dir / "gate_status.json", gate)
    lines = [
        "# Local Open-Weight Strict Matching Audit",
        "",
        f"- Gate: `{gate['status']}`",
        f"- Model: `{model_id}`",
        f"- Planned/observed generations: `{planned}` / `{len(traces)}`",
        f"- Gate rule: {gate['gate_rule']}",
        f"- Entropy gap: `{gate['canonical_entropy_gap_bits']:.3f}` bits",
        f"- Prompt match rate: `{gate['prompt_match_rate']:.3f}`",
        f"- Completion match rate: `{gate['completion_match_rate']:.3f}`",
        f"- Total-token ratio: `{gate['mean_total_token_ratio_control_over_structured']:.3f}`",
        f"- Latency ratio: `{gate['mean_latency_ratio_control_over_structured']:.3f}`",
        "",
        "This forced-budget audit uses local open-weight inference with exact prompt-token padding and fixed generation length. "
        "It is a CPU-local audit in the current workspace; CUDA/GPU-memory logging is unavailable because CUDA is not available.",
        "",
    ]
    (output_dir / "strict_matching_local_report.md").write_text("\n".join(lines), encoding="utf-8")
    return {"traces": traces, "summary": summary, "gate_status": gate}

