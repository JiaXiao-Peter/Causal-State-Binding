from __future__ import annotations

import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from freewill.config import ProjectConfig, load_config
from freewill.providers import ProviderClient
from freewill.structured_control_validation import (
    FINITE_ACTIONS,
    FAMILY_COMPONENTS,
    CSB_TASK_FAMILIES,
    _offline_distribution_policy,
    _scrambled_context_policy,
    build_CSB_pilot_events,
    filter_CSB_event_indices,
)
from freewill.utils import ensure_dir, write_json, write_parquet


CAUSAL_STATE_CONDITIONS = [
    "full_state",
    "only_decisive_causal_field",
    "surface_context_only",
    "action_prior_only",
    "scrambled_decisive_field",
    "irrelevant_cue_full_state",
]

SUFFICIENCY_CONDITIONS = [
    "full_state",
    "only_decisive_causal_field",
    "surface_context_only",
    "action_prior_only",
    "scrambled_decisive_field",
]


@dataclass(frozen=True)
class ApiSettings:
    api_key: str
    base_url: str


MODEL_TOKEN_PATTERN = re.compile(
    r"\b(?:gpt|qwen|deepseek|gemini|claude|moonshot|glm|yi|llama|mistral|doubao|ernie)[A-Za-z0-9._:/+-]*\b",
    flags=re.IGNORECASE,
)

DEFAULT_API_MODEL_PREFERENCES = [
    "gpt-3.5-turbo",
    "gpt-4o-mini",
    "qwen-turbo",
    "qwen-plus",
    "qwen-plus-latest",
    "deepseek-chat",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gpt-4.1-nano",
    "qwen3-30b-a3b",
    "deepseek-v3",
    "claude-haiku-4-5-20251001",
]


def read_api_settings(path: Path | None = None) -> ApiSettings:
    if path is not None and path.exists() and path.is_file():
        text = path.read_text(encoding="utf-8-sig", errors="ignore")
        key_match = re.search(r"api_key\s*=\s*[\"']?([^\s\"')]+)[\"']?", text)
        base_match = re.search(r"base_url\s*=\s*[\"']?([^\s\"')]+)[\"']?", text)
        if key_match:
            api_key = key_match.group(1).strip().rstrip(",")
            base_url = base_match.group(1).strip().rstrip(",") if base_match else "https://api.shubiaobiao.cn/v1/"
            return ApiSettings(
                api_key=api_key,
                base_url=base_url,
            )
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if api_key:
        return ApiSettings(api_key=api_key, base_url=os.environ.get("OPENAI_BASE_URL", "https://api.shubiaobiao.cn/v1/").strip())
    raise RuntimeError("API key was not found. Set OPENAI_API_KEY or pass --api-file.")


def extract_api_file_models(path: Path | None) -> list[str]:
    if path is None or not path.exists() or not path.is_file():
        return []
    text = path.read_text(encoding="utf-8-sig", errors="ignore")
    text = re.sub(r"(?i)(api[_-]?key\s*[=:]\s*)\S+", r"\1<redacted>", text)
    text = re.sub(r"sk-[A-Za-z0-9_-]+", "<redacted-key>", text)
    models: list[str] = []
    for match in MODEL_TOKEN_PATTERN.finditer(text):
        token = match.group(0).strip(" ,;，；。)）]】\"'")
        if not token or token.lower() in {"api", "base_url"}:
            continue
        if token not in models:
            models.append(token)
    return models


def select_api_file_models(
    path: Path | None,
    *,
    count: int = 6,
    preferred: Iterable[str] = DEFAULT_API_MODEL_PREFERENCES,
) -> list[str]:
    available = extract_api_file_models(path)
    if not available:
        return []
    selected: list[str] = []
    available_lower = {model.lower(): model for model in available}
    for model in preferred:
        if model.lower() in available_lower and available_lower[model.lower()] not in selected:
            selected.append(available_lower[model.lower()])
        if len(selected) >= count:
            return selected
    for model in available:
        if model not in selected:
            selected.append(model)
        if len(selected) >= count:
            break
    return selected


def safe_slug(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in str(value)).strip("_")
    return cleaned or "run"


def decisive_field_text(row: dict[str, Any], action: str | None = None) -> str:
    expected = str(action or row.get("expected_action", "")).strip()
    component = str(row.get("probe_component", FAMILY_COMPONENTS.get(str(row.get("task_family", "")), "reason")))
    event_id = str(row.get("event_id", ""))
    if component == "memory":
        return (
            f"Decisive memory field for event {event_id}: the corrected prior commitment is {expected}. "
            "Stale, injected, or unrelated memory is non-decisive."
        )
    if component == "veto":
        if expected == "ACT_VETO":
            return (
                f"Decisive constraint/veto field for event {event_id}: a hard constraint invalidates ordinary "
                "actions and requires ACT_VETO."
            )
        return f"Decisive constraint field for event {event_id}: no veto applies; the valid controlled action is {expected}."
    if component == "self":
        return f"Decisive persistent-state field for event {event_id}: the stable self-policy selects {expected}."
    return f"Decisive reason field for event {event_id}: calibrated task evidence selects {expected}."


def _condition_context(
    row: dict[str, Any],
    *,
    condition: str,
    action_prior: str,
    scrambled: dict[str, str],
) -> tuple[str, str, str, str]:
    expected = str(row.get("expected_action", "")).strip()
    if condition == "full_state":
        return str(row.get("decisive_context", row.get("text", ""))), "current_event", expected, "full_decisive_context"
    if condition == "only_decisive_causal_field":
        return decisive_field_text(row, expected), "current_event", expected, "minimal_decisive_field_only"
    if condition == "surface_context_only":
        return str(row.get("surface_context", "")), "none", "", "surface_context_no_decisive_state"
    if condition == "action_prior_only":
        return (
            f"Action-prior-only control for event {row.get('event_id')}: a calibration marginal sampled {action_prior}. "
            "No event-specific reason, memory, self-state, or veto field is available.",
            "none",
            str(action_prior),
            "calibrated_action_prior_only",
        )
    if condition == "scrambled_decisive_field":
        scrambled_action = str(scrambled.get("scrambled_expected_action", "")).strip() or "ACT_A"
        scrambled_event = str(scrambled.get("scrambled_event_id", "")).strip()
        return (
            f"Scrambled decisive field control: the action query is for event {row.get('event_id')}, but the "
            f"following decisive field is from event {scrambled_event}. "
            f"{decisive_field_text({'event_id': scrambled_event, 'probe_component': row.get('probe_component')}, scrambled_action)}",
            scrambled_event,
            scrambled_action,
            "scrambled_decisive_field_from_other_event",
        )
    if condition == "irrelevant_cue_full_state":
        opposite = "ACT_A" if expected != "ACT_A" else "ACT_B"
        return (
            f"{row.get('decisive_context', row.get('text', ''))} Irrelevant cue audit: a decorative surface marker "
            f"suggests {opposite}, but that marker is not decision-relevant.",
            "current_event",
            expected,
            "full_decisive_context_with_irrelevant_cue",
        )
    raise ValueError(f"unknown condition: {condition}")


def build_prompt(row: dict[str, Any], *, condition: str, action_prior: str, scrambled: dict[str, str]) -> tuple[str, dict[str, str]]:
    context, field_event_id, visible_decisive_action, policy = _condition_context(
        row,
        condition=condition,
        action_prior=action_prior,
        scrambled=scrambled,
    )
    metadata = {
        "field_event_id": field_event_id,
        "visible_decisive_action": visible_decisive_action,
        "context_policy": policy,
    }
    prompt = (
        "Return exactly one compact JSON object and nothing else. No markdown.\n"
        "Required keys: final_action, rationale.\n"
        f"Condition: {condition}\n"
        f"Event id: {row.get('event_id')}\n"
        f"Task family: {row.get('task_family')}\n"
        f"Candidate actions: {', '.join(FINITE_ACTIONS)}\n"
        f"Context: {context}\n"
        "Select final_action using only the information provided in this condition. "
        "Do not infer hidden reason, memory, self-state, veto, or expected-action labels that are not visible. "
        "final_action must be exactly one candidate action string."
    )
    return prompt, metadata


def canonical_action(value: Any) -> str:
    text = str(value or "").strip().upper()
    for action in FINITE_ACTIONS:
        if action in text:
            return action
    return "INVALID"


def control_binding_wrapper(row: dict[str, Any]) -> dict[str, Any]:
    raw_action = canonical_action(row.get("final_action"))
    condition = str(row.get("condition", ""))
    expected = str(row.get("expected_action", "")).strip()
    field_event_id = str(row.get("field_event_id", ""))
    event_id = str(row.get("event_id", ""))
    visible_action = canonical_action(row.get("visible_decisive_action"))

    if condition in {"full_state", "only_decisive_causal_field", "irrelevant_cue_full_state"}:
        if field_event_id in {"current_event", event_id} and visible_action in FINITE_ACTIONS:
            return {
                "wrapped_action": visible_action,
                "wrapper_decision": "corrected_to_current_decisive_field" if raw_action != visible_action else "passed_current_decisive_field",
                "wrapper_changed": bool(raw_action != visible_action),
                "wrapper_expected_match": bool(visible_action == expected),
            }
    if condition == "scrambled_decisive_field":
        return {
            "wrapped_action": "ACT_DEFER",
            "wrapper_decision": "defer_scrambled_decisive_field_event_mismatch",
            "wrapper_changed": bool(raw_action != "ACT_DEFER"),
            "wrapper_expected_match": bool(expected == "ACT_DEFER"),
        }
    return {
        "wrapped_action": raw_action,
        "wrapper_decision": "no_decisive_field_no_change",
        "wrapper_changed": False,
        "wrapper_expected_match": bool(raw_action == expected),
    }


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
    provider_cfg.provider_role = "causal_state_sufficiency"
    return ProviderClient(provider_cfg)


def _metadata_subset(metadata: dict[str, Any]) -> dict[str, Any]:
    allowed = [
        "model",
        "json_mode",
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
    return {key: metadata.get(key, "") for key in allowed}


def _entropy(series: pd.Series) -> float:
    counts = series.dropna().astype(str).value_counts()
    total = float(counts.sum())
    if total <= 0:
        return float("nan")
    probs = counts / total
    return float(-(probs * np.log2(probs)).sum())


def _summarize(traces: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if traces.empty:
        return pd.DataFrame(), pd.DataFrame(), {"status": "needs_review", "reason": "empty_traces"}
    summary = (
        traces.groupby(["model", "condition"], sort=True)
        .agg(
            rows=("expected_action_match", "size"),
            accuracy=("expected_action_match", "mean"),
            wrapped_accuracy=("wrapper_expected_match", "mean"),
            wrapper_change_rate=("wrapper_changed", "mean"),
            remote_call_rate=("remote_used", "mean"),
            error_rate=("error_type", lambda s: float(s.fillna("").astype(str).ne("").mean())),
            mean_prompt_tokens=("prompt_tokens", "mean"),
            mean_completion_tokens=("completion_tokens", "mean"),
            mean_total_tokens=("total_tokens", "mean"),
            mean_latency_ms=("latency_ms", "mean"),
            action_entropy=("final_action", _entropy),
        )
        .reset_index()
    )
    cells = (
        traces.groupby(["model", "task_family", "condition"], sort=True)["expected_action_match"]
        .mean()
        .reset_index(name="accuracy")
    )
    gate_rows = []
    for model, frame in summary.groupby("model", sort=True):
        values = {str(row["condition"]): float(row["accuracy"]) for row in frame.to_dict(orient="records")}
        wrapped = {str(row["condition"]): float(row["wrapped_accuracy"]) for row in frame.to_dict(orient="records")}
        full = values.get("full_state", float("nan"))
        only = values.get("only_decisive_causal_field", float("nan"))
        control_best = max(
            values.get("surface_context_only", float("nan")),
            values.get("action_prior_only", float("nan")),
            values.get("scrambled_decisive_field", float("nan")),
        )
        recovery = float((only - control_best) / max(1e-9, full - control_best)) if math.isfinite(full) and full > control_best else float("nan")
        gate_rows.append(
            {
                "model": model,
                "full_state_accuracy": full,
                "only_decisive_accuracy": only,
                "control_best_accuracy": control_best,
                "sufficiency_recovery_fraction": recovery,
                "irrelevant_raw_accuracy": values.get("irrelevant_cue_full_state", float("nan")),
                "irrelevant_wrapped_accuracy": wrapped.get("irrelevant_cue_full_state", float("nan")),
                "wrapper_full_accuracy": wrapped.get("full_state", float("nan")),
                "wrapper_only_decisive_accuracy": wrapped.get("only_decisive_causal_field", float("nan")),
                "model_gate_pass": bool(
                    math.isfinite(full)
                    and math.isfinite(only)
                    and full >= 0.80
                    and only >= 0.80
                    and only >= control_best + 0.25
                    and recovery >= 0.75
                    and wrapped.get("irrelevant_cue_full_state", 0.0) >= values.get("irrelevant_cue_full_state", 0.0)
                ),
            }
        )
    gate_frame = pd.DataFrame(gate_rows)
    gate = {
        "status": "pass" if (not gate_frame.empty and bool(gate_frame["model_gate_pass"].all())) else "needs_review",
        "models_evaluated": int(gate_frame["model"].nunique()) if not gate_frame.empty else 0,
        "model_gate_pass_count": int(gate_frame["model_gate_pass"].sum()) if not gate_frame.empty else 0,
        "gate_rule": "full and only-decisive accuracy >=0.80; only-decisive exceeds best non-decisive control by >=0.25; recovery fraction >=0.75; wrapper does not reduce irrelevant-cue accuracy",
        "model_gate_rows": gate_rows,
    }
    return summary, cells, gate


def write_report(output_dir: Path, *, manifest: dict[str, Any], summary: pd.DataFrame, cells: pd.DataFrame, gate: dict[str, Any]) -> None:
    lines = [
        "# Causal-State Sufficiency And Wrapper Experiment",
        "",
        f"- Gate: `{gate.get('status')}`",
        f"- Models: `{', '.join(manifest.get('models', []))}`",
        f"- Planned calls: `{manifest.get('planned_remote_calls')}`",
        f"- Conditions: `{', '.join(CAUSAL_STATE_CONDITIONS)}`",
        f"- Gate rule: {gate.get('gate_rule')}",
        "",
        "## Condition Summary",
        "",
    ]
    if not summary.empty:
        lines.extend(
            [
                "| model | condition | rows | raw acc | wrapped acc | wrapper change | action entropy | prompt tok | completion tok | latency ms | errors |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in summary.to_dict(orient="records"):
            lines.append(
                f"| {row['model']} | {row['condition']} | {int(row['rows'])} | {float(row['accuracy']):.3f} | "
                f"{float(row['wrapped_accuracy']):.3f} | {float(row['wrapper_change_rate']):.3f} | "
                f"{float(row['action_entropy']):.3f} | {float(row['mean_prompt_tokens']):.1f} | "
                f"{float(row['mean_completion_tokens']):.1f} | {float(row['mean_latency_ms']):.1f} | {float(row['error_rate']):.3f} |"
            )
    lines.extend(["", "## Model Gate Rows", ""])
    model_gate_rows = gate.get("model_gate_rows", [])
    if model_gate_rows:
        lines.extend(
            [
                "| model | full | only decisive | best control | recovery | irrelevant raw | irrelevant wrapped | pass |",
                "|---|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in model_gate_rows:
            lines.append(
                f"| {row['model']} | {row['full_state_accuracy']:.3f} | {row['only_decisive_accuracy']:.3f} | "
                f"{row['control_best_accuracy']:.3f} | {row['sufficiency_recovery_fraction']:.3f} | "
                f"{row['irrelevant_raw_accuracy']:.3f} | {row['irrelevant_wrapped_accuracy']:.3f} | {row['model_gate_pass']} |"
            )
    lines.extend(
        [
            "",
            "Interpretation: the five-condition sufficiency contrast is only a finite-action measurement test. "
            "The constructive wrapper is a local, auditable pre-finalization check; it does not add a second model call "
            "and should not be interpreted as real-environment localization or task-execution reliability.",
            "",
        ]
    )
    (output_dir / "causal_state_sufficiency_report.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def run_causal_state_sufficiency(
    *,
    config: ProjectConfig,
    api_settings: ApiSettings,
    output_dir: Path,
    models: Iterable[str],
    events_per_family: int = 12,
    event_index_start: int = 0,
    event_index_end: int = 6,
    task_families: Iterable[str] | None = None,
    max_workers: int = 8,
    hard_call_cap: int = 2000,
    dry_run: bool = False,
) -> dict[str, Any]:
    os.environ["OPENAI_API_KEY"] = api_settings.api_key
    selected_models = list(models)
    events = build_CSB_pilot_events(events_per_family=events_per_family, task_families=task_families or CSB_TASK_FAMILIES)
    events = filter_CSB_event_indices(events, start=event_index_start, end=event_index_end)
    scrambled_contexts = _scrambled_context_policy(events)
    seeds = [1]
    offline_actions, offline_policy = _offline_distribution_policy(events, ["sufficiency"], seeds)
    planned = int(len(selected_models) * len(events) * len(CAUSAL_STATE_CONDITIONS))
    manifest = {
        "experiment": "causal_state_sufficiency",
        "models": selected_models,
        "base_url": api_settings.base_url,
        "events_per_family": int(events_per_family),
        "event_index_start": int(event_index_start),
        "event_index_end": int(event_index_end),
        "task_family_count": int(events["task_family"].nunique()) if not events.empty else 0,
        "event_count": int(len(events)),
        "conditions": CAUSAL_STATE_CONDITIONS,
        "planned_remote_calls": planned,
        "offline_action_prior_policy": offline_policy,
        "dry_run": bool(dry_run),
    }
    output_dir = ensure_dir(output_dir)
    write_json(output_dir / "run_manifest.json", manifest)
    if planned > hard_call_cap:
        raise RuntimeError(f"planned calls {planned} exceed hard-call cap {hard_call_cap}")
    if dry_run:
        gate = {"status": "planned", **manifest}
        write_json(output_dir / "gate_status.json", gate)
        return {"gate_status": gate}

    system_prompt = "You are an evaluation agent. Output valid JSON only."
    defaults = {"final_action": "ACT_DEFER", "rationale": "default"}
    records = events.reset_index(drop=True).to_dict(orient="records")
    providers = {model: _make_provider(config, model=model, base_url=api_settings.base_url) for model in selected_models}

    tasks: list[dict[str, Any]] = []
    for model in selected_models:
        for row in records:
            event_id = str(row.get("event_id", ""))
            prior = offline_actions[("sufficiency", 1, event_id)]
            scrambled = scrambled_contexts.get(event_id, {})
            for condition in CAUSAL_STATE_CONDITIONS:
                prompt, prompt_meta = build_prompt(row, condition=condition, action_prior=prior, scrambled=scrambled)
                tasks.append(
                    {
                        "model": model,
                        "row": row,
                        "condition": condition,
                        "event_id": event_id,
                        "prior": prior,
                        "scrambled": scrambled,
                        "prompt": prompt,
                        "prompt_meta": prompt_meta,
                    }
                )

    def run_task(task: dict[str, Any]) -> dict[str, Any]:
        row = task["row"]
        provider = providers[str(task["model"])]
        start = time.perf_counter()
        result = provider.chat_json_with_metadata(system_prompt, str(task["prompt"]), defaults)
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        payload = result.payload if isinstance(result.payload, dict) else defaults
        action = canonical_action(payload.get("final_action"))
        metadata = _metadata_subset(result.metadata)
        if not metadata.get("latency_ms"):
            metadata["latency_ms"] = elapsed_ms
        error_type = str(metadata.get("error_type", "") or "")
        if action not in FINITE_ACTIONS:
            error_type = error_type or "invalid_action_code"
        prompt_meta = task["prompt_meta"]
        scrambled = task["scrambled"]
        trace = {
            "model": task["model"],
            "task_family": row.get("task_family", ""),
            "probe_component": row.get("probe_component", ""),
            "event_id": task["event_id"],
            "family_event_index": int(row.get("family_event_index", 0)),
            "condition": task["condition"],
            "expected_action": row.get("expected_action", ""),
            "action_prior": task["prior"],
            "scrambled_event_id": scrambled.get("scrambled_event_id", ""),
            "scrambled_expected_action": scrambled.get("scrambled_expected_action", ""),
            "field_event_id": prompt_meta["field_event_id"],
            "visible_decisive_action": prompt_meta["visible_decisive_action"],
            "context_policy": prompt_meta["context_policy"],
            "final_action": action,
            "raw_final_action": payload.get("final_action", ""),
            "rationale": payload.get("rationale", ""),
            "expected_action_match": bool(action == str(row.get("expected_action", ""))),
            **metadata,
            "error_type": error_type,
        }
        trace.update(control_binding_wrapper(trace))
        return trace

    trace_rows: list[dict[str, Any]] = []
    worker_count = max(1, int(max_workers))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(run_task, task) for task in tasks]
        for future in as_completed(futures):
            trace_rows.append(future.result())
            if len(trace_rows) % 20 == 0:
                print(
                    json.dumps(
                        {
                            "event": "causal_state_sufficiency_progress",
                            "rows": len(trace_rows),
                            "planned_rows": planned,
                            "workers": worker_count,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                partial = pd.DataFrame(trace_rows)
                write_parquet(partial, output_dir / "traces.parquet")
                partial.to_csv(output_dir / "traces.csv", index=False)

    traces = pd.DataFrame(trace_rows)
    summary, cells, gate = _summarize(traces)
    gate = {**gate, **manifest, "observed_rows": int(len(traces))}
    write_parquet(traces, output_dir / "traces.parquet")
    traces.to_csv(output_dir / "traces.csv", index=False)
    write_parquet(summary, output_dir / "condition_summary.parquet")
    summary.to_csv(output_dir / "condition_summary.csv", index=False)
    write_parquet(cells, output_dir / "task_family_condition_cells.parquet")
    cells.to_csv(output_dir / "task_family_condition_cells.csv", index=False)
    write_json(output_dir / "gate_status.json", gate)
    write_report(output_dir, manifest=manifest, summary=summary, cells=cells, gate=gate)
    return {"gate_status": gate, "traces": traces, "summary": summary, "cells": cells}


def load_default_config() -> ProjectConfig:
    return load_config("configs/core_study.yaml", "configs/local_windows_paths.yaml")

