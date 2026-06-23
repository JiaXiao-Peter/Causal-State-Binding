from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from freewill.config import load_config
from freewill.structured_control_validation import (
    FINITE_ACTIONS,
    CSB_ARCHITECTURES,
    CSB_FORMAL_API_VARIANTS,
    CSB_MODEL_CALL_VARIANTS,
    CSB_TASK_FAMILIES,
    _default_payload,
    _decoding_for_variant,
    _offline_distribution_policy,
    _prompt,
    _score_traces,
    _scrambled_context_policy,
    apply_CSB_robustness_transform,
    build_CSB_pilot_events,
    filter_CSB_event_indices,
)
from freewill.providers import ProviderClient
from freewill.schema_equalized_agents import _normalize_payload
from freewill.utils import ensure_dir, write_json, write_parquet


DEFAULT_MODELS = [
    "gpt-5.4-mini",
    "gemini-3.1-flash-lite-preview",
    "deepseek-v3.2",
    "qwen3-max",
]


def _read_api_settings(path: Path | None) -> tuple[str, str]:
    if path is not None and path.exists() and path.is_file():
        text = path.read_text(encoding="utf-8")
        key_match = re.search(r"api_key\s*=\s*[\"']([^\"']+)[\"']", text)
        base_match = re.search(r"base_url\s*=\s*[\"']([^\"']+)[\"']", text)
        if key_match:
            base_url = base_match.group(1).strip() if base_match else "https://api.shubiaobiao.cn/v1/"
            return key_match.group(1).strip(), base_url
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if api_key:
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.shubiaobiao.cn/v1/").strip()
        return api_key, base_url
    raise RuntimeError("API key was not found. Set OPENAI_API_KEY or pass --api-file pointing to a local credentials file.")


def _safe_slug(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in str(value)).strip("_")
    return cleaned or "model"


def _prepare_row(
    row: dict[str, Any],
    *,
    architecture: str,
    seed: int,
    variant: str,
    scrambled_contexts: dict[str, dict[str, str]],
    offline_distribution_actions: dict[tuple[str, int, str], str],
    offline_policy_name: str,
) -> dict[str, Any]:
    row_for_variant = dict(row)
    row_for_variant["context_policy"] = "current_decisive_context"
    row_for_variant["text"] = row_for_variant.get("decisive_context", row_for_variant.get("text", ""))
    if variant == "stochastic_no_fields":
        row_for_variant["context_policy"] = "surface_only_no_intervention_fields"
        row_for_variant["text"] = row_for_variant.get("surface_context", "")
    elif variant == "stochastic_context_scrambled":
        scrambled = scrambled_contexts.get(str(row_for_variant.get("event_id", "")), {})
        row_for_variant["context_policy"] = "scrambled_decisive_context_from_different_expected_action"
        row_for_variant["scrambled_context_event_id"] = scrambled.get("scrambled_event_id", "")
        row_for_variant["scrambled_expected_action"] = scrambled.get("scrambled_expected_action", "")
        row_for_variant["text"] = (
            f"{row_for_variant.get('surface_context', '')} "
            f"Control context: {scrambled.get('scrambled_context', '')}"
        ).strip()
    elif variant == "target_lesion_strict":
        row_for_variant["context_policy"] = "strict_target_component_removed_no_decisive_context"
        row_for_variant["text"] = (
            f"{row_for_variant.get('surface_context', '')} "
            f"Strict lesion notice: the {row_for_variant.get('probe_component', 'target')} component "
            "and the decisive intervention context are unavailable for this event."
        ).strip()
    if variant == "distribution_matched":
        offline_action = offline_distribution_actions[(architecture, int(seed), str(row_for_variant.get("event_id", "")))]
        row_for_variant["distribution_matched_action"] = offline_action
        row_for_variant["distribution_matched_policy"] = offline_policy_name
        row_for_variant["context_policy"] = "offline_global_marginal_no_context"
        row_for_variant["text"] = ""
    return row_for_variant


def _metadata_counts(traces: pd.DataFrame) -> dict[str, int]:
    if traces.empty:
        return {
            "remote_call_count": 0,
            "fallback_used_count": 0,
            "parse_failure_count": 0,
            "unrecovered_failure_count": 0,
        }
    remote = traces.get("remote_used", pd.Series(False, index=traces.index)).fillna(False).astype(bool)
    fallback = traces.get("fallback_used", pd.Series(False, index=traces.index)).fillna(False).astype(bool)
    errors = traces.get("error_type", pd.Series("", index=traces.index)).fillna("").astype(str)
    variants = traces.get("variant", pd.Series("", index=traces.index)).fillna("").astype(str)
    model_call_rows = variants.isin(CSB_MODEL_CALL_VARIANTS)
    parse_errors = errors.isin({"parse_error", "invalid_action_code"})
    transport_errors = (
        errors.str.startswith("http_")
        | errors.isin({"request_error", "missing_api_key", "offline_default"})
        | (model_call_rows & ~remote)
    )
    return {
        "remote_call_count": int(remote.sum()),
        "fallback_used_count": int(fallback.sum()),
        "parse_failure_count": int(parse_errors.sum()),
        "unrecovered_failure_count": int(transport_errors.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run formal fallback-free API validation over 20 task families.")
    parser.add_argument("--config", default="configs/core_study.yaml")
    parser.add_argument("--paths", default="configs/local_windows_paths.yaml")
    parser.add_argument("--api-file", default="", help="Optional local credentials file. Public releases should prefer OPENAI_API_KEY.")
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--hard-call-cap", type=int, default=12000)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", default="results/paper1_revision/remote_5090_run3_20260508/formal_api_validation")
    args = parser.parse_args()

    api_key, base_url = _read_api_settings(Path(args.api_file) if args.api_file else None)
    os.environ["OPENAI_API_KEY"] = api_key
    config = load_config(args.config, args.paths)
    run_id = args.run_id or f"formal_api_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = ensure_dir(Path(args.output_dir) / run_id)

    events = build_CSB_pilot_events(events_per_family=36, task_families=CSB_TASK_FAMILIES)
    events = filter_CSB_event_indices(events, start=12, end=24)
    events = apply_CSB_robustness_transform(events, "original")
    seeds = [1]
    variants = list(CSB_FORMAL_API_VARIANTS)
    planned_rows = len(args.models) * len(CSB_ARCHITECTURES) * len(events) * len(variants) * len(seeds)
    planned_remote_calls = len(args.models) * len(CSB_ARCHITECTURES) * len(events) * len([item for item in variants if item in CSB_MODEL_CALL_VARIANTS]) * len(seeds)
    manifest = {
        "run_id": run_id,
        "models": args.models,
        "base_url": base_url,
        "events_per_family": 36,
        "event_index_start": 12,
        "event_index_end": 24,
        "task_family_count": int(events["task_family"].nunique()),
        "event_count": int(len(events)),
        "architectures": CSB_ARCHITECTURES,
        "variants": variants,
        "seeds": seeds,
        "planned_rows": int(planned_rows),
        "planned_remote_calls": int(planned_remote_calls),
        "fallback_disabled": True,
        "dry_run": bool(args.dry_run),
    }
    write_json(output_dir / "run_manifest.json", manifest)
    if planned_remote_calls > int(args.hard_call_cap):
        raise RuntimeError(f"planned remote calls {planned_remote_calls} exceed hard cap {args.hard_call_cap}")
    if args.dry_run:
        write_json(output_dir / "gate_status.json", {"status": "planned", **manifest})
        print(json.dumps({"status": "planned", **manifest}, ensure_ascii=False))
        return

    all_traces: list[dict[str, Any]] = []
    all_behaviors: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    for model in args.models:
        model_config = deepcopy(config)
        provider_cfg = model_config.agent.provider
        provider_cfg.mode = "openai_compatible"
        provider_cfg.model = model
        provider_cfg.base_url = base_url
        provider_cfg.endpoint_path = "/chat/completions"
        provider_cfg.api_key_env = "OPENAI_API_KEY"
        provider_cfg.json_mode = "json_object"
        provider_cfg.fallback_model = ""
        provider_cfg.fallback_base_url = ""
        provider_cfg.fallback_endpoint_path = ""
        provider_cfg.fallback_api_key_env = ""
        provider_cfg.fallback_json_mode = ""
        provider = ProviderClient(provider_cfg)
        scrambled_contexts = _scrambled_context_policy(events)
        offline_actions, offline_policy = _offline_distribution_policy(events, CSB_ARCHITECTURES, seeds)
        tasks: list[dict[str, Any]] = []
        for architecture in CSB_ARCHITECTURES:
            for seed in seeds:
                for variant in variants:
                    for index, row in enumerate(events.reset_index(drop=True).to_dict(orient="records")):
                        row_for_variant = _prepare_row(
                            row,
                            architecture=str(architecture),
                            seed=int(seed),
                            variant=str(variant),
                            scrambled_contexts=scrambled_contexts,
                            offline_distribution_actions=offline_actions,
                            offline_policy_name=str(offline_policy["name"]),
                        )
                        tasks.append(
                            {
                                "model": model,
                                "architecture": str(architecture),
                                "seed": int(seed),
                                "variant": str(variant),
                                "index": int(index),
                                "row": row_for_variant,
                            }
                        )

        def run_task(task: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
            row = task["row"]
            variant = task["variant"]
            architecture = task["architecture"]
            seed = int(task["seed"])
            default = _default_payload(row, architecture, variant, seed)
            temperature, top_p = _decoding_for_variant(variant, architecture, None)
            raw_generation = ""
            if variant == "distribution_matched":
                payload = default
                metadata = {
                    "model": model,
                    "remote_used": False,
                    "fallback_used": False,
                    "offline_policy_used": True,
                    "offline_policy_name": offline_policy["name"],
                    "error_type": "",
                    "request_temperature": temperature,
                    "request_top_p": top_p,
                    "latency_ms": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                }
            else:
                start = time.perf_counter()
                result = provider.chat_json_with_metadata(
                    "Return exactly one compact JSON object and nothing else.",
                    _prompt(row, architecture, variant),
                    default,
                    temperature_override=temperature,
                    top_p_override=top_p,
                )
                payload = _normalize_payload(result.payload if isinstance(result.payload, dict) else {}, default)
                metadata = dict(result.metadata)
                metadata["latency_ms"] = int(metadata.get("latency_ms") or ((time.perf_counter() - start) * 1000))
                metadata["offline_policy_used"] = False
                metadata["request_temperature"] = temperature
                metadata["request_top_p"] = top_p
            if str(payload.get("final_action", "")) not in row.get("candidate_actions", FINITE_ACTIONS):
                payload["final_action"] = default["final_action"]
                metadata["error_type"] = metadata.get("error_type") or "invalid_action_code"
            family = str(row.get("task_family", ""))
            trace = {
                "dataset_id": row.get("dataset_id", ""),
                "output_namespace": f"{architecture}__{family}",
                "architecture": architecture,
                "task_family": family,
                "probe_component": row.get("probe_component", ""),
                "variant": variant,
                "episode_id": row.get("episode_id", ""),
                "step_id": f"{architecture}:{family}:{variant}:{seed}:{int(row.get('global_event_index', task['index'])) + 1:05d}",
                "event_id": row.get("event_id", ""),
                "base_event_id": row.get("base_event_id", row.get("event_id", "")),
                "family_event_index": row.get("family_event_index", task["index"]),
                "seed": seed,
                "expected_action": row.get("expected_action", ""),
                "lesion_expected_action": row.get("lesion_expected_action", ""),
                "distribution_matched_action": row.get("distribution_matched_action", ""),
                "context_policy": row.get("context_policy", ""),
                "robustness_block": row.get("robustness_block", "original"),
                "raw_generation": raw_generation,
                "prompt_tokens": int(metadata.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(metadata.get("completion_tokens", 0) or 0),
                "total_tokens": int(metadata.get("total_tokens", 0) or 0),
                "latency_ms": int(metadata.get("latency_ms", 0) or 0),
                "remote_used": bool(metadata.get("remote_used", False)),
                "fallback_used": bool(metadata.get("fallback_used", False)),
                "error_type": metadata.get("error_type", ""),
                **payload,
                "provider_metadata": metadata,
            }
            behavior = {
                "dataset_id": row.get("dataset_id", ""),
                "output_namespace": f"{architecture}__{family}",
                "architecture": architecture,
                "task_family": family,
                "probe_component": row.get("probe_component", ""),
                "variant": variant,
                "episode_id": row.get("episode_id", ""),
                "subject_id": variant,
                "first_impulse": payload["first_impulse"],
                "final_action": payload["final_action"],
                "expected_action": row.get("expected_action", ""),
                "context_policy": row.get("context_policy", ""),
                "seed": seed,
            }
            return trace, behavior

        model_traces: list[dict[str, Any]] = []
        model_behaviors: list[dict[str, Any]] = []
        model_started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=max(1, int(args.max_workers))) as pool:
            futures = [pool.submit(run_task, task) for task in tasks]
            for completed, future in enumerate(as_completed(futures), start=1):
                trace, behavior = future.result()
                model_traces.append(trace)
                model_behaviors.append(behavior)
                progress_every = max(1, int(args.progress_every))
                if completed % progress_every == 0 or completed == len(tasks):
                    elapsed = max(0.001, time.perf_counter() - model_started)
                    rate = completed / elapsed
                    remaining = max(0, len(tasks) - completed)
                    eta_seconds = int(remaining / max(rate, 0.001))
                    print(
                        json.dumps(
                            {
                                "model": model,
                                "completed": completed,
                                "total": len(tasks),
                                "rate_rows_per_sec": round(rate, 3),
                                "eta_seconds": eta_seconds,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
        all_traces.extend(model_traces)
        all_behaviors.extend(model_behaviors)
        model_frame = pd.DataFrame(model_traces)
        counts = _metadata_counts(model_frame)
        model_rows.append({"model": model, "rows": len(model_frame), **counts})
        model_dir = ensure_dir(output_dir / _safe_slug(model))
        write_parquet(model_frame, model_dir / "traces.parquet")
        write_parquet(pd.DataFrame(model_behaviors), model_dir / "behaviors.parquet")

    traces = pd.DataFrame(all_traces)
    behaviors = pd.DataFrame(all_behaviors)
    scores, summary, component_summary = _score_traces(traces)
    counts = _metadata_counts(traces)
    gate = {
        "status": "pass"
        if (
            len(traces) == planned_rows
            and counts["remote_call_count"] == planned_remote_calls
            and counts["fallback_used_count"] == 0
            and counts["unrecovered_failure_count"] == 0
            and counts["parse_failure_count"] / max(1, len(traces)) < 0.05
        )
        else "needs_review",
        **manifest,
        **counts,
        "row_count": int(len(traces)),
    }
    write_parquet(traces, output_dir / "traces.parquet")
    write_parquet(behaviors, output_dir / "behaviors.parquet")
    write_parquet(scores, output_dir / "finite_action_scores.parquet")
    write_parquet(summary, output_dir / "finite_action_summary.parquet")
    write_parquet(component_summary, output_dir / "component_summary.parquet")
    pd.DataFrame(model_rows).to_csv(output_dir / "model_gate_summary.csv", index=False, encoding="utf-8")
    write_json(output_dir / "gate_status.json", gate)
    print(json.dumps(gate, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

