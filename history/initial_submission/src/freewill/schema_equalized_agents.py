from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from freewill.behavioral_metrics import compute_afci_records, schema_equalized_gate, summarize_afci
from freewill.config import ProjectConfig
from freewill.providers import ProviderClient
from freewill.utils import ensure_dir, read_parquet, write_json, write_parquet


SCHEMA_EQUALIZED_FIELDS = [
    "first_impulse",
    "candidate_actions",
    "reason_graph",
    "self_state",
    "memory_trace",
    "veto_state",
    "final_action",
    "final_action_rationale",
    "module_metadata",
]

SCHEMA_EQUALIZED_VARIANTS = [
    "A4_full",
    "A5_plain",
    "A5_schema_random_fields",
    "A5_schema_posthoc_fields",
    "A5_schema_scrambled_fields",
    "A4_posthoc_fields",
    "A4_scrambled_fields",
    "A4_compressed",
    "A5_long_scratchpad",
]


@dataclass(frozen=True)
class SchemaVariantSpec:
    variant: str
    action_policy: str
    field_generation: str
    reason_coupled: bool
    memory_coupled: bool
    self_coupled: bool
    veto_coupled: bool
    compressed: bool = False
    long_scratchpad: bool = False


SCHEMA_VARIANT_SPECS: dict[str, SchemaVariantSpec] = {
    "A4_full": SchemaVariantSpec("A4_full", "structured", "action_coupled", True, True, True, True),
    "A5_plain": SchemaVariantSpec("A5_plain", "stochastic", "empty", False, False, False, False),
    "A5_schema_random_fields": SchemaVariantSpec("A5_schema_random_fields", "stochastic", "random", False, False, False, False),
    "A5_schema_posthoc_fields": SchemaVariantSpec("A5_schema_posthoc_fields", "stochastic", "posthoc", False, False, False, False),
    "A5_schema_scrambled_fields": SchemaVariantSpec("A5_schema_scrambled_fields", "stochastic", "scrambled", False, False, False, False),
    "A4_posthoc_fields": SchemaVariantSpec("A4_posthoc_fields", "structured_no_fields", "posthoc", False, False, False, False),
    "A4_scrambled_fields": SchemaVariantSpec("A4_scrambled_fields", "structured", "scrambled", False, False, False, False),
    "A4_compressed": SchemaVariantSpec("A4_compressed", "structured", "action_coupled", True, True, True, True, compressed=True),
    "A5_long_scratchpad": SchemaVariantSpec("A5_long_scratchpad", "stochastic", "random", False, False, False, False, long_scratchpad=True),
}


def is_schema_equalized_variant(variant: str) -> bool:
    return str(variant) in SCHEMA_VARIANT_SPECS


def _jsonish(value: Any, default: Any = None) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _construct_score(row: dict[str, Any], dimension: str) -> float:
    text_value = _safe_float(row.get(f"{dimension}_score"))
    task_value = _safe_float(row.get(f"task_{dimension}_score"))
    return max(text_value, task_value)


def _keywords(text: str) -> list[str]:
    stop = {"the", "a", "an", "and", "or", "of", "to", "is", "was", "were", "in", "on", "for"}
    tokens = [token.strip(".,!?;:()[]{}").lower() for token in str(text).split()]
    return [token for token in tokens if token and token not in stop][:8]


def _candidate_actions(row: dict[str, Any]) -> list[str]:
    text = str(row.get("text", ""))
    lower = f"{row.get('dataset_id', '')} {row.get('granularity', '')} {text}".lower()
    keys = _keywords(text) or ["context"]
    if "recall" in lower or "memory" in lower:
        return ["retrieve_relevant_memory", "summarize_memory", "defer_recall", "random_associate"]
    if "choice" in lower or "uncertain" in lower or "risk" in lower:
        return [f"choose_{keys[0]}", f"avoid_{keys[-1]}", "defer_decision", "veto_or_withhold"]
    if "self" in lower:
        return [f"preserve_{keys[0]}", f"revise_{keys[-1]}", "defer_identity_update", "random_explore"]
    return [f"pursue_{keys[0]}", f"avoid_{keys[-1]}", "defer_action", "random_explore"]


def _structured_action(row: dict[str, Any], candidates: list[str]) -> tuple[str, str]:
    first = candidates[0]
    final = first
    rationale = "default first impulse"
    if _construct_score(row, "reason") >= max(_construct_score(row, "branch"), 0.35) and len(candidates) > 1:
        final = candidates[1]
        rationale = "decisive reason favored the alternative action"
    if _construct_score(row, "continuity") > 0.45 and len(candidates) > 1:
        final = candidates[0]
        rationale = "memory and continuity favored preserving the prior commitment"
    if _construct_score(row, "veto") > 0.25:
        final = "veto_or_withhold" if "veto_or_withhold" in candidates else candidates[min(1, len(candidates) - 1)]
        rationale = "veto cue overrode the first impulse"
    return final, rationale


def _stochastic_action(row: dict[str, Any], candidates: list[str], seed: int, variant: str) -> tuple[str, str]:
    key = (str(row.get("event_id", "")), int(seed), str(variant), str(row.get("text", ""))[:48])
    rng = np.random.default_rng(abs(hash(key)) % (2**32))
    final = str(rng.choice(candidates).item())
    return final, "stochastic branch sample"


def _field_source_row(row: dict[str, Any], scramble_row: dict[str, Any] | None) -> dict[str, Any]:
    return scramble_row if scramble_row is not None else row


def _coupling_map(spec: SchemaVariantSpec) -> dict[str, bool]:
    return {
        "reason": spec.reason_coupled,
        "memory": spec.memory_coupled,
        "self": spec.self_coupled,
        "veto": spec.veto_coupled,
    }


def _empty_fields() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    return (
        {"nodes": [], "weights": {}, "preferred_action": ""},
        {"identity_weight": 0.0, "continuity_weight": 0.0, "commitment": ""},
        {"items": [], "commitment": ""},
        {"applied": False, "condition_required": False, "strength": 0.0},
    )


def _schema_payload(row: dict[str, Any], variant: str, seed: int, scramble_row: dict[str, Any] | None = None) -> dict[str, Any]:
    spec = SCHEMA_VARIANT_SPECS[variant]
    candidates = _candidate_actions(row)
    first_impulse = candidates[0]
    if spec.action_policy.startswith("structured"):
        final_action, rationale = _structured_action(row, candidates)
    else:
        final_action, rationale = _stochastic_action(row, candidates, seed, variant)

    field_row = _field_source_row(row, scramble_row) if spec.field_generation == "scrambled" else row
    if spec.field_generation == "empty":
        reason_graph, self_state, memory_trace, veto_state = _empty_fields()
    else:
        source_candidates = _candidate_actions(field_row)
        if spec.field_generation in {"random", "scrambled"}:
            rng = np.random.default_rng(abs(hash((variant, seed, row.get("event_id", ""), "fields"))) % (2**32))
            field_action = str(rng.choice(source_candidates).item())
        elif spec.field_generation == "posthoc":
            field_action = final_action
        else:
            field_action = final_action
        reason_graph = {
            "nodes": [f"reason_supports::{field_action}"] if not spec.compressed else ["reason_supports_final"],
            "weights": {"top_reason": 1.0},
            "preferred_action": field_action,
        }
        self_state = {
            "identity_weight": round(_construct_score(field_row, "self"), 4),
            "continuity_weight": round(_construct_score(field_row, "continuity"), 4),
            "commitment": field_action if spec.self_coupled else "",
        }
        memory_trace = {
            "items": [field_action] if not spec.compressed else ["commitment"],
            "commitment": field_action if spec.memory_coupled else "",
            "source_event_id": field_row.get("event_id", ""),
        }
        veto_applied = final_action != first_impulse and ("veto" in final_action or _construct_score(row, "veto") > 0.25)
        veto_state = {
            "applied": bool(veto_applied),
            "condition_required": bool(_construct_score(row, "veto") > 0.25),
            "strength": round(_construct_score(field_row, "veto"), 4),
            "vetoed_action": first_impulse if veto_applied else "",
        }

    if spec.long_scratchpad:
        reason_graph["scratchpad_budget_note"] = "long stochastic scratchpad without action-coupled modules"
        reason_graph["nodes"] = [*reason_graph.get("nodes", []), "template_reason", "template_alternative", "template_uncertainty"]

    module_metadata = {
        "variant": variant,
        "schema_equalized": True,
        "action_policy": spec.action_policy,
        "field_generation": spec.field_generation,
        "action_field_coupling": _coupling_map(spec),
        "decision_order": "fields_then_action" if spec.field_generation == "action_coupled" else "action_then_fields",
        "compressed": spec.compressed,
        "long_scratchpad": spec.long_scratchpad,
        "irrelevant_field_suppression": spec.field_generation == "action_coupled",
        "schema_fields": SCHEMA_EQUALIZED_FIELDS,
    }
    return {
        "first_impulse": first_impulse,
        "candidate_actions": candidates,
        "reason_graph": reason_graph,
        "self_state": self_state,
        "memory_trace": memory_trace,
        "veto_state": veto_state,
        "final_action": final_action,
        "final_action_rationale": rationale if not spec.compressed else rationale[:72],
        "module_metadata": module_metadata,
    }


def _system_prompt() -> str:
    return (
        "Return one strict JSON object with exactly these top-level keys: "
        f"{', '.join(SCHEMA_EQUALIZED_FIELDS)}. Do not include prose outside JSON."
    )


def _user_prompt(row: dict[str, Any], variant: str, spec: SchemaVariantSpec, scramble_row: dict[str, Any] | None) -> str:
    scramble_text = str(scramble_row.get("text", ""))[:600] if scramble_row else ""
    return (
        f"Variant: {variant}\n"
        f"Action policy: {spec.action_policy}\n"
        f"Field generation: {spec.field_generation}\n"
        f"Coupled modules: reason={spec.reason_coupled}, memory={spec.memory_coupled}, self={spec.self_coupled}, veto={spec.veto_coupled}\n"
        f"Dataset: {row.get('dataset_id')}\n"
        f"Event id: {row.get('event_id')}\n"
        f"Context: {str(row.get('text', ''))[:1800]}\n"
        f"Scramble source context, if field_generation is scrambled: {scramble_text}\n"
        "For action-coupled variants, reason/memory/self/veto fields must constrain final_action. "
        "For stochastic/posthoc/random/scrambled variants, final_action must be sampled before fields or fields must be unrelated as specified."
    )


def _normalize_payload(payload: dict[str, Any], default: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(default)
    if isinstance(payload, dict):
        for field in SCHEMA_EQUALIZED_FIELDS:
            if field in payload:
                normalized[field] = payload[field]
    normalized["candidate_actions"] = normalized["candidate_actions"] if isinstance(normalized["candidate_actions"], list) else default["candidate_actions"]
    for field in ["reason_graph", "self_state", "memory_trace", "veto_state", "module_metadata"]:
        normalized[field] = normalized[field] if isinstance(normalized[field], dict) else default[field]
    normalized["first_impulse"] = str(normalized.get("first_impulse") or default["first_impulse"])
    normalized["final_action"] = str(normalized.get("final_action") or normalized["first_impulse"])
    normalized["final_action_rationale"] = str(normalized.get("final_action_rationale") or "")
    metadata = dict(default["module_metadata"])
    metadata.update(normalized.get("module_metadata", {}))
    metadata["schema_fields"] = SCHEMA_EQUALIZED_FIELDS
    normalized["module_metadata"] = metadata
    return normalized


def run_schema_equalized_family(events: pd.DataFrame, config: ProjectConfig, *, variants: list[str], seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    provider = ProviderClient(config.agent.provider)
    remote_ready = provider.is_remote_ready(False) or provider.is_remote_ready(True)
    trace_rows: list[dict[str, Any]] = []
    behavior_rows: list[dict[str, Any]] = []
    if events.empty:
        return pd.DataFrame(), pd.DataFrame()
    ordered = events.sort_values([column for column in ["episode_id", "subject_id", "onset", "event_id"] if column in events.columns], kind="stable").reset_index(drop=True)
    records = ordered.to_dict(orient="records")
    for variant in variants:
        if variant not in SCHEMA_VARIANT_SPECS:
            raise KeyError(f"Unknown schema-equalized variant: {variant}")
        spec = SCHEMA_VARIANT_SPECS[variant]
        for index, row in enumerate(records):
            scramble_row = records[(index + 1) % len(records)] if records else None
            default = _schema_payload(row, variant, seed, scramble_row=scramble_row)
            if remote_ready:
                result = provider.chat_json_with_metadata(
                    _system_prompt(),
                    _user_prompt(row, variant, spec, scramble_row),
                    default,
                    temperature_override=0.2 if spec.action_policy.startswith("structured") else 0.9,
                    top_p_override=1.0,
                )
                payload = _normalize_payload(result.payload if isinstance(result.payload, dict) else {}, default)
                provider_metadata = dict(result.metadata)
                if not isinstance(result.payload, dict):
                    provider_metadata["error_type"] = provider_metadata.get("error_type") or "payload_not_object"
            else:
                payload = default
                provider_metadata = {
                    "provider_role": config.agent.provider.provider_role,
                    "model": config.agent.provider.model,
                    "remote_used": False,
                    "fallback_used": False,
                    "error_type": "missing_api_key",
                    "latency_ms": 0,
                    "request_temperature": 0.2 if spec.action_policy.startswith("structured") else 0.9,
                    "request_top_p": 1.0,
                }
            trace_rows.append(
                {
                    "dataset_id": row.get("dataset_id", ""),
                    "variant": variant,
                    "episode_id": row.get("episode_id", ""),
                    "step_id": f"{row.get('episode_id', '')}:{variant}:{index + 1:03d}",
                    "event_id": row.get("event_id", ""),
                    **payload,
                    "provider_metadata": provider_metadata,
                }
            )
            behavior_rows.append(
                {
                    "dataset_id": row.get("dataset_id", ""),
                    "variant": variant,
                    "episode_id": row.get("episode_id", ""),
                    "subject_id": variant,
                    "first_impulse": payload["first_impulse"],
                    "final_action": payload["final_action"],
                    "reasons": payload.get("reason_graph", {}).get("nodes", []),
                    "choice_metadata": {
                        "event_id": row.get("event_id", ""),
                        "schema_equalized": True,
                        "module_metadata": payload["module_metadata"],
                        "provider_metadata": provider_metadata,
                    },
                }
            )
    return pd.DataFrame(trace_rows), pd.DataFrame(behavior_rows)


def _runtime_root(config: ProjectConfig) -> Path:
    return Path(config.paths.runtime_root)


def _paper1_revision_results(config: ProjectConfig, *parts: str) -> Path:
    return ensure_dir(_runtime_root(config) / "results" / "paper1_revision" / Path(*parts))


def _paper1_revision_reports(config: ProjectConfig) -> Path:
    return ensure_dir(_runtime_root(config) / "reports" / "paper1_revision")


def _paper1_revision_runtime(config: ProjectConfig, *parts: str) -> Path:
    return ensure_dir(_runtime_root(config) / "runtime" / "paper1_revision" / Path(*parts))


def _load_or_build_manifest(config: ProjectConfig, *, sample_size: int, smoke: bool, dataset_ids: Iterable[str] | None) -> pd.DataFrame:
    from freewill.ai_matrix import build_sample_manifest

    if not smoke:
        frozen = _runtime_root(config) / "artifacts" / "paper1_revision" / "frozen_sample_manifest.parquet"
        if frozen.exists():
            manifest = read_parquet(frozen)
            if dataset_ids:
                wanted = {str(item) for item in dataset_ids}
                manifest = manifest[manifest["output_namespace"].astype(str).isin(wanted) | manifest["dataset_id"].astype(str).isin(wanted)].copy()
            sampled = manifest[manifest.get("sample_status", pd.Series(dtype=str)).astype(str).eq("sampled")].copy()
            if sample_size:
                sampled = (
                    sampled.groupby("output_namespace", group_keys=False, sort=True)
                    .head(sample_size)
                    .reset_index(drop=True)
                )
            return sampled
    manifest = build_sample_manifest(config, dataset_ids=dataset_ids, sample_size=sample_size, smoke=smoke)
    return manifest[manifest.get("sample_status", pd.Series(dtype=str)).astype(str).eq("sampled")].copy()


def _run_chunk(
    config: ProjectConfig,
    chunk: pd.DataFrame,
    *,
    namespace: str,
    variant: str,
    seed: int,
    chunk_index: int,
    run_dir: Path,
    resume: bool,
) -> dict[str, Any]:
    trace_path = run_dir / f"{namespace}__{variant}__seed{seed}__chunk{chunk_index:04d}__traces.parquet"
    behavior_path = run_dir / f"{namespace}__{variant}__seed{seed}__chunk{chunk_index:04d}__behaviors.parquet"
    if resume and trace_path.exists() and behavior_path.exists():
        return {"output_namespace": namespace, "variant": variant, "seed": seed, "chunk_index": chunk_index, "status": "skipped_existing"}
    traces, behaviors = run_schema_equalized_family(chunk, config, variants=[variant], seed=seed)
    traces["seed"] = int(seed)
    traces["chunk_index"] = int(chunk_index)
    traces["output_namespace"] = namespace
    behaviors["seed"] = int(seed)
    behaviors["chunk_index"] = int(chunk_index)
    behaviors["output_namespace"] = namespace
    write_parquet(traces, trace_path)
    write_parquet(behaviors, behavior_path)
    return {
        "output_namespace": namespace,
        "variant": variant,
        "seed": seed,
        "chunk_index": chunk_index,
        "status": "completed",
        "trace_rows": int(len(traces)),
        "behavior_rows": int(len(behaviors)),
    }


def _collect_run_frames(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    traces = [read_parquet(path) for path in sorted(run_dir.glob("*__traces.parquet"))]
    behaviors = [read_parquet(path) for path in sorted(run_dir.glob("*__behaviors.parquet"))]
    return (
        pd.concat(traces, ignore_index=True) if traces else pd.DataFrame(),
        pd.concat(behaviors, ignore_index=True) if behaviors else pd.DataFrame(),
    )


def _run_counts(traces: pd.DataFrame) -> dict[str, Any]:
    if traces.empty or "provider_metadata" not in traces.columns:
        return {"call_count": 0, "remote_call_count": 0, "failure_count": 0, "failure_rate": 0.0}
    rows = []
    for value in traces["provider_metadata"].tolist():
        payload = _jsonish(value, {})
        rows.append(payload if isinstance(payload, dict) else {"error_type": "unreadable_metadata", "remote_used": False})
    meta = pd.DataFrame(rows)
    call_count = int(len(meta))
    failure = int(meta.get("error_type", pd.Series("", index=meta.index)).fillna("").astype(str).ne("").sum()) if call_count else 0
    remote = int(meta.get("remote_used", pd.Series(False, index=meta.index)).fillna(False).astype(bool).sum()) if call_count else 0
    return {
        "call_count": call_count,
        "remote_call_count": remote,
        "failure_count": failure,
        "failure_rate": float(failure / call_count) if call_count else 0.0,
    }


def run_schema_equalized_controls(
    config: ProjectConfig,
    *,
    dataset_ids: Iterable[str] | None = None,
    smoke: bool = False,
    sample_size: int | None = None,
    seeds: Iterable[int] | None = None,
    variants: list[str] | None = None,
    hard_call_cap: int | None = None,
    max_workers: int | None = None,
    resume: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    profile = "smoke" if smoke else "full"
    selected_variants = variants or (
        ["A4_full", "A5_plain", "A5_schema_random_fields", "A5_schema_posthoc_fields", "A5_schema_scrambled_fields"]
        if smoke
        else ["A4_full", "A5_plain", "A5_schema_random_fields", "A5_schema_posthoc_fields", "A5_schema_scrambled_fields", "A4_posthoc_fields", "A4_scrambled_fields", "A4_compressed", "A5_long_scratchpad"]
    )
    selected_seeds = [int(seed) for seed in (seeds or ([1] if smoke else [1, 2, 3]))]
    chosen_sample_size = int(sample_size or (20 if smoke else 150))
    cap = int(hard_call_cap or (1000 if smoke else 30000))
    selected_dataset_ids = list(dataset_ids) if dataset_ids else list(config.ai_matrix.datasets)
    sampled = _load_or_build_manifest(config, sample_size=chosen_sample_size, smoke=smoke, dataset_ids=selected_dataset_ids)
    planned_calls = int(len(sampled) * len(selected_variants) * len(selected_seeds))
    result_dir = _paper1_revision_results(config, "schema_equalized_controls", profile)
    plan = {
        "experiment": "schema_equalized_controls",
        "profile": profile,
        "planned_calls": planned_calls,
        "sampled_events": int(len(sampled)),
        "variants": selected_variants,
        "seeds": selected_seeds,
        "sample_size_per_dataset": chosen_sample_size,
        "hard_call_cap": cap,
        "dry_run": bool(dry_run),
    }
    write_json(result_dir / "run_plan.json", plan)
    if planned_calls > cap:
        raise RuntimeError(f"schema_equalized_controls planned calls {planned_calls} exceed hard cap {cap}")
    if dry_run:
        gate = {"status": "planned", **plan}
        write_json(result_dir / "gate_status.json", gate)
        return {"gate_status": gate, "planned_calls": planned_calls}

    run_dir = _paper1_revision_runtime(config, "schema_equalized_controls", profile, "runs")
    workers = int(max_workers or (4 if smoke else 8))
    chunk_size = int(config.ai_matrix.smoke_chunk_size if smoke else config.ai_matrix.chunk_size)
    tasks = []
    status_rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for namespace, frame in sampled.groupby("output_namespace", sort=True):
            frame = frame.reset_index(drop=True)
            for seed in selected_seeds:
                for variant in selected_variants:
                    for chunk_index, start in enumerate(range(0, len(frame), max(1, chunk_size))):
                        chunk = frame.iloc[start : start + max(1, chunk_size)].copy()
                        tasks.append(
                            executor.submit(
                                _run_chunk,
                                config,
                                chunk,
                                namespace=str(namespace),
                                variant=str(variant),
                                seed=int(seed),
                                chunk_index=int(chunk_index),
                                run_dir=run_dir,
                                resume=resume,
                            )
                        )
        for task in as_completed(tasks):
            status_rows.append(task.result())
    write_parquet(pd.DataFrame(status_rows), result_dir / "run_status.parquet")

    traces, behaviors = _collect_run_frames(run_dir)
    afci_records = compute_afci_records(traces)
    afci_summary = summarize_afci(afci_records)
    gate = schema_equalized_gate(afci_summary, bootstrap_samples=int(config.ai_matrix.bootstrap_samples))
    counts = _run_counts(traces)
    gate.update({"experiment": "schema_equalized_controls", "profile": profile, **counts, "planned_calls": planned_calls, "model": config.agent.provider.model})
    write_parquet(traces, result_dir / "traces.parquet")
    write_parquet(behaviors, result_dir / "behaviors.parquet")
    write_parquet(afci_records, result_dir / "schema_equalized_metrics.parquet")
    write_parquet(afci_summary, result_dir / "schema_equalized_summary.parquet")
    write_json(result_dir / "gate_status.json", gate)
    _write_schema_report(config, gate, afci_summary)
    return {"gate_status": gate, "afci_records": afci_records, "afci_summary": afci_summary}


def _write_schema_report(config: ProjectConfig, gate: dict[str, Any], summary: pd.DataFrame) -> Path:
    path = _paper1_revision_reports(config) / "schema_equalized_controls.md"
    lines = [
        "# Paper1 Revision Schema-Equalized Controls",
        "",
        f"- Gate: `{gate.get('status')}`",
        f"- Calls: `{gate.get('call_count')}`",
        f"- Failures: `{gate.get('failure_count')}`",
        f"- Min A4_full AFCI > control dataset count: `{gate.get('min_a4_afci_gt_control_dataset_count')}`",
        f"- Rule: {gate.get('pass_rule')}",
        "",
    ]
    if not summary.empty:
        lines.extend(["| dataset | variant | seed | AFCI | reason | memory | veto | self |", "|---|---|---:|---:|---:|---:|---:|---:|"])
        for row in summary.head(160).to_dict(orient="records"):
            lines.append(
                f"| {row.get('dataset_namespace')} | {row.get('variant')} | {int(row.get('seed', 0))} | "
                f"{float(row.get('AFCI', 0.0)):.3f} | {float(row.get('reason_action_alignment', 0.0)):.3f} | "
                f"{float(row.get('memory_action_alignment', 0.0)):.3f} | {float(row.get('veto_action_alignment', 0.0)):.3f} | "
                f"{float(row.get('self_action_alignment', 0.0)):.3f} |"
            )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path

