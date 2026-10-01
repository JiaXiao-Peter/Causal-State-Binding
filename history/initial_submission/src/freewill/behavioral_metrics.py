from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from freewill.action_canonicalization import add_action_canonical_columns, canonical_action_rule


BEHAVIORAL_PROBE_TYPES = {
    "reason_flip",
    "reason_strength_gradient",
    "memory_conflict",
    "memory_irrelevant",
    "veto_cue",
    "late_veto_cue",
    "self_continuity_probe",
    "irrelevant_cue",
    "placebo_cue",
    "adversarial_randomness_cue",
}

TARGET_PROBE_TO_METRIC = {
    "reason_flip": "B_RSI",
    "reason_strength_gradient": "B_RSI",
    "memory_conflict": "B_MCI",
    "self_continuity_probe": "B_SCI",
    "veto_cue": "B_VEI",
    "late_veto_cue": "B_VEI",
}


def jsonish(value: Any, default: Any = None) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return default


def as_dict(value: Any) -> dict[str, Any]:
    payload = jsonish(value, {})
    return payload if isinstance(payload, dict) else {}


def as_list(value: Any) -> list[Any]:
    payload = jsonish(value, [])
    return payload if isinstance(payload, list) else []


def safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "applied", "triggered"}


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def choice_event_id(row: pd.Series) -> str:
    if "event_id" in row and str(row.get("event_id", "")):
        return str(row.get("event_id", ""))
    metadata = as_dict(row.get("choice_metadata"))
    return str(metadata.get("event_id", ""))


def probe_type_from_namespace(namespace: str) -> str:
    value = str(namespace)
    for probe in sorted(BEHAVIORAL_PROBE_TYPES, key=len, reverse=True):
        suffix = f"__{probe}"
        if value.endswith(suffix):
            return probe
    return ""


def base_namespace(namespace: str) -> str:
    probe = probe_type_from_namespace(namespace)
    return str(namespace)[: -(len(probe) + 2)] if probe else str(namespace)


def base_event_id(event_id: str) -> str:
    return str(event_id).split("::", 1)[0]


def _canonical_equals(left: Any, right: Any) -> bool:
    return canonical_action_rule(left) == canonical_action_rule(right)


def _field_source_penalty(module_metadata: dict[str, Any], component: str) -> float:
    coupling = as_dict(module_metadata.get("action_field_coupling"))
    value = coupling.get(component, module_metadata.get(f"{component}_coupled"))
    if value is True or str(value).strip().lower() in {"action_coupled", "coupled", "true"}:
        return 1.0
    if str(value).strip().lower() in {"posthoc", "scrambled", "random", "template", "uncoupled"}:
        return 0.0
    field_generation = str(module_metadata.get("field_generation", "")).lower()
    if field_generation in {"posthoc", "scrambled", "random", "template"}:
        return 0.0
    return 0.5


def _reason_alignment(row: pd.Series, final_action: str, module_metadata: dict[str, Any]) -> float:
    reason_graph = as_dict(row.get("reason_graph"))
    preferred = reason_graph.get("preferred_action") or reason_graph.get("supports_final_action")
    nodes = as_list(reason_graph.get("nodes"))
    score = 0.0
    if preferred and _canonical_equals(preferred, final_action):
        score = 1.0
    elif any(canonical_action_rule(item) == canonical_action_rule(final_action) for item in nodes):
        score = 0.75
    elif nodes:
        node_text = " ".join(str(item) for item in nodes).lower()
        final_terms = canonical_action_rule(final_action).replace("_", " ").split()
        score = 0.5 if any(term in node_text for term in final_terms) else 0.25
    return score * _field_source_penalty(module_metadata, "reason")


def _memory_alignment(row: pd.Series, final_action: str, module_metadata: dict[str, Any]) -> float:
    memory_trace = as_dict(row.get("memory_trace"))
    if not memory_trace:
        memory_trace = as_dict(row.get("memory_write"))
    commitment = memory_trace.get("commitment") or memory_trace.get("relevant_memory_action")
    items = as_list(memory_trace.get("items")) or as_list(memory_trace.get("retrieved_context"))
    score = 0.0
    if commitment and _canonical_equals(commitment, final_action):
        score = 1.0
    elif any(_canonical_equals(item, final_action) for item in items):
        score = 0.75
    elif memory_trace:
        score = 0.5
    return score * _field_source_penalty(module_metadata, "memory")


def _veto_alignment(row: pd.Series, final_action: str, first_impulse: str, module_metadata: dict[str, Any]) -> float:
    veto_state = as_dict(row.get("veto_state"))
    applied = safe_bool(veto_state.get("applied", veto_state.get("triggered", False)))
    required = safe_bool(veto_state.get("condition_required", veto_state.get("veto_condition", applied)))
    changed = canonical_action_rule(final_action) != canonical_action_rule(first_impulse)
    if required and applied and changed:
        score = 1.0
    elif not required and not applied:
        score = 1.0
    elif applied and changed:
        score = 0.75
    else:
        score = 0.0
    return score * _field_source_penalty(module_metadata, "veto")


def _self_alignment(row: pd.Series, final_action: str, module_metadata: dict[str, Any]) -> float:
    self_state = as_dict(row.get("self_state"))
    commitment = self_state.get("commitment") or self_state.get("preferred_action") or self_state.get("long_term_goal_action")
    if commitment:
        score = 1.0 if _canonical_equals(commitment, final_action) else 0.0
    elif self_state:
        identity_weight = safe_float(self_state.get("identity_weight")) + safe_float(self_state.get("continuity_weight"))
        score = min(1.0, max(0.0, identity_weight))
    else:
        score = 0.0
    return score * _field_source_penalty(module_metadata, "self")


def _irrelevant_field_suppression(module_metadata: dict[str, Any]) -> float:
    value = module_metadata.get("irrelevant_field_suppression")
    if value is not None:
        return 1.0 if safe_bool(value) else 0.0
    field_generation = str(module_metadata.get("field_generation", "")).lower()
    if field_generation in {"scrambled", "random"}:
        return 0.0
    return 0.5


def compute_afci_records(traces: pd.DataFrame) -> pd.DataFrame:
    if traces.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for record in traces.to_dict(orient="records"):
        row = pd.Series(record)
        final_action = str(row.get("final_action", ""))
        first_impulse = str(row.get("first_impulse", final_action))
        module_metadata = as_dict(row.get("module_metadata"))
        reason = _reason_alignment(row, final_action, module_metadata)
        memory = _memory_alignment(row, final_action, module_metadata)
        veto = _veto_alignment(row, final_action, first_impulse, module_metadata)
        self_score = _self_alignment(row, final_action, module_metadata)
        irrelevant = _irrelevant_field_suppression(module_metadata)
        afci = float(np.mean([reason, memory, veto, self_score]))
        rows.append(
            {
                "dataset_namespace": row.get("output_namespace", row.get("dataset_namespace", "")),
                "variant": row.get("variant", ""),
                "seed": int(safe_float(row.get("seed", 0))),
                "event_id": row.get("event_id", ""),
                "chunk_index": int(safe_float(row.get("chunk_index", 0))),
                "reason_action_alignment": reason,
                "memory_action_alignment": memory,
                "veto_action_alignment": veto,
                "self_action_alignment": self_score,
                "irrelevant_field_suppression": irrelevant,
                "AFCI": afci,
                "AFCI_with_suppression": float(np.mean([reason, memory, veto, self_score, irrelevant])),
            }
        )
    return pd.DataFrame(rows)


def summarize_afci(records: pd.DataFrame) -> pd.DataFrame:
    if records.empty:
        return pd.DataFrame()
    columns = [
        "reason_action_alignment",
        "memory_action_alignment",
        "veto_action_alignment",
        "self_action_alignment",
        "irrelevant_field_suppression",
        "AFCI",
        "AFCI_with_suppression",
    ]
    return (
        records.groupby(["dataset_namespace", "variant", "seed"], sort=True)[columns]
        .mean(numeric_only=True)
        .reset_index()
    )


def schema_equalized_gate(summary: pd.DataFrame, *, bootstrap_samples: int = 1000, seed: int = 7) -> dict[str, Any]:
    if summary.empty:
        return {"status": "needs_review", "reason": "missing_afci_summary"}
    controls = [
        "A5_plain",
        "A5_schema_random_fields",
        "A5_schema_posthoc_fields",
        "A5_schema_scrambled_fields",
        "A4_posthoc_fields",
        "A4_scrambled_fields",
        "A5_long_scratchpad",
    ]
    dataset_count = int(summary["dataset_namespace"].nunique())
    rows: list[dict[str, Any]] = []
    dataset_deltas: list[float] = []
    for control in controls:
        control_rows = summary[summary["variant"].astype(str).eq(control)]
        if control_rows.empty:
            continue
        pass_count = 0
        for namespace, frame in summary.groupby("dataset_namespace", sort=True):
            a4 = frame[frame["variant"].astype(str).eq("A4_full")]
            other = frame[frame["variant"].astype(str).eq(control)]
            if a4.empty or other.empty:
                continue
            delta = float(a4["AFCI"].mean() - other["AFCI"].mean())
            dataset_deltas.append(delta)
            pass_count += int(delta > 0)
        rows.append(
            {
                "control_variant": control,
                "dataset_count": dataset_count,
                "a4_afci_gt_control_dataset_count": pass_count,
                "control_recovered_a4_dataset_count": dataset_count - pass_count,
            }
        )
    control_status = pd.DataFrame(rows)
    min_pass = int(control_status["a4_afci_gt_control_dataset_count"].min()) if not control_status.empty else 0
    low_ci = float("nan")
    if dataset_deltas:
        rng = np.random.default_rng(seed)
        deltas = np.asarray(dataset_deltas, dtype=float)
        boot = [float(rng.choice(deltas, size=len(deltas), replace=True).mean()) for _ in range(max(10, bootstrap_samples))]
        low_ci = float(np.quantile(boot, 0.025))
    status = "pass" if dataset_count and min_pass >= dataset_count and (np.isnan(low_ci) or low_ci > 0) else "needs_review"
    return {
        "status": status,
        "dataset_count": dataset_count,
        "control_count": int(len(control_status)),
        "min_a4_afci_gt_control_dataset_count": min_pass,
        "cluster_bootstrap_afci_delta_low95": low_ci,
        "pass_rule": "A4_full AFCI exceeds each schema/posthoc/scrambled control in all datasets; cluster bootstrap low95 must be > 0 when estimable.",
        "control_status": control_status.to_dict(orient="records"),
    }


def counterfactual_records(behaviors: pd.DataFrame) -> pd.DataFrame:
    if behaviors.empty:
        return pd.DataFrame()
    frame = behaviors.copy()
    frame["event_id"] = frame.apply(choice_event_id, axis=1)
    frame["base_event_id"] = frame["event_id"].map(base_event_id)
    frame["probe_type"] = frame["output_namespace"].astype(str).map(probe_type_from_namespace)
    frame["dataset_namespace"] = frame["output_namespace"].astype(str).map(base_namespace)
    frame = frame[frame["probe_type"].astype(str).isin(BEHAVIORAL_PROBE_TYPES)].copy()
    if frame.empty:
        return pd.DataFrame()
    frame = add_action_canonical_columns(frame, action_column="final_action")
    frame["first_impulse_canonical"] = frame["first_impulse"].map(canonical_action_rule)
    frame["changed_from_first_impulse"] = frame["canonical_action_rule"].astype(str).ne(frame["first_impulse_canonical"].astype(str))
    return frame


def counterfactual_metric_tables(records: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if records.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    key = ["dataset_namespace", "variant", "seed", "base_event_id"]
    target = records[key + ["probe_type", "canonical_action_rule", "changed_from_first_impulse"]].copy()
    wide = target.pivot_table(index=key, columns="probe_type", values="canonical_action_rule", aggfunc="first").reset_index()
    baseline_candidates = ["irrelevant_cue", "placebo_cue", "memory_irrelevant"]
    change_rows: list[dict[str, Any]] = []
    for _, row in wide.iterrows():
        baseline = ""
        for candidate in baseline_candidates:
            value = str(row.get(candidate, ""))
            if value and value != "nan":
                baseline = value
                break
        for probe, metric in TARGET_PROBE_TO_METRIC.items():
            probe_action = str(row.get(probe, ""))
            if not baseline or not probe_action or probe_action == "nan":
                continue
            change_rows.append(
                {
                    "dataset_namespace": row["dataset_namespace"],
                    "variant": row["variant"],
                    "seed": row["seed"],
                    "base_event_id": row["base_event_id"],
                    "probe_type": probe,
                    "metric": metric,
                    "changed_vs_baseline": probe_action != baseline,
                }
            )
    paired_changes = pd.DataFrame(change_rows)
    baseline_records = records[records["probe_type"].isin(baseline_candidates)].copy()
    irrelevant_rates = (
        baseline_records.groupby(["dataset_namespace", "variant", "seed"], sort=True)["changed_from_first_impulse"]
        .mean()
        .reset_index(name="irrelevant_change_rate")
        if not baseline_records.empty
        else pd.DataFrame()
    )
    target_rates = (
        paired_changes.groupby(["dataset_namespace", "variant", "seed", "probe_type", "metric"], sort=True)["changed_vs_baseline"]
        .mean()
        .reset_index(name="target_change_rate")
        if not paired_changes.empty
        else pd.DataFrame()
    )
    metrics_rows: list[dict[str, Any]] = []
    for keys, group in target_rates.groupby(["dataset_namespace", "variant", "seed"], sort=True) if not target_rates.empty else []:
        dataset_namespace, variant, seed = keys
        row: dict[str, Any] = {"dataset_namespace": dataset_namespace, "variant": variant, "seed": seed}
        base = irrelevant_rates[
            irrelevant_rates["dataset_namespace"].astype(str).eq(str(dataset_namespace))
            & irrelevant_rates["variant"].astype(str).eq(str(variant))
            & irrelevant_rates["seed"].astype(str).eq(str(seed))
        ]
        irrelevant_rate = float(base["irrelevant_change_rate"].iloc[0]) if not base.empty else float("nan")
        row["irrelevant_change_rate"] = irrelevant_rate
        for metric in sorted(set(TARGET_PROBE_TO_METRIC.values())):
            metric_frame = group[group["metric"].astype(str).eq(metric)]
            target_rate = float(metric_frame["target_change_rate"].mean()) if not metric_frame.empty else float("nan")
            row[f"{metric}_target_rate"] = target_rate
            row[metric] = target_rate - irrelevant_rate if np.isfinite(target_rate) and np.isfinite(irrelevant_rate) else float("nan")
        target_values = [float(value) for value in group["target_change_rate"].tolist() if np.isfinite(float(value))]
        row["CueSpecificity"] = float(np.mean(target_values) - irrelevant_rate) if target_values and np.isfinite(irrelevant_rate) else float("nan")
        components = [row.get("B_RSI"), row.get("B_MCI"), row.get("B_VEI"), row.get("B_SCI"), row.get("CueSpecificity")]
        clean_components = [float(value) for value in components if value is not None and np.isfinite(float(value))]
        row["BehavioralStructuredControl"] = float(np.mean(clean_components)) if clean_components else float("nan")
        metrics_rows.append(row)
    metrics = pd.DataFrame(metrics_rows)
    summary = (
        metrics.groupby(["dataset_namespace", "variant"], sort=True)[["B_RSI", "B_MCI", "B_VEI", "B_SCI", "CueSpecificity", "BehavioralStructuredControl"]]
        .mean(numeric_only=True)
        .reset_index()
        if not metrics.empty
        else pd.DataFrame()
    )
    return paired_changes, metrics, summary

