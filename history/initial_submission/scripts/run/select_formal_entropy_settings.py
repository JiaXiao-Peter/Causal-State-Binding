from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from freewill.structured_control_validation import FINITE_ACTIONS, CSB_ENTROPY_CANDIDATE_SETTINGS


def _entropy(values: pd.Series) -> float:
    counts = values.astype(str).value_counts(normalize=True)
    if counts.empty:
        return float("nan")
    return float(-(counts * np.log2(counts)).sum())


def _metadata_frame(traces: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for value in traces.get("provider_metadata", pd.Series([], dtype=object)).tolist():
        if isinstance(value, dict):
            rows.append(value)
        elif isinstance(value, str):
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError:
                decoded = {}
            rows.append(decoded if isinstance(decoded, dict) else {})
        else:
            rows.append({})
    return pd.DataFrame(rows)


def _load_traces(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        trace_path = path if path.is_file() else path / "traces.parquet"
        if trace_path.exists():
            frame = pd.read_parquet(trace_path)
            gate_path = trace_path.parent / "gate_status.json"
            if gate_path.exists() and "model" not in frame.columns:
                gate = json.loads(gate_path.read_text(encoding="utf-8"))
                profile = str(gate.get("profile", trace_path.parent.name))
                model_key = str(gate.get("model_path", gate.get("model_identifier", trace_path.parent.name)))
                prefix = "formal_calibration_"
                shard_marker = "_shard_"
                if profile.startswith(prefix) and shard_marker in profile:
                    model_key = profile[len(prefix) : profile.index(shard_marker)]
                frame = frame.copy()
                frame["model"] = model_key
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _select(traces: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    if traces.empty:
        return {}, pd.DataFrame()
    meta = _metadata_frame(traces)
    for column in ["model", "prompt_tokens", "completion_tokens", "total_tokens", "error_type"]:
        if column not in traces.columns and column in meta.columns:
            traces[column] = meta[column]
    if "model" not in traces.columns:
        traces["model"] = traces.get("model_path", "unknown_model")
    traces["error_type"] = traces.get("error_type", pd.Series("", index=traces.index)).fillna("").astype(str)
    rows = []
    group_cols = ["model", "architecture", "variant"]
    summary = (
        traces.groupby(group_cols, dropna=False)
        .agg(
            final_action_entropy_bits=("final_action", _entropy),
            mean_total_tokens=("total_tokens", "mean"),
            rows=("variant", "size"),
            parse_failure_rate=("error_type", lambda values: float(values.astype(str).ne("").mean())),
        )
        .reset_index()
    )
    settings: dict[str, Any] = {}
    for (model, architecture), frame in summary.groupby(["model", "architecture"], dropna=False):
        structured = frame[frame["variant"].eq("structured")]
        if structured.empty:
            continue
        structured_entropy = float(structured["final_action_entropy_bits"].iloc[0])
        structured_tokens = float(structured["mean_total_tokens"].iloc[0] or 0.0)
        structured_rows = traces[
            traces["model"].astype(str).eq(str(model))
            & traces["architecture"].astype(str).eq(str(architecture))
            & traces["variant"].astype(str).eq("structured")
        ]
        action_counts = {
            action: int(structured_rows["final_action"].astype(str).eq(action).sum())
            for action in FINITE_ACTIONS
        }
        action_total = int(sum(action_counts.values()))
        action_probabilities = (
            {action: float(action_counts[action] / action_total) for action in FINITE_ACTIONS}
            if action_total > 0
            else {action: float(1.0 / len(FINITE_ACTIONS)) for action in FINITE_ACTIONS}
        )
        candidates = frame[frame["variant"].isin(CSB_ENTROPY_CANDIDATE_SETTINGS.keys())].copy()
        if candidates.empty:
            variant = "calibrated_action_prior_schedule_v1"
            chosen = {"temperature": 0.2, "top_p": 1.0}
            selection_score = 0.0
            entropy_gap = 0.0
            token_ratio_gap = 0.0
            parse_failure_rate = float(structured["parse_failure_rate"].iloc[0])
        else:
            candidates["entropy_gap_abs"] = (candidates["final_action_entropy_bits"] - structured_entropy).abs()
            candidates["token_ratio_gap_abs"] = (
                candidates["mean_total_tokens"].astype(float) / structured_tokens - 1.0
            ).abs() if structured_tokens > 0 else 0.0
            candidates["selection_score"] = (
                candidates["entropy_gap_abs"].fillna(999.0)
                + candidates["token_ratio_gap_abs"].fillna(999.0)
                + candidates["parse_failure_rate"].fillna(1.0) * 10.0
            )
            best = candidates.sort_values(["selection_score", "entropy_gap_abs", "token_ratio_gap_abs"], kind="stable").iloc[0]
            variant = str(best["variant"])
            chosen = dict(CSB_ENTROPY_CANDIDATE_SETTINGS[variant])
            selection_score = float(best["selection_score"])
            entropy_gap = float(best["entropy_gap_abs"])
            token_ratio_gap = float(best["token_ratio_gap_abs"])
            parse_failure_rate = float(best["parse_failure_rate"])
        settings.setdefault(str(model), {})[str(architecture)] = {
            **chosen,
            "selected_candidate_variant": variant,
            "selection_score": selection_score,
            "calibration_entropy_gap_bits": entropy_gap,
            "calibration_token_ratio_gap": token_ratio_gap,
            "calibration_parse_failure_rate": parse_failure_rate,
            "entropy_strategy": "calibrated_action_prior_schedule_v1",
            "structured_entropy_bits": structured_entropy,
            "action_counts": action_counts,
            "action_probabilities": action_probabilities,
            "calibration_rows": action_total,
        }
        rows.append(
            {
                "model": model,
                "architecture": architecture,
                "selected_candidate_variant": variant,
                "entropy_strategy": "calibrated_action_prior_schedule_v1",
                "structured_entropy_bits": structured_entropy,
                "calibration_rows": action_total,
                **{f"p_{action}": action_probabilities[action] for action in FINITE_ACTIONS},
                **settings[str(model)][str(architecture)],
            }
        )
    return settings, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Select frozen entropy decoding settings from calibration traces.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Calibration profile directories or traces.parquet files.")
    parser.add_argument("--output-dir", default="results/paper1_revision/remote_5090_run3_20260508/formal_entropy_calibration")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    traces = _load_traces([Path(item) for item in args.inputs])
    settings, table = _select(traces)
    (output_dir / "entropy_decoding_settings_by_model.json").write_text(
        json.dumps(settings, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    table.to_csv(output_dir / "entropy_decoding_selection.csv", index=False, encoding="utf-8")
    if not table.empty:
        table.to_parquet(output_dir / "entropy_decoding_selection.parquet", index=False)
    print(json.dumps({"settings_path": str(output_dir / "entropy_decoding_settings_by_model.json"), "selected_rows": len(table)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

