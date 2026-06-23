from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TraceSource:
    evidence_block: str
    model: str
    model_family: str
    model_size_b: float
    paths: tuple[Path, ...]


SOURCES: tuple[TraceSource, ...] = (
    TraceSource(
        "full_control_matrix",
        "Qwen2.5-7B-Instruct",
        "Qwen2.5",
        7.0,
        (Path("results/paper1_revision/remote_5090_run2_qwen25_7b_20260508/merged/traces.parquet"),),
    ),
    TraceSource(
        "full_control_matrix",
        "Qwen2.5-14B-Instruct",
        "Qwen2.5",
        14.0,
        (Path("results/paper1_revision/remote_5090_run1_qwen14b_20260507/merged/traces.parquet"),),
    ),
    TraceSource(
        "full_control_matrix",
        "Qwen2.5-32B-Instruct-4bit",
        "Qwen2.5",
        32.0,
        (Path("results/paper1_revision/remote_5090_run2_qwen25_32b_20260508/merged/traces.parquet"),),
    ),
    TraceSource(
        "full_control_matrix",
        "Mistral-7B-Instruct-v0.3",
        "Mistral",
        7.0,
        (
            Path("results/paper1_revision/CSB_open_weight_pilot/CSB_run3_mistral7b_full_t768_shard_00_of_02/traces.parquet"),
            Path("results/paper1_revision/CSB_open_weight_pilot/CSB_run3_mistral7b_full_t768_shard_01_of_02/traces.parquet"),
        ),
    ),
    TraceSource(
        "qwen32b_strict_lesion_slice",
        "Qwen2.5-32B-Instruct strict lesion",
        "Qwen2.5",
        32.0,
        (
            Path("results/paper1_revision/CSB_open_weight_pilot/CSB_run3_qwen25_32b_strict_lesion_e80_shard_00_of_02/traces.parquet"),
            Path("results/paper1_revision/CSB_open_weight_pilot/CSB_run3_qwen25_32b_strict_lesion_e80_shard_01_of_02/traces.parquet"),
        ),
    ),
)


def safe_json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return {}
    text = str(value).strip()
    if not text:
        return {}
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return decoded if isinstance(decoded, dict) else {}


def entropy_bits(series: pd.Series) -> float:
    counts = series.fillna("").astype(str).value_counts()
    if counts.empty:
        return 0.0
    probs = counts.to_numpy(dtype=float) / float(counts.sum())
    return float(-(probs * np.log2(probs)).sum())


def load_traces(root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    for source in SOURCES:
        for path in source.paths:
            full = root / path
            if not full.exists():
                missing.append(str(path))
                continue
            frame = pd.read_parquet(full)
            frame = frame.copy()
            frame["evidence_block"] = source.evidence_block
            frame["model"] = source.model
            frame["model_family"] = source.model_family
            frame["model_size_b"] = source.model_size_b
            frame["source_path"] = str(path)
            frames.append(frame)
    if missing:
        raise FileNotFoundError("Missing expected trace files:\n" + "\n".join(missing))
    if not frames:
        raise RuntimeError("No trace frames loaded.")
    traces = pd.concat(frames, ignore_index=True)
    for col in ["raw_generation", "final_action", "final_action_rationale", "first_impulse"]:
        if col not in traces.columns:
            traces[col] = ""
        traces[f"{col}_chars"] = traces[col].fillna("").astype(str).str.len()
    provider = traces["provider_metadata"].map(safe_json) if "provider_metadata" in traces.columns else pd.Series([{}] * len(traces))
    for key in [
        "request_temperature",
        "request_top_p",
        "batch_size",
        "offline_policy_used",
        "context_used",
        "current_intervention_context_used",
        "fallback_used",
        "error_type",
    ]:
        traces[f"provider_{key}"] = provider.map(lambda meta, k=key: meta.get(k, None))
    token_cols = {"prompt_tokens", "completion_tokens", "total_tokens"}
    traces.attrs["provider_token_columns_present"] = sorted(token_cols.intersection(traces.columns))
    key_cols = [
        "evidence_block",
        "model",
        "architecture",
        "task_family",
        "probe_component",
        "variant",
        "seed",
        "event_id",
    ]
    traces["duplicate_key"] = traces.duplicated(key_cols, keep=False)
    return traces


def summarize_variant(traces: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["evidence_block", "model", "model_family", "model_size_b", "variant"]
    for keys, subset in traces.groupby(group_cols, observed=True):
        block, model, family, size_b, variant = keys
        temperatures = sorted({str(x) for x in subset["provider_request_temperature"].dropna().unique()})
        top_ps = sorted({str(x) for x in subset["provider_request_top_p"].dropna().unique()})
        batch_sizes = sorted({str(x) for x in subset["provider_batch_size"].dropna().unique()})
        rows.append(
            {
                "evidence_block": block,
                "model": model,
                "model_family": family,
                "model_size_b": size_b,
                "variant": variant,
                "rows": int(len(subset)),
                "raw_generation_chars_mean": float(subset["raw_generation_chars"].mean()),
                "raw_generation_chars_median": float(subset["raw_generation_chars"].median()),
                "raw_generation_chars_p90": float(subset["raw_generation_chars"].quantile(0.90)),
                "final_action_rationale_chars_mean": float(subset["final_action_rationale_chars"].mean()),
                "final_action_chars_mean": float(subset["final_action_chars"].mean()),
                "final_action_entropy_bits": entropy_bits(subset["final_action"]),
                "expected_action_entropy_bits": entropy_bits(subset["expected_action"]) if "expected_action" in subset.columns else np.nan,
                "distinct_final_actions": int(subset["final_action"].fillna("").astype(str).nunique()),
                "provider_temperatures": ",".join(temperatures),
                "provider_top_p": ",".join(top_ps),
                "provider_batch_sizes": ",".join(batch_sizes),
                "offline_policy_rows": int(subset["provider_offline_policy_used"].fillna(False).astype(bool).sum()),
                "context_used_rows": int(subset["provider_context_used"].fillna(False).astype(bool).sum()),
                "current_context_used_rows": int(subset["provider_current_intervention_context_used"].fillna(False).astype(bool).sum()),
                "fallback_rows": int(subset["provider_fallback_used"].fillna(False).astype(bool).sum()),
                "duplicate_rows": int(subset["duplicate_key"].sum()),
            }
        )
    summary = pd.DataFrame(rows).sort_values(["evidence_block", "model", "variant"]).reset_index(drop=True)

    structured = summary[summary["variant"].eq("structured")][
        ["evidence_block", "model", "raw_generation_chars_mean", "final_action_entropy_bits"]
    ].rename(
        columns={
            "raw_generation_chars_mean": "structured_raw_generation_chars_mean",
            "final_action_entropy_bits": "structured_final_action_entropy_bits",
        }
    )
    summary = summary.merge(structured, on=["evidence_block", "model"], how="left")
    summary["raw_generation_char_ratio_vs_structured"] = (
        summary["raw_generation_chars_mean"] / summary["structured_raw_generation_chars_mean"]
    )
    summary["final_action_entropy_gap_vs_structured"] = (
        summary["final_action_entropy_bits"] - summary["structured_final_action_entropy_bits"]
    )
    return summary


def summarize_control_gaps(summary: pd.DataFrame) -> pd.DataFrame:
    controls = summary[~summary["variant"].eq("structured")].copy()
    rows: list[dict[str, Any]] = []
    for (block, variant), subset in controls.groupby(["evidence_block", "variant"], observed=True):
        rows.append(
            {
                "evidence_block": block,
                "variant": variant,
                "models": int(subset["model"].nunique()),
                "rows": int(subset["rows"].sum()),
                "mean_raw_char_ratio_vs_structured": float(subset["raw_generation_char_ratio_vs_structured"].mean()),
                "min_raw_char_ratio_vs_structured": float(subset["raw_generation_char_ratio_vs_structured"].min()),
                "max_raw_char_ratio_vs_structured": float(subset["raw_generation_char_ratio_vs_structured"].max()),
                "mean_final_action_entropy_gap_vs_structured": float(subset["final_action_entropy_gap_vs_structured"].mean()),
                "min_final_action_entropy_gap_vs_structured": float(subset["final_action_entropy_gap_vs_structured"].min()),
                "max_final_action_entropy_gap_vs_structured": float(subset["final_action_entropy_gap_vs_structured"].max()),
            }
        )
    return pd.DataFrame(rows).sort_values(["evidence_block", "variant"]).reset_index(drop=True)


def write_frame(df: pd.DataFrame, output_dir: Path, stem: str) -> None:
    df.to_csv(output_dir / f"{stem}.csv", index=False)
    df.to_parquet(output_dir / f"{stem}.parquet", index=False)


def fmt(value: Any, digits: int = 3) -> str:
    try:
        if pd.isna(value):
            return "n/a"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def render_summary(traces: pd.DataFrame, variant_summary: pd.DataFrame, gap_summary: pd.DataFrame, output_dir: Path) -> str:
    token_cols = traces.attrs.get("provider_token_columns_present", [])
    lines: list[str] = []
    lines.append("# CSB Open-Weight Token/Entropy Proxy Audit")
    lines.append("")
    lines.append(f"- Output directory: `{output_dir.as_posix()}`")
    lines.append(f"- Trace rows loaded: `{len(traces)}`")
    lines.append(f"- Duplicate analysis keys: `{int(traces['duplicate_key'].sum())}`")
    lines.append(f"- Provider token columns present: `{', '.join(token_cols) if token_cols else 'none'}`")
    lines.append("")
    lines.append("## Control-Level Proxy Gaps")
    lines.append("")
    lines.append("| block | variant | models | rows | mean char ratio vs structured | entropy gap vs structured |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for _, row in gap_summary.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['evidence_block']}`",
                    f"`{row['variant']}`",
                    str(int(row["models"])),
                    str(int(row["rows"])),
                    fmt(row["mean_raw_char_ratio_vs_structured"], 3),
                    fmt(row["mean_final_action_entropy_gap_vs_structured"], 3),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- This is a proxy audit, not a strict compute- or token-matched experiment.")
    lines.append("- Exact provider `prompt_tokens`, `completion_tokens` and `total_tokens` were not uniformly retained in the Run2/Run3 trace tables.")
    lines.append("- Completion JSON character length and canonical final-action entropy are available for all retained traces and can support conservative wording about verbosity/action-distribution diagnostics.")
    lines.append("- Do not claim strict token matching or strict entropy matching from these outputs.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit token/entropy proxies for CSB open-weight Run2/Run3 traces.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/paper1_revision/remote_5090_run3_20260508/token_entropy_proxy_audit"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    traces = load_traces(root)
    variant_summary = summarize_variant(traces)
    gap_summary = summarize_control_gaps(variant_summary)

    write_frame(variant_summary, output_dir, "variant_token_entropy_proxy_summary")
    write_frame(gap_summary, output_dir, "control_token_entropy_proxy_gaps")
    manifest = {
        "row_count": int(len(traces)),
        "duplicate_analysis_keys": int(traces["duplicate_key"].sum()),
        "provider_token_columns_present": traces.attrs.get("provider_token_columns_present", []),
        "sources": [
            {
                "evidence_block": source.evidence_block,
                "model": source.model,
                "paths": [str(path) for path in source.paths],
            }
            for source in SOURCES
        ],
    }
    (output_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary = render_summary(traces, variant_summary, gap_summary, output_dir)
    (output_dir / "token_entropy_proxy_audit_summary.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()

