from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from freewill.ai_matrix import DatasetSpec, _dataset_specs
from freewill.config import ProjectConfig
from freewill.utils import ensure_dir, read_parquet, write_json, write_parquet


RESIDUAL_BLOCKS = ["delta_L3T_resid", "delta_L3V_resid", "delta_L3_resid_combined"]


def _runtime_root(config: ProjectConfig) -> Path:
    return Path(config.paths.runtime_root)


def _results_root(config: ProjectConfig) -> Path:
    return _runtime_root(config) / "results"


def _tier1_result_dir(config: ProjectConfig, spec: DatasetSpec, fold_root: Path | None = None) -> Path:
    if fold_root is not None:
        return fold_root / spec.output_namespace
    return _runtime_root(config) / "results" / "tier1" / spec.output_namespace


def _normal_approx_two_sided_p(successes: int, n: int, p0: float = 0.5) -> float:
    if n <= 0:
        return float("nan")
    observed = successes / n
    se = (p0 * (1.0 - p0) / n) ** 0.5
    if np.isclose(se, 0.0):
        return 1.0
    z = abs(observed - p0) / se
    return float(2.0 * (1.0 - 0.5 * (1.0 + math.erf(z / np.sqrt(2.0)))))


def _prediction_from_metrics(config: ProjectConfig, specs: list[DatasetSpec]) -> pd.DataFrame:
    metrics_path = _results_root(config) / "ai_matrix" / "metrics.parquet"
    if not metrics_path.exists():
        rows = [
            {
                "dataset_id": spec.dataset_id,
                "output_namespace": spec.output_namespace,
                "prediction_source": "default_missing_ai_matrix",
                "a4_prediction_sign": 1,
                "a5_prediction_sign": -1,
                "null_prediction_sign": 1,
            }
            for spec in specs
        ]
        return pd.DataFrame(rows)
    metrics = read_parquet(metrics_path)
    rows: list[dict[str, Any]] = []
    for spec in specs:
        frame = metrics[metrics["output_namespace"].astype(str).eq(spec.output_namespace)]
        a4 = frame[frame["variant"].astype(str).eq("A4")]
        a5 = frame[frame["variant"].astype(str).eq("A5")]
        if a4.empty or a5.empty:
            prediction = 1
            source = "missing_variant_default_positive"
            a5_prediction = -1
        else:
            a4_score = float(a4.iloc[0].get("SCI_mean", 0.0)) + float(a4.iloc[0].get("RSI_mean", 0.0)) + float(a4.iloc[0].get("VEI_mean", 0.0))
            a5_score = float(a5.iloc[0].get("SCI_mean", 0.0)) + float(a5.iloc[0].get("RSI_mean", 0.0)) + float(a5.iloc[0].get("VEI_mean", 0.0))
            prediction = 1 if a4_score >= a5_score else -1
            a5_prediction = 1 if a5_score > a4_score else -1
            source = "ai_matrix_structured_score"
        rows.append(
            {
                "dataset_id": spec.dataset_id,
                "output_namespace": spec.output_namespace,
                "prediction_source": source,
                "a4_prediction_sign": prediction,
                "a5_prediction_sign": a5_prediction,
                "null_prediction_sign": 1,
            }
        )
    return pd.DataFrame(rows)


def _load_fold_results(config: ProjectConfig, specs: list[DatasetSpec], fold_root: Path | None = None) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for spec in specs:
        path = _tier1_result_dir(config, spec, fold_root=fold_root) / "human_fold_results.parquet"
        if not path.exists():
            continue
        frame = read_parquet(path)
        if frame.empty:
            continue
        frame = frame.copy()
        frame["dataset_id"] = spec.dataset_id
        frame["output_namespace"] = spec.output_namespace
        rows.append(frame)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _cell_concordance(folds: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    if folds.empty:
        return pd.DataFrame()
    frame = folds[folds["block_name"].astype(str).isin(RESIDUAL_BLOCKS)].copy()
    if frame.empty:
        return pd.DataFrame()
    grouped = (
        frame.groupby(["dataset_id", "output_namespace", "unit_id", "block_name"], dropna=False)["r2"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "fold_mean_r2", "std": "fold_std_r2", "count": "fold_count"})
    )
    grouped["fold_se"] = grouped["fold_std_r2"].fillna(0.0) / np.sqrt(grouped["fold_count"].clip(lower=1))
    grouped["z_stat"] = grouped["fold_mean_r2"] / grouped["fold_se"].replace(0.0, np.nan)
    grouped["z_stat"] = grouped["z_stat"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    grouped["observed_sign"] = np.where(grouped["fold_mean_r2"] >= 0.0, 1, -1)
    merged = grouped.merge(predictions, on=["dataset_id", "output_namespace"], how="left")
    for column in ["a4_prediction_sign", "a5_prediction_sign", "null_prediction_sign"]:
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(1).astype(int)
        merged[f"{column.removesuffix('_prediction_sign')}_concordant"] = merged["observed_sign"].eq(merged[column])
    merged["weight"] = merged["z_stat"].abs()
    return merged


def _dataset_concordance(cells: pd.DataFrame) -> pd.DataFrame:
    if cells.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (dataset_id, namespace), frame in cells.groupby(["dataset_id", "output_namespace"], sort=True):
        weight = pd.to_numeric(frame["weight"], errors="coerce").fillna(0.0)
        if np.isclose(float(weight.sum()), 0.0):
            weighted = float(frame["a4_concordant"].mean())
        else:
            weighted = float(np.average(frame["a4_concordant"].astype(float), weights=weight))
        rows.append(
            {
                "dataset_id": dataset_id,
                "output_namespace": namespace,
                "cell_count": int(len(frame)),
                "a4_concordance": float(frame["a4_concordant"].mean()),
                "a4_weighted_concordance": weighted,
                "a5_concordance": float(frame["a5_concordant"].mean()),
                "null_concordance": float(frame["null_concordant"].mean()),
                "positive_cell_fraction": float((frame["observed_sign"] > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def _null_distribution(dataset_frame: pd.DataFrame, permutations: int, seed: int) -> pd.DataFrame:
    if dataset_frame.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    values = dataset_frame["a4_concordance"].to_numpy(dtype=float)
    rows = []
    for index in range(permutations):
        signs = rng.choice([-1.0, 1.0], size=len(values), replace=True)
        centered = 0.5 + (values - 0.5) * signs
        rows.append({"permutation": index, "mean_concordance": float(centered.mean())})
    return pd.DataFrame(rows)


def _gate_status(dataset_frame: pd.DataFrame, nulls: pd.DataFrame, *, residual: bool = False) -> dict[str, Any]:
    if dataset_frame.empty:
        return {
            "status": "fail",
            "reason": "no_residual_fold_rows",
            "residual_mode": residual,
        }
    mean_concordance = float(dataset_frame["a4_concordance"].mean())
    weighted = float(dataset_frame["a4_weighted_concordance"].mean())
    a5 = float(dataset_frame["a5_concordance"].mean())
    random_null = float(dataset_frame["null_concordance"].mean())
    successes = int((dataset_frame["a4_concordance"] > 0.5).sum())
    p_value = _normal_approx_two_sided_p(successes, len(dataset_frame))
    if not nulls.empty:
        permutation_p = float((int((nulls["mean_concordance"] >= mean_concordance).sum()) + 1) / (len(nulls) + 1))
    else:
        permutation_p = float("nan")
    pass_gate = mean_concordance > 0.5 and weighted > 0.5 and mean_concordance > a5 and mean_concordance >= random_null and (
        (np.isfinite(p_value) and p_value < 0.05) or (np.isfinite(permutation_p) and permutation_p < 0.05)
    )
    return {
        "status": "pass" if pass_gate else "needs_review",
        "residual_mode": residual,
        "dataset_count": int(len(dataset_frame)),
        "mean_dataset_concordance": mean_concordance,
        "mean_weighted_concordance": weighted,
        "mean_a5_concordance": a5,
        "mean_null_concordance": random_null,
        "dataset_success_count": successes,
        "dataset_level_p": p_value,
        "sign_flip_permutation_p": permutation_p,
        "interpretation": (
            "supportive_human_residual_direction"
            if pass_gate
            else "human_residual_direction_not_robust_under_cluster_aware_gate"
        ),
    }


def run_sign_test(
    config: ProjectConfig,
    *,
    dataset_ids: Iterable[str] | None = None,
    fold_root: str | Path | None = None,
    output_name: str = "sign_test",
    residual: bool = False,
    permutations: int = 1000,
) -> dict[str, Any]:
    specs = _dataset_specs(config, dataset_ids, smoke=False)
    output_dir = ensure_dir(_results_root(config) / output_name)
    fold_base = Path(fold_root) if fold_root else None
    predictions = _prediction_from_metrics(config, specs)
    folds = _load_fold_results(config, specs, fold_root=fold_base)
    cells = _cell_concordance(folds, predictions)
    datasets = _dataset_concordance(cells)
    nulls = _null_distribution(datasets, permutations, int(config.ai_matrix.random_seed))
    gate = _gate_status(datasets, nulls, residual=residual)
    write_parquet(predictions, output_dir / "frozen_predictions.parquet")
    write_parquet(cells, output_dir / "cell_concordance.parquet")
    write_parquet(datasets, output_dir / "dataset_concordance.parquet")
    write_parquet(nulls, output_dir / "nulls.parquet")
    write_json(output_dir / "gate_status.json", gate)
    return {
        "predictions": predictions,
        "cell_concordance": cells,
        "dataset_concordance": datasets,
        "nulls": nulls,
        "gate_status": gate,
    }

