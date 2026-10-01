from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from freewill.real_task_predictive_validity import _design_matrix  # noqa: E402


RESULT_DIR = ROOT / "results/paper1_revision/real_task_predictive_validity/swebench_v2_full300_api_20260529_merged"
TABLE_DIR = ROOT / "manuscript/tables"
SOURCE_DIR = ROOT / "manuscript/figures/source_data"
FIGURE_DIR = ROOT / "manuscript/figures/extended_data"

BASELINE_COLUMNS = [
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
AUGMENTED_COLUMNS = [*BASELINE_COLUMNS, "state_binding_score"]
CATEGORICAL = ["model", "repo"]


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.fillna(False).astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def _cv_predict(frame: pd.DataFrame, y: Sequence[int], columns: Sequence[str], *, folds: int = 5) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler

    clean_y = np.array([int(value) for value in y], dtype=int)
    groups = frame["instance_id"].astype(str).to_numpy()
    unique_groups = np.array(sorted(set(groups)))
    predictions = np.full(len(frame), np.nan, dtype=float)
    n_splits = max(2, min(int(folds), len(unique_groups)))
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
        x_all = _design_matrix(combined, columns, categorical=CATEGORICAL)
        x_train = x_all.iloc[: len(train_frame)].to_numpy(dtype=float)
        x_test = x_all.iloc[len(train_frame) :].to_numpy(dtype=float)
        scaler = StandardScaler(with_mean=False)
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        model = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")
        model.fit(x_train, clean_y[train_mask])
        predictions[test_mask] = model.predict_proba(x_test)[:, 1]
    return predictions


def _ntile(values: pd.Series, n: int = 10) -> pd.Series:
    order = values.sort_values(kind="mergesort").index.to_list()
    bins = pd.Series(index=values.index, dtype=int)
    total = len(order)
    for rank, idx in enumerate(order, start=1):
        bins.loc[idx] = int(math.ceil(rank * n / max(1, total)))
    return bins.astype(int)


def _long_predictions(pred: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in pred.iterrows():
        for col, method in [
            ("baseline_probability", "Full non-CSB baseline"),
            ("csb_probability", "Baseline + CSB"),
        ]:
            rows.append(
                {
                    "method": method,
                    "instance_id": row["instance_id"],
                    "hit_at3": int(row["hit_at3"]),
                    "hard_violation": int(row["hard_violation"]),
                    "predicted_probability": float(row[col]),
                }
            )
    return pd.DataFrame(rows)


def _calibration_summary(pred: pd.DataFrame) -> pd.DataFrame:
    long = _long_predictions(pred)
    parts = []
    for method, group in long.groupby("method", sort=True):
        group = group.copy()
        group["bin"] = _ntile(group["predicted_probability"], 10)
        parts.append(
            group.groupby("bin", as_index=False)
            .agg(
                n=("hit_at3", "size"),
                mean_predicted=("predicted_probability", "mean"),
                observed_hit_at3=("hit_at3", "mean"),
                hard_violation_rate=("hard_violation", "mean"),
            )
            .assign(method=method)
        )
    return pd.concat(parts, axis=0, ignore_index=True)[
        ["method", "bin", "n", "mean_predicted", "observed_hit_at3", "hard_violation_rate"]
    ]


def _threshold_summary(pred: pd.DataFrame, thresholds: Sequence[float]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    long = _long_predictions(pred)
    for (method, threshold), group in long.assign(_key=1).merge(
        pd.DataFrame({"threshold": list(thresholds), "_key": 1}), on="_key"
    ).groupby(["method", "threshold"], sort=True):
        accepted = group["predicted_probability"] >= float(threshold)
        accepted_count = int(accepted.sum())
        hit = group.loc[accepted, "hit_at3"]
        hard = group.loc[accepted, "hard_violation"]
        n_total = int(len(group))
        rows.append(
            {
                "method": method,
                "threshold": float(threshold),
                "n_total": n_total,
                "n_accepted": accepted_count,
                "accepted_fraction": float(accepted.mean()),
                "accepted_hit_at3": float(hit.mean()) if accepted_count else float("nan"),
                "accepted_hard_violation": float(hard.mean()) if accepted_count else float("nan"),
                "net_benefit": float(
                    ((accepted & (group["hit_at3"] == 1)).sum() / n_total)
                    - ((accepted & (group["hit_at3"] == 0)).sum() / n_total)
                    * (float(threshold) / (1.0 - float(threshold)))
                )
                if threshold < 1.0
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _bootstrap_bands(pred: pd.DataFrame, thresholds: Sequence[float], *, samples: int = 500, seed: int = 31) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    issue_ids = np.array(sorted(pred["instance_id"].astype(str).unique()))
    calibration_rows: list[pd.DataFrame] = []
    threshold_rows: list[pd.DataFrame] = []
    for idx in range(samples):
        sampled = rng.choice(issue_ids, size=len(issue_ids), replace=True)
        boot = pd.concat([pred[pred["instance_id"].astype(str) == issue].copy() for issue in sampled], ignore_index=True)
        calibration_rows.append(_calibration_summary(boot).assign(bootstrap_index=idx))
        threshold_rows.append(_threshold_summary(boot, thresholds).assign(bootstrap_index=idx))
    calibration = pd.concat(calibration_rows, ignore_index=True)
    cal_ci = (
        calibration.groupby(["method", "bin"], as_index=False)
        .agg(
            observed_hit_at3_lower=("observed_hit_at3", lambda x: float(np.nanquantile(x, 0.025))),
            observed_hit_at3_upper=("observed_hit_at3", lambda x: float(np.nanquantile(x, 0.975))),
            hard_violation_lower=("hard_violation_rate", lambda x: float(np.nanquantile(x, 0.025))),
            hard_violation_upper=("hard_violation_rate", lambda x: float(np.nanquantile(x, 0.975))),
        )
    )
    threshold = pd.concat(threshold_rows, ignore_index=True)
    thr_ci = (
        threshold.groupby(["method", "threshold"], as_index=False)
        .agg(
            accepted_fraction_lower=("accepted_fraction", lambda x: float(np.nanquantile(x, 0.025))),
            accepted_fraction_upper=("accepted_fraction", lambda x: float(np.nanquantile(x, 0.975))),
            accepted_hit_at3_lower=("accepted_hit_at3", lambda x: float(np.nanquantile(x, 0.025))),
            accepted_hit_at3_upper=("accepted_hit_at3", lambda x: float(np.nanquantile(x, 0.975))),
            accepted_hard_violation_lower=("accepted_hard_violation", lambda x: float(np.nanquantile(x, 0.025))),
            accepted_hard_violation_upper=("accepted_hard_violation", lambda x: float(np.nanquantile(x, 0.975))),
            net_benefit_lower=("net_benefit", lambda x: float(np.nanquantile(x, 0.025))),
            net_benefit_upper=("net_benefit", lambda x: float(np.nanquantile(x, 0.975))),
        )
    )
    return cal_ci, thr_ci


def _write_figures(calibration: pd.DataFrame, threshold: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    palette = {"Full non-CSB baseline": "#D8A83F", "Baseline + CSB": "#3B7EAE"}
    fig, ax = plt.subplots(figsize=(4.7, 3.6), dpi=220)
    ax.plot([0, 1], [0, 1], "--", color="0.55", linewidth=0.8)
    for method, group in calibration.groupby("method", sort=True):
        group = group.sort_values("mean_predicted")
        color = palette.get(method, "0.3")
        ax.fill_between(group["mean_predicted"], group["observed_hit_at3_lower"], group["observed_hit_at3_upper"], color=color, alpha=0.13)
        ax.plot(group["mean_predicted"], group["observed_hit_at3"], color=color, linewidth=1.2, label=method)
        ax.scatter(group["mean_predicted"], group["observed_hit_at3"], s=np.clip(group["n"], 15, 60), color=color, edgecolor="white", linewidth=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Predicted task-only implementation-file hit@3 probability")
    ax.set_ylabel("Observed task-only implementation-file hit@3 rate")
    ax.set_title("SWE-bench Lite localization reliability calibration", fontsize=8)
    ax.legend(frameon=False, fontsize=6, loc="lower right")
    fig.tight_layout()
    for suffix in ["png", "display.png", "svg", "pdf"]:
        fig.savefig(FIGURE_DIR / f"extended_data_figure9_swebench_reliability_calibration.{suffix}")
    plt.close(fig)

    measures = [
        ("accepted_fraction", "Accepted fraction"),
        ("accepted_hit_at3", "Accepted hit@3"),
        ("accepted_hard_violation", "Hard violation"),
        ("net_benefit", "Net benefit"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(5.2, 3.9), dpi=220, sharex=True)
    for ax, (field, title) in zip(axes.ravel(), measures):
        lower = f"{field}_lower"
        upper = f"{field}_upper"
        for method, group in threshold.groupby("method", sort=True):
            group = group.sort_values("threshold")
            color = palette.get(method, "0.3")
            if lower in group and upper in group:
                ax.fill_between(group["threshold"], group[lower], group[upper], color=color, alpha=0.13)
            ax.plot(group["threshold"], group[field], color=color, linewidth=1.1, label=method)
        ax.set_title(title, fontsize=7)
        ax.grid(axis="y", linewidth=0.3, color="0.88")
    axes[1, 0].set_xlabel("Acceptance threshold")
    axes[1, 1].set_xlabel("Acceptance threshold")
    axes[0, 0].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    for suffix in ["png", "display.png", "svg", "pdf"]:
        fig.savefig(FIGURE_DIR / f"extended_data_figure10_swebench_threshold_analysis.{suffix}")
    plt.close(fig)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    gate = json.loads((RESULT_DIR / "gate_status.json").read_text(encoding="utf-8"))
    instance = pd.read_csv(RESULT_DIR / "instance_summary.csv")
    predictor = pd.read_csv(RESULT_DIR / "predictor_summary.csv")

    predictor.to_csv(TABLE_DIR / "table17_swebench_fileloc_predictor_summary.csv", index=False)

    old_model_path = TABLE_DIR / "table17_swebench_fileloc_model_summary.csv"
    old_model = pd.read_csv(old_model_path) if old_model_path.exists() else pd.DataFrame()
    model = (
        instance.groupby("model", as_index=False)
        .agg(
            issue_rows=("instance_id", "size"),
            mean_state_binding_score=("state_binding_score", "mean"),
            mean_implementation_hit_at3=("implementation_hit_at3_rate", "mean"),
            mean_constraint_clean_hit_at3=("constraint_clean_hit_at3_rate", "mean"),
            mean_hard_constraint_violation=("hard_constraint_violation_rate", "mean"),
            mean_vote_margin=("vote_margin", "mean"),
        )
    )
    execution_cols = [
        "model",
        "official_slice_rows",
        "official_resolved",
        "official_unresolved",
        "official_errors",
        "v3_apply_repair_rows",
        "v3_apply_repair_resolved",
        "v3_apply_repair_errors",
    ]
    if not old_model.empty and set(execution_cols).issubset(old_model.columns):
        model = model.merge(old_model[execution_cols], on="model", how="left")
    for col in execution_cols[1:]:
        if col not in model:
            model[col] = 0
    model.to_csv(TABLE_DIR / "table17_swebench_fileloc_model_summary.csv", index=False)

    gate_cols = {
        "status": gate.get("status"),
        "rows": gate.get("rows"),
        "model_instance_rows": gate.get("model_instance_rows"),
        "primary_outcome": gate.get("primary_outcome"),
        "success_positive_rows": gate.get("success_positive_rows"),
        "success_negative_rows": gate.get("success_negative_rows"),
        "baseline_incremental_auc": gate.get("baseline_incremental_auc"),
        "augmented_csb_auc": gate.get("augmented_csb_auc"),
        "delta_auc": gate.get("delta_auc"),
        "delta_auc_ci_low": gate.get("delta_auc_ci_low"),
        "delta_auc_ci_high": gate.get("delta_auc_ci_high"),
        "state_binding_success_auc": gate.get("state_binding_success_auc"),
        "best_simple_baseline_success_auc": gate.get("best_simple_baseline_success_auc"),
        "no_violation_baseline_auc": gate.get("no_violation_baseline_auc"),
        "no_violation_augmented_csb_auc": gate.get("no_violation_augmented_csb_auc"),
        "no_violation_delta_auc": gate.get("no_violation_delta_auc"),
        "constraint_clean_baseline_auc": gate.get("constraint_clean_baseline_auc"),
        "constraint_clean_augmented_csb_auc": gate.get("constraint_clean_augmented_csb_auc"),
        "constraint_clean_delta_auc": gate.get("constraint_clean_delta_auc"),
        "leave_one_repo_positive": gate.get("leave_one_repo_positive"),
        "leave_one_repo_total": gate.get("leave_one_repo_total"),
        "leave_one_model_positive": gate.get("leave_one_model_positive"),
        "leave_one_model_total": gate.get("leave_one_model_total"),
        "within_model_positive": gate.get("within_model_positive"),
        "within_model_total": gate.get("within_model_total"),
        "within_repo_positive": gate.get("within_repo_positive"),
        "within_repo_total": gate.get("within_repo_total"),
        "wrapper_delta_hard_violation": gate.get("wrapper_delta_hard_violation"),
        "wrapper_delta_hit_at3": gate.get("wrapper_delta_hit_at3"),
        "wrapper_status": gate.get("wrapper_status"),
    }
    pd.DataFrame([gate_cols]).to_csv(TABLE_DIR / "table17_swebench_fileloc_gate_summary.csv", index=False)

    pd.DataFrame(
        [
            {
                "condition": "Full non-CSB baseline",
                "outcome": gate.get("primary_outcome"),
                "auc": gate.get("baseline_incremental_auc"),
                "delta_auc": gate.get("delta_auc"),
                "delta_auc_ci_low": gate.get("delta_auc_ci_low"),
                "delta_auc_ci_high": gate.get("delta_auc_ci_high"),
                "issue_records": 300,
                "api_models": int(instance["model"].nunique()),
                "task_rows": gate.get("rows"),
            },
            {
                "condition": "Baseline + CSB",
                "outcome": gate.get("primary_outcome"),
                "auc": gate.get("augmented_csb_auc"),
                "delta_auc": gate.get("delta_auc"),
                "delta_auc_ci_low": gate.get("delta_auc_ci_low"),
                "delta_auc_ci_high": gate.get("delta_auc_ci_high"),
                "issue_records": 300,
                "api_models": int(instance["model"].nunique()),
                "task_rows": gate.get("rows"),
            },
        ]
    ).to_csv(SOURCE_DIR / "figure5c_swebench_incremental_auc.csv", index=False)

    predictor.sort_values("success_auc", ascending=False).to_csv(
        SOURCE_DIR / "figure5d_swebench_single_predictor_auc.csv", index=False
    )
    pd.DataFrame(
        [
            {"item": "leave_one_repository", "value": f"{gate.get('leave_one_repo_positive')}/{gate.get('leave_one_repo_total')}", "note": "positive primary delta-AUC sensitivities", "fill_status": "pass"},
            {"item": "leave_one_model", "value": f"{gate.get('leave_one_model_positive')}/{gate.get('leave_one_model_total')}", "note": "positive primary delta-AUC sensitivities", "fill_status": "pass"},
            {"item": "within_model", "value": f"{gate.get('within_model_positive')}/{gate.get('within_model_total')}", "note": "positive primary delta-AUC slices", "fill_status": "pass"},
            {"item": "within_repository", "value": f"{gate.get('within_repo_positive')}/{gate.get('within_repo_total')}", "note": "positive primary delta-AUC slices", "fill_status": "boundary"},
            {"item": "secondary_no_violation_delta_auc", "value": f"{float(gate.get('no_violation_delta_auc')):+.3f}", "note": "task-only hard-constraint no-violation", "fill_status": "secondary"},
            {"item": "composite_clean_delta_auc", "value": f"{float(gate.get('constraint_clean_delta_auc')):+.3f}", "note": "task-only constraint-clean hit@3", "fill_status": "secondary"},
            {"item": "guard_delta_violation", "value": f"{float(gate.get('wrapper_delta_hard_violation')):+.3f}", "note": "task-only hard violations", "fill_status": "secondary"},
            {"item": "guard_delta_hit_at3", "value": f"{float(gate.get('wrapper_delta_hit_at3')):+.3f}", "note": "primary hit@3 change; wrapper not used as primary evidence", "fill_status": gate.get("wrapper_status")},
        ]
    ).to_csv(SOURCE_DIR / "figure5e_swebench_sensitivity_guard.csv", index=False)

    y = _as_bool(instance["implementation_hit_at3_majority"]).astype(int).to_numpy()
    hard = (pd.to_numeric(instance["hard_constraint_violation_rate"], errors="coerce").fillna(1.0) > 0).astype(int).to_numpy()
    pred = pd.DataFrame(
        {
            "model": instance["model"],
            "repo": instance["repo"],
            "instance_id": instance["instance_id"],
            "hit_at3": y,
            "hard_violation": hard,
            "baseline_probability": _cv_predict(instance, y, BASELINE_COLUMNS),
            "csb_probability": _cv_predict(instance, y, AUGMENTED_COLUMNS),
        }
    ).dropna()
    pred.to_csv(SOURCE_DIR / "swebench_cv_reliability_predictions.csv", index=False)

    thresholds = [round(x, 2) for x in np.arange(0.50, 0.951, 0.05)]
    calibration = _calibration_summary(pred)
    threshold = _threshold_summary(pred, thresholds)
    cal_ci, thr_ci = _bootstrap_bands(pred, thresholds, samples=500, seed=31)
    calibration = calibration.merge(cal_ci, on=["method", "bin"], how="left")
    threshold = threshold.merge(thr_ci, on=["method", "threshold"], how="left")
    calibration.to_csv(SOURCE_DIR / "extended_data_figure9_swebench_reliability_calibration.csv", index=False)
    threshold.to_csv(SOURCE_DIR / "extended_data_figure10_swebench_threshold_analysis.csv", index=False)
    _write_figures(calibration, threshold)


if __name__ == "__main__":
    main()

