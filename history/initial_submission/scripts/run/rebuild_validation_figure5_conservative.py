from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "manuscript/figures/source_data"
MAIN_DIR = ROOT / "manuscript/figures/main"
SUBMISSION_MAIN_DIR = ROOT / "submission/ai_open/_build/source/figures/main"

BLUE = "#3B7EAE"
GOLD = "#D8A83F"
GREY = "#C9C9C9"
DARK = "#2D2D2D"


LABELS_D = {
    "state_binding_score": "CSB diagnostic",
    "task_action_entropy": "Action entropy",
    "file_set_entropy": "File-set entropy",
    "top_file_set_agreement": "Top set agreement",
    "vote_margin": "Vote margin",
    "mean_confidence": "Confidence",
    "issue_length": "Issue length",
    "retrieved_candidate_count": "Candidate count",
    "mean_rationale_chars": "Rationale length",
    "path_count_entropy": "Path-count entropy",
}

LABELS_E = {
    "leave_one_repository": "Leave-one\nrepository",
    "leave_one_model": "Leave-one\nmodel",
    "within_model": "Within\nmodel",
    "within_repository": "Within\nrepository",
    "secondary_no_violation_delta_auc": "No-violation\nDelta AUC",
    "composite_clean_delta_auc": "Clean hit@3\nDelta AUC",
    "guard_delta_violation": "Guard delta\nviolation",
    "guard_delta_hit_at3": "Guard delta\nhit@3",
}


def _style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color="#E8E8E8", linewidth=0.8)
    ax.set_axisbelow(True)


def _panel_heading(ax, letter: str, title: str, subtitle: str, letter_x: float = -0.27) -> None:
    ax.text(letter_x, 1.17, letter, transform=ax.transAxes, fontsize=13, fontweight="bold")
    ax.text(0.0, 1.13, title, transform=ax.transAxes, fontsize=11, fontweight="bold")
    ax.text(0.0, 1.055, subtitle, transform=ax.transAxes, fontsize=8, color="#666")


def build() -> None:
    MAIN_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_MAIN_DIR.mkdir(parents=True, exist_ok=True)

    a = pd.read_csv(SOURCE / "figure5a_open_weight_core_controls.csv")
    b = pd.read_csv(SOURCE / "figure5b_open_weight_state_boundary_lesion.csv")
    c = pd.read_csv(SOURCE / "figure5c_swebench_incremental_auc.csv")
    d = pd.read_csv(SOURCE / "figure5d_swebench_single_predictor_auc.csv")
    e = pd.read_csv(SOURCE / "figure5e_swebench_sensitivity_guard.csv")

    fig = plt.figure(figsize=(13.8, 11.0), dpi=240)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.04, 0.76], hspace=0.66, wspace=0.42)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[2, :])

    # Panel a
    order_a = ["distribution matched", "scrambled context", "no-fields"]
    a = a.set_index("control").loc[order_a].reset_index()
    y = np.arange(len(a))
    ax_a.barh(y, a["mean_delta"], color=BLUE, height=0.55)
    ax_a.set_yticks(y, ["Distribution matched", "Scrambled context", "No fields"])
    ax_a.set_xlim(0, 1.02)
    ax_a.set_xlabel("Mean structured-control accuracy delta")
    _panel_heading(
        ax_a,
        "a",
        "Open-weight finite-action controls",
        "Qwen2.5 sizes plus Mistral; 20 task families and three scaffolds",
        letter_x=-0.31,
    )
    for idx, row in a.iterrows():
        text = f"{int(row.positive_cells)}/{int(row.total_cells)} | low95 {row.lowest_cluster_low95:.3f}"
        ax_a.text(min(row.mean_delta + 0.035, 0.80), idx, text, va="center", fontsize=8, fontweight="bold", color=DARK)
    _style_axis(ax_a)

    # Panel b
    order_b = [
        "strict target lesion, Qwen2.5-32B",
        "ordinary target lesion, Mistral-7B",
        "ordinary target lesion, 4 open-weight models",
    ]
    b = b.set_index("comparison").loc[order_b].reset_index()
    y = np.arange(len(b))
    colors = [BLUE if str(s).lower() == "pass" else GOLD for s in b["status"]]
    ax_b.barh(y, b["mean_delta"], color=colors, height=0.55)
    ax_b.set_yticks(y, ["Strict lesion\nQwen2.5-32B", "Ordinary lesion\nMistral-7B", "Ordinary lesion\n4 models"])
    ax_b.set_xlim(0, 1.08)
    ax_b.set_xlabel("Mean structured-control accuracy delta")
    _panel_heading(
        ax_b,
        "b",
        "Open-weight state-boundary lesion",
        "Strict lesion identifies an operational information boundary",
        letter_x=-0.29,
    )
    for idx, row in b.iterrows():
        label = "PASS" if str(row.status).lower() == "pass" else "BOUNDARY"
        x = row.mean_delta - 0.3 if row.mean_delta > 0.7 else row.mean_delta + 0.035
        color = "white" if row.mean_delta > 0.7 else DARK
        ax_b.text(x, idx, f"{row.mean_delta:.3f} | {int(row.positive_cells)}/{int(row.total_cells)} | {label}", va="center", fontsize=8, fontweight="bold", color=color)
    _style_axis(ax_b)

    # Panel c
    c = c.set_index("condition").loc[["Full non-CSB baseline", "Baseline + CSB"]].reset_index()
    x = np.arange(len(c))
    ax_c.bar(x, c["auc"], color=[GOLD, BLUE], width=0.58)
    ax_c.set_xticks(x, ["Full non-CSB\nbaseline", "Baseline +\nCSB"])
    ax_c.set_ylim(0.68, 0.96)
    ax_c.set_ylabel("Task-only hit@3 AUC")
    _panel_heading(
        ax_c,
        "c",
        "300-record issue-to-file prediction",
        "Gold-free baseline; primary outcome uses raw task-only calls",
        letter_x=-0.31,
    )
    for idx, row in c.iterrows():
        ax_c.text(idx, row.auc + 0.006, f"{row.auc:.3f}", ha="center", fontsize=9, fontweight="bold", color=DARK)
    delta = float(c["delta_auc"].iloc[0])
    low = float(c["delta_auc_ci_low"].iloc[0])
    ax_c.plot([0, 1], [0.94, 0.94], color=DARK, linewidth=1.1)
    ax_c.text(0.5, 0.946, f"Delta AUC {delta:.3f}; issue-cluster low95 {low:.3f}", ha="center", fontsize=8, fontweight="bold", color=DARK)
    ax_c.grid(axis="y", color="#E8E8E8", linewidth=0.8)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)

    # Panel d
    d = d.sort_values("success_auc", ascending=True).tail(7)
    y = np.arange(len(d))
    colors = [BLUE if p == "state_binding_score" else GREY for p in d["predictor"]]
    ax_d.barh(y, d["success_auc"], color=colors, height=0.58)
    ax_d.set_yticks(y, [LABELS_D.get(p, p) for p in d["predictor"]])
    ax_d.set_xlim(0.45, 0.94)
    ax_d.set_xlabel("AUC")
    _panel_heading(
        ax_d,
        "d",
        "Single-predictor baselines",
        "AUC for task-only implementation-file hit@3",
        letter_x=-0.19,
    )
    for idx, row in d.iterrows():
        ax_d.text(row.success_auc + 0.008, np.where(d.index == idx)[0][0], f"{row.success_auc:.3f}", va="center", fontsize=8, fontweight="bold", color=DARK)
    _style_axis(ax_d)

    # Panel e
    ax_e.axis("off")
    ax_e.text(0.0, 1.05, "e", transform=ax_e.transAxes, fontsize=13, fontweight="bold")
    ax_e.text(0.055, 1.05, "Issue-cluster sensitivity and secondary guardrails", transform=ax_e.transAxes, fontsize=11, fontweight="bold")
    ax_e.text(0.055, 0.95, "Primary robustness plus secondary no-violation, clean-hit and wrapper diagnostics", transform=ax_e.transAxes, fontsize=8, color="#666")
    left = 0.055
    gap_x = 0.035
    tile_w = 0.205
    tile_h = 0.28
    y_rows = [0.56, 0.18]
    for i, row in e.reset_index(drop=True).iterrows():
        col = i % 4
        row_idx = i // 4
        x0 = left + col * (tile_w + gap_x)
        y0 = y_rows[row_idx]
        status = str(row["fill_status"]).lower()
        color = BLUE if status == "pass" else GOLD if status in {"boundary", "secondary"} else "#8B8B8B"
        rect = plt.Rectangle((x0, y0), tile_w, tile_h, transform=ax_e.transAxes, color=color, clip_on=False)
        ax_e.add_patch(rect)
        label = LABELS_E.get(row["item"], row["item"])
        ax_e.text(x0 + tile_w / 2, y0 + 0.17, str(row["value"]), transform=ax_e.transAxes, ha="center", va="center", color="white", fontsize=13, fontweight="bold")
        ax_e.text(x0 + tile_w / 2, y0 + 0.075, label, transform=ax_e.transAxes, ha="center", va="center", color="white", fontsize=8)
    ax_e.text(
        0.055,
        0.045,
        "Blue cells are primary positive sensitivities; gold cells are boundary or secondary analyses; grey marks wrapper diagnostics not used as primary evidence.",
        transform=ax_e.transAxes,
        fontsize=7,
        color="#666",
    )

    for ext in ["png", "display.png", "svg", "pdf"]:
        path = MAIN_DIR / f"figure5_robustness_validation.{ext}"
        fig.savefig(path, bbox_inches="tight")
        sub_path = SUBMISSION_MAIN_DIR / f"figure5_robustness_validation.{ext}"
        fig.savefig(sub_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    build()

