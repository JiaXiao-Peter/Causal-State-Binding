from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from freewill.causal_state_sufficiency import read_api_settings, safe_slug, select_api_file_models
from freewill.real_task_predictive_validity import (
    load_default_config,
    run_swebench_file_localization_predictive_validity_v2,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SWE-bench Lite issue-to-file localization v2 with oracle-free CSB prediction gates."
    )
    parser.add_argument("--api-file", default="", help="Local credentials file. The key is read but never written.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpt-3.5-turbo", "gpt-4o-mini", "qwen-turbo", "qwen-plus", "deepseek-chat", "gemini-2.5-flash-lite"],
    )
    parser.add_argument("--models-from-api-file", action="store_true", help="Select model IDs from --api-file without printing or writing the API key.")
    parser.add_argument("--model-count", type=int, default=6, help="Number of API-file models to select when --models-from-api-file is used.")
    parser.add_argument("--instances", type=int, default=300)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--split", default="test")
    parser.add_argument("--self-consistency-repeats", type=int, default=3)
    parser.add_argument("--diagnostic-repeats", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--fetch-repo-tree", action="store_true")
    parser.add_argument(
        "--repo-checkout-root",
        default="",
        help="Optional directory containing local git checkouts as owner/repo, owner__repo, or repo-name for fixed retrieval.",
    )
    parser.add_argument("--include-oracle-diagnostic", action="store_true")
    parser.add_argument("--no-wrapper-arms", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=400)
    parser.add_argument("--delta-auc-gate", type=float, default=0.03)
    parser.add_argument("--noninferiority-margin", type=float, default=-0.03)
    parser.add_argument("--execution-slice-size", type=int, default=0, help="Optionally freeze 30-50 instances for later official SWE-bench patch/test execution.")
    parser.add_argument("--hard-call-cap", type=int, default=20000)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--resume", action="store_true", help="Resume from traces_checkpoint.jsonl in the output directory.")
    parser.add_argument("--shard-index", type=int, default=0, help="Issue-shard index for splitting large runs.")
    parser.add_argument("--shard-count", type=int, default=1, help="Total number of issue shards.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-root", default="results/paper1_revision/real_task_predictive_validity")
    args = parser.parse_args()

    api_path = Path(args.api_file) if args.api_file else None
    api_settings = read_api_settings(api_path)
    selected_models = list(args.models)
    if args.models_from_api_file:
        selected_models = select_api_file_models(api_path, count=args.model_count)
        if not selected_models:
            raise RuntimeError("--models-from-api-file was set, but no model IDs were found in --api-file")
    config = load_default_config()
    model_slug = "_".join(safe_slug(model) for model in selected_models)
    run_id = args.run_id or f"swebench_fileloc_v2_{model_slug}_i{args.instances}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(args.output_root) / run_id
    result = run_swebench_file_localization_predictive_validity_v2(
        config=config,
        api_settings=api_settings,
        output_dir=output_dir,
        models=selected_models,
        instances=args.instances,
        offset=args.offset,
        split=args.split,
        self_consistency_repeats=args.self_consistency_repeats,
        diagnostic_repeats=args.diagnostic_repeats,
        top_k=args.top_k,
        fetch_repo_tree=bool(args.fetch_repo_tree),
        repo_checkout_root=Path(args.repo_checkout_root) if args.repo_checkout_root else None,
        include_oracle_diagnostic=bool(args.include_oracle_diagnostic),
        include_wrapper_arms=not bool(args.no_wrapper_arms),
        bootstrap_samples=args.bootstrap_samples,
        delta_auc_gate=args.delta_auc_gate,
        noninferiority_margin=args.noninferiority_margin,
        execution_slice_size=args.execution_slice_size,
        hard_call_cap=args.hard_call_cap,
        max_workers=args.max_workers,
        resume=bool(args.resume),
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        dry_run=bool(args.dry_run),
    )
    gate = result["gate_status"]
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "status": gate.get("status"),
                "models": selected_models,
                "planned_remote_calls": gate.get("planned_remote_calls"),
                "shard_index": gate.get("shard_index"),
                "shard_count": gate.get("shard_count"),
                "shard_issue_rows": gate.get("shard_issue_rows"),
                "observed_rows": gate.get("observed_rows", 0),
                "checkpoint_tasks_skipped": gate.get("checkpoint_tasks_skipped", 0),
                "delta_auc": gate.get("delta_auc"),
                "delta_auc_ci_low": gate.get("delta_auc_ci_low"),
                "wrapper_status": gate.get("wrapper_status"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

