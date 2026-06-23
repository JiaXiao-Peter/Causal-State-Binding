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
    run_swebench_execution_slice_patch_generation_v2,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate or dry-run SWE-bench Lite execution-slice model_patch predictions. "
            "The output JSONL files are inputs for the official SWE-bench harness, not execution results."
        )
    )
    parser.add_argument("--slice-manifest", required=True, help="official_execution_slice_instances.csv or .jsonl")
    parser.add_argument("--api-file", default="", help="Local credentials file. The key is read but never written.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpt-3.5-turbo", "gpt-4o-mini", "qwen-turbo", "qwen-plus", "deepseek-chat", "gemini-2.5-flash-lite"],
    )
    parser.add_argument("--models-from-api-file", action="store_true", help="Select model IDs from --api-file without printing or writing the API key.")
    parser.add_argument("--model-count", type=int, default=6, help="Number of API-file models to select when --models-from-api-file is used.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--fetch-repo-tree", action="store_true")
    parser.add_argument(
        "--include-file-context",
        action="store_true",
        help="Fetch repository-visible source text for top candidate files at each base commit.",
    )
    parser.add_argument("--max-context-files", type=int, default=3)
    parser.add_argument("--max-context-chars-per-file", type=int, default=6000)
    parser.add_argument("--max-context-candidate-attempts", type=int, default=50)
    parser.add_argument(
        "--repo-checkout-root",
        default="",
        help="Optional directory containing local git checkouts as owner/repo, owner__repo, or repo-name for tree and file-context fetches.",
    )
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--hard-call-cap", type=int, default=500)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--resume", action="store_true", help="Resume from patch_generation_checkpoint.jsonl in the output directory.")
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

    model_slug = "_".join(safe_slug(model) for model in selected_models)
    run_id = args.run_id or f"swebench_execution_slice_patchgen_{model_slug}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(args.output_root) / run_id
    result = run_swebench_execution_slice_patch_generation_v2(
        config=load_default_config(),
        api_settings=api_settings,
        slice_manifest=Path(args.slice_manifest),
        output_dir=output_dir,
        models=selected_models,
        split=args.split,
        top_k=args.top_k,
        fetch_repo_tree=bool(args.fetch_repo_tree),
        include_file_context=bool(args.include_file_context),
        max_context_files=args.max_context_files,
        max_context_chars_per_file=args.max_context_chars_per_file,
        max_context_candidate_attempts=args.max_context_candidate_attempts,
        repo_checkout_root=Path(args.repo_checkout_root) if args.repo_checkout_root else None,
        max_workers=args.max_workers,
        hard_call_cap=args.hard_call_cap,
        resume=bool(args.resume),
        dry_run=bool(args.dry_run),
    )
    gate = result["gate_status"]
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "status": gate.get("status"),
                "models": selected_models,
                "slice_instance_count": gate.get("slice_instance_count"),
                "planned_remote_calls": gate.get("planned_remote_calls"),
                "observed_rows": gate.get("observed_rows", 0),
                "prediction_files_status": gate.get("prediction_files_status"),
                "file_context_status": gate.get("file_context_status"),
                "scope_note": gate.get("scope_note", "patch-generation only; official execution still required"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

