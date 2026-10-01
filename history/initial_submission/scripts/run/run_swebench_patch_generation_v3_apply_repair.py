from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from freewill.causal_state_sufficiency import read_api_settings, safe_slug
from freewill.real_task_predictive_validity import (
    _coerce_file_list,
    _make_provider,
    _metadata_subset,
    changed_files,
    collect_repo_file_contexts,
    fetch_swe_bench_lite_rows,
    implementation_files,
    includes_test_file,
    load_default_config,
    normalize_repo_file_path,
    oracle_free_retrieval_candidates,
    read_swebench_execution_slice_manifest,
    resolve_local_repo_checkout,
)
from freewill.utils import ensure_dir, write_json, write_parquet


DEFAULT_SELECTED_MODELS = ["gpt-4o-mini", "qwen-plus-latest"]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _json_or_empty(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    text = str(value or "").strip()
    if not text:
        return []
    try:
        return json.loads(text)
    except Exception:
        return []


def normalize_patch_text(text: str) -> str:
    patch = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    fence = re.search(r"```(?:diff|patch)?\s*(.*?)```", patch, flags=re.IGNORECASE | re.DOTALL)
    if fence:
        patch = fence.group(1).strip()
    start_positions = [pos for token in ["diff --git ", "--- a/", "--- "] if (pos := patch.find(token)) >= 0]
    if start_positions:
        patch = patch[min(start_positions) :].strip()
    patch = re.sub(r"\n{3,}", "\n\n", patch)
    if patch and not patch.endswith("\n"):
        patch += "\n"
    return patch


def _run_git(args: list[str], *, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)


def check_patch_applies(
    *,
    repo: str,
    base_commit: str,
    patch_text: str,
    checkout_root: Path,
    work_root: Path,
    timeout: int = 180,
) -> dict[str, Any]:
    patch = normalize_patch_text(patch_text)
    if not patch.strip():
        return {"apply_ok": False, "apply_error": "empty_patch", "normalized_patch": patch}
    repo_dir = resolve_local_repo_checkout(repo, checkout_root)
    if repo_dir is None:
        return {"apply_ok": False, "apply_error": f"missing_checkout:{repo}", "normalized_patch": patch}
    if not base_commit:
        return {"apply_ok": False, "apply_error": "missing_base_commit", "normalized_patch": patch}
    ensure_dir(work_root)
    with tempfile.TemporaryDirectory(prefix="applycheck_", dir=str(work_root)) as tmp:
        worktree = Path(tmp) / "repo"
        patch_file = Path(tmp) / "candidate.patch"
        patch_file.write_text(patch, encoding="utf-8")
        add = _run_git(
            ["git", "-C", str(repo_dir), "worktree", "add", "--force", "--detach", str(worktree), base_commit],
            timeout=timeout,
        )
        if add.returncode != 0:
            return {
                "apply_ok": False,
                "apply_error": (add.stderr or add.stdout or "worktree_add_failed")[-4000:],
                "normalized_patch": patch,
            }
        try:
            check = _run_git(
                ["git", "-C", str(worktree), "apply", "--check", "--whitespace=nowarn", str(patch_file)],
                timeout=timeout,
            )
            if check.returncode == 0:
                return {"apply_ok": True, "apply_error": "", "normalized_patch": patch}
            return {
                "apply_ok": False,
                "apply_error": (check.stderr or check.stdout or "git_apply_check_failed")[-4000:],
                "normalized_patch": patch,
            }
        finally:
            _run_git(["git", "-C", str(repo_dir), "worktree", "remove", "--force", str(worktree)], timeout=timeout)
            if worktree.exists():
                shutil.rmtree(worktree, ignore_errors=True)


def build_repair_prompt(
    *,
    row: Any,
    model: str,
    failed_patch: str,
    apply_error: str,
    candidate_files: list[str],
    file_contexts: list[dict[str, Any]],
) -> str:
    fail_tests = ", ".join(str(item) for item in row.fail_to_pass[:10]) if isinstance(row.fail_to_pass, list) else str(row.fail_to_pass)[:1200]
    pass_tests = ", ".join(str(item) for item in row.pass_to_pass[:6]) if isinstance(row.pass_to_pass, list) else str(row.pass_to_pass)[:800]
    contexts = []
    for context in file_contexts:
        path = normalize_repo_file_path(row.repo, str(context.get("path", "")))
        content = str(context.get("content", ""))
        if path and content:
            contexts.append(f"File: {path}\n```\n{content[:12000]}\n```")
    context_block = "\n\n".join(contexts) if contexts else "No source file content was available."
    candidates = "\n".join(f"- {path}" for path in candidate_files[:30]) or "- no candidate files"
    return (
        "Return exactly one compact JSON object and nothing else. No markdown.\n"
        "Required keys: model_patch, target_files, rationale, confidence, defer.\n"
        "Task: repair the previous unified-diff patch so it applies cleanly with git apply --check "
        "against the listed base commit. Use only issue-visible and repository-visible information below. "
        "Do not use benchmark reference patches, do not edit tests, and do not invent files outside the repository.\n"
        f"Model being repaired: {model}\n"
        f"Instance: {row.instance_id}\n"
        f"Repository: {row.repo}\n"
        f"Base commit: {row.base_commit}\n"
        f"Problem statement:\n{str(row.problem_statement)[:7000]}\n"
        f"FAIL_TO_PASS tests: {fail_tests or 'not provided'}\n"
        f"PASS_TO_PASS examples: {pass_tests or 'not provided'}\n"
        f"Candidate implementation files:\n{candidates}\n"
        f"git apply --check error:\n{apply_error[-3000:]}\n"
        f"Previous patch:\n```diff\n{failed_patch[:9000]}\n```\n"
        f"Repository-visible source context:\n{context_block}\n"
        "If you cannot produce an applicable implementation-only unified diff, set model_patch to an empty string, "
        "target_files to [], confidence to 0, and defer to true."
    )


def _write_predictions(
    rows: pd.DataFrame,
    output_dir: Path,
    *,
    models: list[str],
    only_applicable: bool,
) -> pd.DataFrame:
    subdir = "predictions_by_model_applicable" if only_applicable else "predictions_by_model_all"
    id_dir = ensure_dir(output_dir / "instance_ids_by_model_applicable")
    pred_dir = ensure_dir(output_dir / subdir)
    manifest_rows = []
    for model in models:
        model_rows = rows[rows["model"].astype(str).eq(model)].copy()
        if only_applicable:
            model_rows = model_rows[model_rows["final_apply_ok"].astype(bool)]
        model_rows = model_rows.sort_values("instance_id").drop_duplicates("instance_id", keep="last")
        pred_path = pred_dir / f"{safe_slug(model)}.jsonl"
        lines = [
            json.dumps(
                {
                    "instance_id": str(record["instance_id"]),
                    "model_name_or_path": model,
                    "model_patch": str(record.get("final_patch", "")),
                },
                ensure_ascii=False,
            )
            for record in model_rows.to_dict(orient="records")
        ]
        pred_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        id_path = id_dir / f"{safe_slug(model)}.txt"
        id_path.write_text("\n".join(str(item) for item in model_rows["instance_id"].astype(str).tolist()) + ("\n" if len(model_rows) else ""), encoding="utf-8")
        manifest_rows.append(
            {
                "model": model,
                "predictions_path": str(pred_path),
                "instance_ids_path": str(id_path),
                "rows": int(len(model_rows)),
                "only_applicable": bool(only_applicable),
            }
        )
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(output_dir / f"{subdir}_manifest.csv", index=False)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SWE-bench Lite patch-generation v3: local git apply-check, API repair, and filtered predictions."
    )
    parser.add_argument("--input-predictions-dir", required=True, help="Directory containing v2 predictions_by_model/*.jsonl")
    parser.add_argument("--slice-manifest", required=True)
    parser.add_argument("--api-file", default="")
    parser.add_argument("--models", nargs="+", default=DEFAULT_SELECTED_MODELS)
    parser.add_argument("--split", default="test")
    parser.add_argument("--repo-checkout-root", required=True)
    parser.add_argument(
        "--work-root",
        default="",
        help="Short temporary directory for git worktrees. Defaults to the system temp directory to avoid Windows path-length failures.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-repair-attempts", type=int, default=1)
    parser.add_argument("--max-context-files", type=int, default=4)
    parser.add_argument("--max-context-chars-per-file", type=int, default=12000)
    parser.add_argument("--hard-call-cap", type=int, default=150)
    parser.add_argument("--resume", action="store_true", help="Resume from patch_generation_v3_checkpoint.jsonl.")
    parser.add_argument("--dry-run", action="store_true", help="Do apply-check only; do not call remote repair.")
    args = parser.parse_args()

    output_dir = ensure_dir(Path(args.output_dir))
    checkout_root = Path(args.repo_checkout_root)
    work_root = ensure_dir(
        Path(args.work_root)
        if args.work_root
        else Path(tempfile.gettempdir()) / "swebench_v3_apply_worktrees"
    )
    selected_models = [str(model).strip() for model in args.models if str(model).strip()]
    if not selected_models:
        raise ValueError("at least one model is required")

    slice_frame = read_swebench_execution_slice_manifest(Path(args.slice_manifest))
    instance_ids = slice_frame["instance_id"].astype(str).tolist()
    rows = fetch_swe_bench_lite_rows(split=args.split, limit=300, offset=0)
    row_map = {row.instance_id: row for row in rows}
    missing = [instance_id for instance_id in instance_ids if instance_id not in row_map]
    if missing:
        raise RuntimeError(f"slice instances missing from dataset: {missing[:5]}")

    input_dir = Path(args.input_predictions_dir)
    records_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for model in selected_models:
        pred_path = input_dir / "predictions_by_model" / f"{safe_slug(model)}.jsonl"
        if not pred_path.exists():
            raise FileNotFoundError(f"missing predictions for {model}: {pred_path}")
        for record in _read_jsonl(pred_path):
            records_by_key[(model, str(record.get("instance_id", "")))] = record

    manifest = {
        "experiment": "swebench_lite_execution_slice_patch_generation_v3_apply_repair",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_predictions_dir": str(input_dir),
        "slice_manifest": str(args.slice_manifest),
        "models": selected_models,
        "split": args.split,
        "repo_checkout_root": str(checkout_root),
        "max_repair_attempts": int(args.max_repair_attempts),
        "dry_run": bool(args.dry_run),
        "oracle_free_repair_prompt": True,
        "reference_patch_used_in_prompt": False,
    }
    write_json(output_dir / "run_manifest.json", manifest)

    api_settings = None
    providers = {}
    if not args.dry_run and args.max_repair_attempts > 0:
        api_settings = read_api_settings(Path(args.api_file) if args.api_file else None)
        os.environ["OPENAI_API_KEY"] = api_settings.api_key
        config = load_default_config()
        providers = {
            model: _make_provider(config, model=model, base_url=api_settings.base_url)
            for model in selected_models
        }

    checkpoint_path = output_dir / "patch_generation_v3_checkpoint.jsonl"
    existing_rows = _read_jsonl(checkpoint_path) if checkpoint_path.exists() else []
    if existing_rows and not args.resume:
        raise RuntimeError(f"checkpoint exists at {checkpoint_path}; use --resume or a new output directory")
    completed_keys = {
        (str(record.get("model", "")), str(record.get("instance_id", "")))
        for record in existing_rows
        if record.get("model") and record.get("instance_id")
    }
    remote_calls = int(sum(int(record.get("repair_attempts", 0) or 0) for record in existing_rows))
    output_rows: list[dict[str, Any]] = list(existing_rows)
    for model in selected_models:
        for instance_id in instance_ids:
            if (model, instance_id) in completed_keys:
                continue
            row = row_map[instance_id]
            source_record = records_by_key.get((model, instance_id), {})
            initial_patch = normalize_patch_text(str(source_record.get("model_patch", "")))
            current_patch = initial_patch
            initial_check = check_patch_applies(
                repo=row.repo,
                base_commit=row.base_commit,
                patch_text=current_patch,
                checkout_root=checkout_root,
                work_root=work_root,
            )
            final_check = dict(initial_check)
            repair_attempts = 0
            repair_error = ""
            repair_metadata: dict[str, Any] = {}
            if not initial_check["apply_ok"] and not args.dry_run and args.max_repair_attempts > 0 and remote_calls < args.hard_call_cap:
                retrieved = oracle_free_retrieval_candidates(row, repo_tree_files=[], top_k=args.top_k)
                patch_files = implementation_files(changed_files(current_patch))
                candidates = list(dict.fromkeys([*patch_files, *retrieved]))
                contexts = collect_repo_file_contexts(
                    row,
                    candidates,
                    max_files=args.max_context_files,
                    max_chars_per_file=args.max_context_chars_per_file,
                    max_candidate_attempts=max(20, args.top_k),
                    checkout_root=checkout_root,
                )
                for attempt in range(args.max_repair_attempts):
                    if remote_calls >= args.hard_call_cap:
                        repair_error = "hard_call_cap_reached"
                        break
                    defaults = {"model_patch": "", "target_files": [], "rationale": "repair fallback", "confidence": 0, "defer": True}
                    prompt = build_repair_prompt(
                        row=row,
                        model=model,
                        failed_patch=current_patch,
                        apply_error=str(final_check.get("apply_error", "")),
                        candidate_files=candidates,
                        file_contexts=contexts,
                    )
                    try:
                        result = providers[model].chat_json_with_metadata(
                            "You are a SWE-bench patch repair agent. Output valid compact JSON only.",
                            prompt,
                            defaults,
                            temperature_override=0.1,
                            top_p_override=1.0,
                        )
                        remote_calls += 1
                        repair_attempts += 1
                        payload = result.payload if isinstance(result.payload, dict) else defaults
                        repair_metadata = _metadata_subset(result.metadata)
                        repaired_patch = normalize_patch_text(str(payload.get("model_patch", "") or ""))
                        target_files = _coerce_file_list(payload.get("target_files", []))
                        if includes_test_file(target_files) or includes_test_file(changed_files(repaired_patch)):
                            final_check = {
                                "apply_ok": False,
                                "apply_error": "repair_edits_test_files",
                                "normalized_patch": repaired_patch,
                            }
                            current_patch = repaired_patch
                            continue
                        current_patch = repaired_patch
                        final_check = check_patch_applies(
                            repo=row.repo,
                            base_commit=row.base_commit,
                            patch_text=current_patch,
                            checkout_root=checkout_root,
                            work_root=work_root,
                        )
                        if final_check["apply_ok"]:
                            break
                    except Exception as exc:
                        repair_error = f"{type(exc).__name__}: {exc}"
                        break

            final_patch = normalize_patch_text(str(final_check.get("normalized_patch", current_patch)))
            edits_tests = includes_test_file(changed_files(final_patch))
            final_apply_ok = bool(final_check.get("apply_ok", False)) and not edits_tests
            out_record = {
                "model": model,
                "instance_id": instance_id,
                "repo": row.repo,
                "base_commit": row.base_commit,
                "initial_patch_chars": int(len(initial_patch)),
                "initial_apply_ok": bool(initial_check.get("apply_ok", False)),
                "initial_apply_error": str(initial_check.get("apply_error", ""))[:4000],
                "repair_attempts": int(repair_attempts),
                "repair_error": repair_error,
                "final_patch": final_patch if final_apply_ok else "",
                "final_patch_chars": int(len(final_patch if final_apply_ok else "")),
                "final_apply_ok": bool(final_apply_ok),
                "final_apply_error": "" if final_apply_ok else str(final_check.get("apply_error", ""))[:4000],
                "edits_tests": bool(edits_tests),
                **{f"repair_{key}": value for key, value in repair_metadata.items()},
            }
            output_rows.append(out_record)
            with checkpoint_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(out_record, ensure_ascii=False) + "\n")

    frame = pd.DataFrame(output_rows)
    frame.to_csv(output_dir / "patch_generation_v3_apply_repair_traces.csv", index=False)
    write_parquet(frame, output_dir / "patch_generation_v3_apply_repair_traces.parquet")
    applicable_manifest = _write_predictions(frame, output_dir, models=selected_models, only_applicable=True)
    all_manifest = _write_predictions(frame, output_dir, models=selected_models, only_applicable=False)
    by_model = (
        frame.groupby("model")
        .agg(
            rows=("instance_id", "count"),
            initial_apply_ok=("initial_apply_ok", "sum"),
            final_apply_ok=("final_apply_ok", "sum"),
            repair_attempts=("repair_attempts", "sum"),
            filtered_rows=("final_apply_ok", lambda s: int((~s.astype(bool)).sum())),
        )
        .reset_index()
    )
    by_model.to_csv(output_dir / "patch_generation_v3_model_summary.csv", index=False)
    gate = {
        **manifest,
        "status": "ready_for_official_harness"
        if int(frame["final_apply_ok"].astype(bool).sum()) > 0
        else "needs_review",
        "rows": int(len(frame)),
        "initial_apply_ok_rows": int(frame["initial_apply_ok"].astype(bool).sum()),
        "final_apply_ok_rows": int(frame["final_apply_ok"].astype(bool).sum()),
        "filtered_non_applicable_rows": int((~frame["final_apply_ok"].astype(bool)).sum()),
        "remote_repair_calls": int(remote_calls),
        "hard_call_cap": int(args.hard_call_cap),
        "applicable_predictions_manifest": str(output_dir / "predictions_by_model_applicable_manifest.csv"),
        "all_predictions_manifest": str(output_dir / "predictions_by_model_all_manifest.csv"),
        "scope_note": "v3 filters official harness inputs to patches that pass local git apply --check at the benchmark base commit.",
    }
    write_json(output_dir / "gate_status.json", gate)
    report = [
        "# SWE-bench Patch Generation v3 Apply-Repair",
        "",
        f"- Status: `{gate['status']}`",
        f"- Rows: `{gate['rows']}`",
        f"- Initial apply-ok rows: `{gate['initial_apply_ok_rows']}`",
        f"- Final apply-ok rows: `{gate['final_apply_ok_rows']}`",
        f"- Remote repair calls: `{gate['remote_repair_calls']}`",
        "",
        "| model | rows | initial apply-ok | final apply-ok | repair calls | filtered |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for record in by_model.to_dict(orient="records"):
        report.append(
            f"| {record['model']} | {record['rows']} | {record['initial_apply_ok']} | "
            f"{record['final_apply_ok']} | {record['repair_attempts']} | {record['filtered_rows']} |"
        )
    (output_dir / "patch_generation_v3_apply_repair_report.md").write_text("\n".join(report).rstrip() + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "status": gate["status"],
                "initial_apply_ok_rows": gate["initial_apply_ok_rows"],
                "final_apply_ok_rows": gate["final_apply_ok_rows"],
                "remote_repair_calls": gate["remote_repair_calls"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

