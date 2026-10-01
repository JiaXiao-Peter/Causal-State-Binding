from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from freewill.real_task_predictive_validity import (
    SWE_BENCH_DATASET,
    fetch_local_git_tree_files,
    fetch_swe_bench_lite_rows,
    resolve_local_repo_checkout,
)


def _commit_exists(repo_dir: Path, commit: str) -> bool:
    if not commit:
        return False
    completed = subprocess.run(
        ["git", "-C", str(repo_dir), "cat-file", "-e", f"{commit}^{{commit}}"],
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit local SWE-bench Lite repo checkouts for base-commit tree retrieval.")
    parser.add_argument("--checkout-root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--output-dir", default="results/paper1_revision/real_task_predictive_validity/swebench_repo_checkout_audit")
    args = parser.parse_args()

    checkout_root = Path(args.checkout_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = fetch_swe_bench_lite_rows(split=args.split, limit=args.limit, offset=0)
    unique_repos = sorted({row.repo for row in rows})
    repo_rows = []
    for repo in unique_repos:
        repo_dir = resolve_local_repo_checkout(repo, checkout_root)
        repo_rows.append(
            {
                "repo": repo,
                "checkout_path": str(repo_dir or ""),
                "checkout_exists": bool(repo_dir is not None),
            }
        )

    commit_rows = []
    for row in rows:
        repo_dir = resolve_local_repo_checkout(row.repo, checkout_root)
        commit_ok = bool(repo_dir is not None and _commit_exists(repo_dir, row.base_commit))
        tree_count = 0
        tree_error = ""
        if commit_ok:
            try:
                tree_count = len(fetch_local_git_tree_files(row.repo, row.base_commit, checkout_root))
            except Exception as exc:
                tree_error = type(exc).__name__
        commit_rows.append(
            {
                "repo": row.repo,
                "instance_id": row.instance_id,
                "base_commit": row.base_commit,
                "checkout_exists": bool(repo_dir is not None),
                "commit_exists": bool(commit_ok),
                "tree_file_count": int(tree_count),
                "tree_ok": bool(tree_count > 0),
                "tree_error": tree_error,
            }
        )

    repo_count = len(unique_repos)
    checkout_repos = sum(1 for row in repo_rows if row["checkout_exists"])
    commit_ok_rows = sum(1 for row in commit_rows if row["commit_exists"])
    tree_ok_rows = sum(1 for row in commit_rows if row["tree_ok"])
    gate = {
        "status": "pass" if checkout_repos == repo_count and commit_ok_rows == len(commit_rows) and tree_ok_rows == len(commit_rows) else "needs_review",
        "dataset": SWE_BENCH_DATASET,
        "split": args.split,
        "limit": int(args.limit),
        "checkout_root": str(checkout_root),
        "repo_count": int(repo_count),
        "checkout_repo_count": int(checkout_repos),
        "instance_rows": int(len(commit_rows)),
        "commit_ok_rows": int(commit_ok_rows),
        "tree_ok_rows": int(tree_ok_rows),
        "missing_repos": [row["repo"] for row in repo_rows if not row["checkout_exists"]],
        "missing_or_bad_instances": [
            {
                "repo": row["repo"],
                "instance_id": row["instance_id"],
                "base_commit": row["base_commit"],
                "commit_exists": row["commit_exists"],
                "tree_ok": row["tree_ok"],
                "tree_error": row["tree_error"],
            }
            for row in commit_rows
            if not row["commit_exists"] or not row["tree_ok"]
        ][:50],
        "scope_note": "Audits local git tree availability for oracle-free retrieval/source-context prompts; it does not run model calls.",
    }
    (output_dir / "repo_checkout_rows.json").write_text(json.dumps(repo_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "repo_checkout_commit_rows.json").write_text(json.dumps(commit_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "repo_checkout_gate_status.json").write_text(json.dumps(gate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# SWE-bench Repo Checkout Audit",
        "",
        f"- Status: `{gate['status']}`",
        f"- Repos present: `{checkout_repos}` / `{repo_count}`",
        f"- Base commits present: `{commit_ok_rows}` / `{len(commit_rows)}`",
        f"- Trees readable: `{tree_ok_rows}` / `{len(commit_rows)}`",
        "",
        "| repo | checkout exists | path |",
        "|---|---:|---|",
    ]
    for row in repo_rows:
        lines.append(f"| {row['repo']} | {row['checkout_exists']} | {row['checkout_path']} |")
    if gate["missing_or_bad_instances"]:
        lines.extend(["", "## Missing Or Bad Instances", ""])
        for row in gate["missing_or_bad_instances"][:20]:
            lines.append(
                f"- `{row['instance_id']}` `{row['repo']}` commit `{row['base_commit']}` "
                f"commit_exists={row['commit_exists']} tree_ok={row['tree_ok']} tree_error={row['tree_error']}"
            )
    (output_dir / "repo_checkout_audit.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(json.dumps({"status": gate["status"], "output_dir": str(output_dir), "repo_count": repo_count, "tree_ok_rows": tree_ok_rows}, ensure_ascii=False))


if __name__ == "__main__":
    main()

