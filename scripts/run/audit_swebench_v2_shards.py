from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from freewill.real_task_predictive_validity import audit_swebench_v2_shards
from freewill.utils import ensure_dir, write_json


def _write_report(path: Path, audit: dict) -> None:
    lines = [
        "# SWE-bench Lite v2 Shard Status",
        "",
        f"- Status: `{audit.get('status')}`",
        f"- Shards started: `{audit.get('started_shards')}` / `{audit.get('shard_count')}`",
        f"- Shards complete: `{audit.get('complete_shards')}` / `{audit.get('shard_count')}`",
        f"- Merge ready: `{audit.get('merge_ready')}`",
        f"- Merged results present: `{audit.get('merged_results_present')}`",
        "",
        audit.get("scope_note", ""),
        "",
        "| shard | run_id | status | planned rows | trace rows | checkpoint rows | errors | error rate | coverage |",
        "|---:|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in audit.get("shards", []):
        lines.append(
            f"| {row.get('shard_index')} | `{row.get('run_id')}` | `{row.get('status')}` | "
            f"{row.get('planned_task_rows')} | {row.get('trace_rows')} | {row.get('checkpoint_rows')} | "
            f"{row.get('error_rows')} | {float(row.get('error_rate', 0.0) or 0.0):.4f} | `{row.get('coverage_status')}` |"
        )
    missing = audit.get("missing_or_needs_review", [])
    lines.extend(["", "## Missing Or Needs Review", ""])
    if missing:
        for item in missing:
            lines.append(f"- `{item}`")
    else:
        lines.append("- None")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit SWE-bench Lite v2 shard run completeness and merge readiness.")
    parser.add_argument("--launch-plan", required=True, help="swebench_v2_launch_plan.json")
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    launch_plan = Path(args.launch_plan)
    output_dir = ensure_dir(Path(args.output_dir) if args.output_dir else launch_plan.parent)
    audit = audit_swebench_v2_shards(launch_plan_path=launch_plan)
    write_json(output_dir / "swebench_v2_shard_status.json", audit)
    _write_report(output_dir / "swebench_v2_shard_status.md", audit)
    print(
        json.dumps(
            {
                "status": audit.get("status"),
                "output_json": str(output_dir / "swebench_v2_shard_status.json"),
                "output_report": str(output_dir / "swebench_v2_shard_status.md"),
                "started_shards": audit.get("started_shards"),
                "complete_shards": audit.get("complete_shards"),
                "missing_or_needs_review": audit.get("missing_or_needs_review", []),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

