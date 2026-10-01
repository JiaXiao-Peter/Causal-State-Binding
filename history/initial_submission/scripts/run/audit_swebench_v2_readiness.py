from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from freewill.real_task_predictive_validity import audit_swebench_v2_readiness
from freewill.utils import ensure_dir, write_json


def _write_report(path: Path, audit: dict) -> None:
    checks = audit.get("checks", {})
    lines = [
        "# SWE-bench Lite v2 Readiness Audit",
        "",
        f"- Status: `{audit.get('status')}`",
        f"- Planned remote calls: `{audit.get('planned_remote_calls')}`",
        f"- Planned task rows: `{audit.get('planned_task_rows')}`",
        f"- Planned issue rows: `{audit.get('planned_issue_rows')}`",
        f"- Execution-slice rows: `{audit.get('execution_slice_rows')}`",
        f"- Observed full-run rows: `{audit.get('observed_rows')}`",
        "",
        audit.get("scope_note", ""),
        "",
        "| check | status |",
        "|---|---:|",
    ]
    for key in sorted(checks):
        lines.append(f"| {key} | `{bool(checks[key])}` |")
    missing = audit.get("missing_or_needs_review", [])
    lines.extend(["", "## Missing Or Needs Review", ""])
    if missing:
        for item in missing:
            lines.append(f"- `{item}`")
    else:
        lines.append("- None")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit SWE-bench Lite file-localization v2 readiness artifacts.")
    parser.add_argument("--plan-dir", required=True)
    parser.add_argument("--smoke-dir", default="")
    parser.add_argument("--official-command-dir", default="")
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    plan_dir = Path(args.plan_dir)
    output_dir = ensure_dir(Path(args.output_dir) if args.output_dir else plan_dir)
    audit = audit_swebench_v2_readiness(
        plan_dir=plan_dir,
        smoke_dir=Path(args.smoke_dir) if args.smoke_dir else None,
        official_command_dir=Path(args.official_command_dir) if args.official_command_dir else None,
    )
    write_json(output_dir / "swebench_v2_readiness_audit.json", audit)
    _write_report(output_dir / "swebench_v2_readiness_audit.md", audit)
    print(
        json.dumps(
            {
                "status": audit.get("status"),
                "output_json": str(output_dir / "swebench_v2_readiness_audit.json"),
                "output_report": str(output_dir / "swebench_v2_readiness_audit.md"),
                "missing_or_needs_review": audit.get("missing_or_needs_review", []),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

