from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from freewill.real_task_predictive_validity import audit_swebench_v2_claim_support
from freewill.utils import ensure_dir, write_json


def _write_report(path: Path, audit: dict) -> None:
    lines = [
        "# SWE-bench v2 Claim-Support Guard",
        "",
        f"- Status: `{audit.get('status')}`",
        f"- Scanned files: `{audit.get('scanned_files')}`",
        f"- Full API results complete: `{audit.get('full_api_results_complete')}`",
        f"- Official execution results present: `{audit.get('official_execution_results_present')}`",
        "",
        audit.get("scope_note", ""),
        "",
        "## Conservative Wording Checks",
        "",
        "| check | status |",
        "|---|---:|",
    ]
    for key, value in sorted(audit.get("conservative_checks", {}).items()):
        lines.append(f"| {key} | `{bool(value)}` |")
    lines.extend(["", "## Findings", ""])
    findings = audit.get("findings", [])
    if findings:
        lines.extend(["| file | line | issue | text |", "|---|---:|---|---|"])
        for item in findings:
            text = str(item.get("text", "")).replace("|", "\\|")
            lines.append(f"| {item.get('file', '')} | {item.get('line', 0)} | `{item.get('issue', '')}` | {text} |")
    else:
        lines.append("- None")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit manuscript/report wording against SWE-bench v2 result support.")
    parser.add_argument("--readiness-audit", required=True)
    parser.add_argument("--paths", nargs="+", required=True)
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    readiness_path = Path(args.readiness_audit)
    output_dir = ensure_dir(Path(args.output_dir) if args.output_dir else readiness_path.parent)
    audit = audit_swebench_v2_claim_support(
        readiness_audit_path=readiness_path,
        manuscript_paths=[Path(path) for path in args.paths],
    )
    write_json(output_dir / "swebench_v2_claim_support_audit.json", audit)
    _write_report(output_dir / "swebench_v2_claim_support_audit.md", audit)
    print(
        json.dumps(
            {
                "status": audit.get("status"),
                "output_json": str(output_dir / "swebench_v2_claim_support_audit.json"),
                "output_report": str(output_dir / "swebench_v2_claim_support_audit.md"),
                "finding_count": len(audit.get("findings", [])),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

