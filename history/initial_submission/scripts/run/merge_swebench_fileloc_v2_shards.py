from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from freewill.real_task_predictive_validity import merge_swebench_filelocalization_v2_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge SWE-bench Lite file-localization v2 shard outputs.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Shard run directories or trace files.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=400)
    parser.add_argument("--delta-auc-gate", type=float, default=0.03)
    parser.add_argument("--noninferiority-margin", type=float, default=-0.03)
    args = parser.parse_args()

    result = merge_swebench_filelocalization_v2_outputs(
        [Path(path) for path in args.inputs],
        output_dir=Path(args.output_dir),
        bootstrap_samples=args.bootstrap_samples,
        delta_auc_gate=args.delta_auc_gate,
        noninferiority_margin=args.noninferiority_margin,
    )
    gate = result["gate_status"]
    print(
        json.dumps(
            {
                "output_dir": args.output_dir,
                "status": gate.get("status"),
                "protocol_coverage_status": gate.get("protocol_coverage_status"),
                "merged_trace_rows": gate.get("merged_trace_rows"),
                "merged_unique_task_keys": gate.get("merged_unique_task_keys"),
                "delta_auc": gate.get("delta_auc"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

