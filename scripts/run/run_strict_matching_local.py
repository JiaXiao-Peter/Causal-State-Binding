from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from freewill.causal_state_sufficiency import safe_slug
from freewill.real_task_predictive_validity import load_default_config
from freewill.strict_matching_local import run_local_strict_matching_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local open-weight strict entropy/token matching audit.")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--events-per-family", type=int, default=8)
    parser.add_argument("--event-index-start", type=int, default=0)
    parser.add_argument("--event-index-end", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-root", default="results/paper1_revision/local_strict_matching")
    args = parser.parse_args()

    # Load the project config once so this runner fails early if the repository config is broken.
    load_default_config()
    run_id = args.run_id or f"strict_match_{safe_slug(args.model_id)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(args.output_root) / run_id
    result = run_local_strict_matching_audit(
        output_dir=output_dir,
        model_id=args.model_id,
        events_per_family=args.events_per_family,
        event_index_start=args.event_index_start,
        event_index_end=args.event_index_end,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        dry_run=bool(args.dry_run),
    )
    gate = result["gate_status"]
    print(
        {
            "output_dir": str(output_dir),
            "status": gate.get("status"),
            "observed_rows": gate.get("observed_rows", 0),
            "entropy_gap": gate.get("canonical_entropy_gap_bits"),
            "prompt_match_rate": gate.get("prompt_match_rate"),
        }
    )


if __name__ == "__main__":
    main()

