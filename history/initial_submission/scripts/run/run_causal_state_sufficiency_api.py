from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from freewill.causal_state_sufficiency import (
    load_default_config,
    read_api_settings,
    run_causal_state_sufficiency,
    safe_slug,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the decisive-field positive-control and control-binding wrapper API experiment.")
    parser.add_argument("--api-file", default="", help="Local credentials file. The key is read but never written to outputs.")
    parser.add_argument("--models", nargs="+", default=["gpt-5.4-mini"])
    parser.add_argument("--events-per-family", type=int, default=12)
    parser.add_argument("--event-index-start", type=int, default=0)
    parser.add_argument("--event-index-end", type=int, default=6)
    parser.add_argument("--hard-call-cap", type=int, default=2000)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-root", default="results/paper1_revision/causal_state_sufficiency")
    args = parser.parse_args()

    api_settings = read_api_settings(Path(args.api_file) if args.api_file else None)
    config = load_default_config()
    model_slug = "_".join(safe_slug(model) for model in args.models)
    run_id = args.run_id or f"api_{model_slug}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(args.output_root) / run_id
    result = run_causal_state_sufficiency(
        config=config,
        api_settings=api_settings,
        output_dir=output_dir,
        models=args.models,
        events_per_family=args.events_per_family,
        event_index_start=args.event_index_start,
        event_index_end=args.event_index_end,
        hard_call_cap=args.hard_call_cap,
        max_workers=args.max_workers,
        dry_run=bool(args.dry_run),
    )
    gate = result["gate_status"]
    print({"output_dir": str(output_dir), "status": gate.get("status"), "observed_rows": gate.get("observed_rows", 0)})


if __name__ == "__main__":
    main()

