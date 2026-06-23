from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

from freewill.config import load_config
from freewill.structured_control_validation import (
    CSB_ENTROPY_PRIOR_MATCHED_VARIANT,
    CSB_ARCHITECTURES,
    CSB_FORMAL_CALIBRATION_VARIANTS,
    CSB_FORMAL_CONTROL_VARIANTS,
    CSB_TASK_FAMILIES,
    run_CSB_open_weight_pilot,
)


ROBUSTNESS_VARIANTS = [
    "structured",
    "stochastic_no_fields",
    "stochastic_context_scrambled",
    "distribution_matched",
]
ROBUSTNESS_TRANSFORMS = ["prompt_paraphrase_v1", "counterfactual_flip_v1", "counterfactual_flip_v2"]


def _phase_config(phase: str, transform: str) -> dict[str, Any]:
    if phase == "calibration":
        return {
            "variants": CSB_FORMAL_CALIBRATION_VARIANTS,
            "seeds": [1],
            "event_index_start": 0,
            "event_index_end": 12,
            "robustness_transform": "original",
            "default_cap": 6000,
        }
    if phase == "holdout_controls":
        return {
            "variants": CSB_FORMAL_CONTROL_VARIANTS,
            "seeds": [1, 2],
            "event_index_start": 12,
            "event_index_end": 36,
            "robustness_transform": "original",
            "default_cap": 25000,
        }
    if phase == "robustness":
        return {
            "variants": ROBUSTNESS_VARIANTS,
            "seeds": [1, 2],
            "event_index_start": 24,
            "event_index_end": 36,
            "robustness_transform": transform,
            "default_cap": 7000,
        }
    if phase == "entropy_prior_calibration":
        return {
            "variants": ["structured"],
            "seeds": [1],
            "event_index_start": 0,
            "event_index_end": 12,
            "robustness_transform": "original",
            "default_cap": 1200,
        }
    if phase == "entropy_prior_holdout":
        return {
            "variants": ["structured", CSB_ENTROPY_PRIOR_MATCHED_VARIANT],
            "seeds": [1, 2],
            "event_index_start": 12,
            "event_index_end": 36,
            "robustness_transform": "original",
            "default_cap": 7000,
        }
    raise ValueError(f"Unknown phase: {phase}")


def _safe_slug(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in str(value)).strip("_")


def _settings_for_model(settings: dict[str, Any], model_slug: str) -> dict[str, Any]:
    if not settings:
        return {}
    if any(key in settings for key in CSB_ARCHITECTURES):
        return settings
    if model_slug in settings and isinstance(settings[model_slug], dict):
        return settings[model_slug]
    if len(settings) == 1:
        only_value = next(iter(settings.values()))
        return only_value if isinstance(only_value, dict) else {}
    for key, value in settings.items():
        if model_slug in str(key) and isinstance(value, dict):
            return value
    return settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or dry-run formal open-weight validation slices.")
    parser.add_argument("--config", default="configs/core_study.yaml")
    parser.add_argument("--paths", default="configs/local_windows_paths.yaml")
    parser.add_argument(
        "--phase",
        choices=["calibration", "holdout_controls", "robustness", "entropy_prior_calibration", "entropy_prior_holdout"],
        required=True,
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-slug", default="")
    parser.add_argument("--profile-prefix", default="formal_validation")
    parser.add_argument("--robustness-transform", choices=ROBUSTNESS_TRANSFORMS, default="prompt_paraphrase_v1")
    parser.add_argument("--all-robustness-transforms", action="store_true")
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--hard-generation-cap", type=int, default=None)
    parser.add_argument("--entropy-settings-file", default="")
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config, args.paths)
    model_slug = args.model_slug or _safe_slug(Path(args.model_path).name or args.model_path)
    entropy_settings = {}
    if args.entropy_settings_file:
        loaded_settings = json.loads(Path(args.entropy_settings_file).read_text(encoding="utf-8"))
        entropy_settings = _settings_for_model(loaded_settings, model_slug)
    transforms = ROBUSTNESS_TRANSFORMS if args.phase == "robustness" and args.all_robustness_transforms else [args.robustness_transform]
    outputs = []
    for transform in transforms:
        phase_cfg = _phase_config(args.phase, transform)
        shard_suffix = ""
        if args.num_shards is not None or args.shard_index is not None:
            shard = int(args.shard_index or 0)
            shards = int(args.num_shards or 1)
            shard_suffix = f"_shard_{shard:02d}_of_{shards:02d}"
        transform_suffix = "" if args.phase != "robustness" else f"_{transform}"
        profile = f"{args.profile_prefix}_{args.phase}_{model_slug}{transform_suffix}{shard_suffix}"
        out = run_CSB_open_weight_pilot(
            config,
            profile_name=profile,
            events_per_family=36,
            event_index_start=phase_cfg["event_index_start"],
            event_index_end=phase_cfg["event_index_end"],
            robustness_transform=phase_cfg["robustness_transform"],
            task_families=CSB_TASK_FAMILIES,
            architectures=CSB_ARCHITECTURES,
            variants=args.variants or phase_cfg["variants"],
            entropy_decoding_settings=entropy_settings,
            seeds=phase_cfg["seeds"],
            model_path=args.model_path,
            hard_generation_cap=int(args.hard_generation_cap or phase_cfg["default_cap"]),
            dry_run=bool(args.dry_run),
            shard_index=args.shard_index,
            num_shards=args.num_shards,
        )
        outputs.append(out["gate_status"])
        print(json.dumps({"profile": profile, **out["gate_status"]}, ensure_ascii=False), flush=True)
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

    total_planned = sum(int(item.get("planned_generations", 0)) for item in outputs)
    total_calls = sum(int(item.get("planned_model_calls", 0)) for item in outputs)
    print(json.dumps({"total_planned_generations": total_planned, "total_planned_model_calls": total_calls}, ensure_ascii=False))


if __name__ == "__main__":
    main()

