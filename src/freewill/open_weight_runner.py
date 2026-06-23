from __future__ import annotations

import hashlib
import json
import os
import platform
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from freewill.behavioral_metrics import compute_afci_records, summarize_afci
from freewill.config import ProjectConfig
from freewill.schema_equalized_agents import (
    SCHEMA_EQUALIZED_FIELDS,
    _load_or_build_manifest,
    _normalize_payload,
    _schema_payload,
)
from freewill.utils import ensure_dir, write_json, write_parquet


OPEN_WEIGHT_DEFAULT_VARIANTS = ["A4", "A5", "A4_no_reason", "A4_no_veto", "A5_schema_posthoc"]
OPEN_WEIGHT_DEFAULT_MODEL_IDS = [
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
]


def _runtime_root(config: ProjectConfig) -> Path:
    return Path(config.paths.runtime_root)


def _result_dir(config: ProjectConfig, *parts: str) -> Path:
    return ensure_dir(_runtime_root(config) / "results" / "paper1_revision" / "open_weight_anchor" / Path(*parts))


def _report_dir(config: ProjectConfig) -> Path:
    return ensure_dir(_runtime_root(config) / "reports" / "paper1_revision")


def _hash_file(path: Path, max_bytes: int = 1024 * 1024 * 128) -> str:
    digest = hashlib.sha256()
    read = 0
    with path.open("rb") as handle:
        while read < max_bytes:
            block = handle.read(min(1024 * 1024, max_bytes - read))
            if not block:
                break
            digest.update(block)
            read += len(block)
    return digest.hexdigest()


def _hash_tree_hint(path: Path) -> str:
    digest = hashlib.sha256()
    for candidate in sorted(path.glob("*")):
        if candidate.is_file() and candidate.name in {"config.json", "tokenizer.json", "tokenizer_config.json"}:
            digest.update(candidate.name.encode("utf-8"))
            digest.update(candidate.read_bytes())
    for candidate in sorted(path.glob("*.safetensors"))[:2]:
        digest.update(candidate.name.encode("utf-8"))
        digest.update(_hash_file(candidate).encode("utf-8"))
    return digest.hexdigest()


def _hash_identifier_hint(identifier: str) -> str:
    return hashlib.sha256(str(identifier).encode("utf-8")).hexdigest()


def _candidate_model_paths() -> list[Path]:
    roots = [
        Path.home() / ".cache" / "huggingface" / "hub",
        Path("E:/Data/hf_cache/hub"),
    ]
    names = [
        "models--Qwen--Qwen3-14B",
        "models--Qwen--Qwen3-14B-Instruct",
        "models--Qwen--Qwen2.5-14B-Instruct",
        "models--Qwen--Qwen2.5-7B-Instruct",
        "models--Qwen--Qwen3-1.7B",
        "models--Qwen--Qwen2.5-1.5B-Instruct",
        "models--HuggingFaceTB--SmolLM2-1.7B-Instruct",
        "models--microsoft--Phi-3.5-mini-instruct",
    ]
    found: list[Path] = []
    for root in roots:
        for name in names:
            base = root / name
            snapshots = base / "snapshots"
            if snapshots.exists():
                found.extend(path for path in sorted(snapshots.iterdir()) if path.is_dir())
            elif base.exists():
                found.append(base)
    return found


def _model_candidates(preferred: str | None = None) -> list[str | Path]:
    candidates = _candidate_model_paths()
    ordered: list[str | Path] = []
    if preferred:
        preferred_path = Path(preferred)
        if preferred_path.exists():
            ordered.append(preferred_path)
        else:
            ordered.append(preferred)
        preferred_text = str(preferred)
        if "Mistral-7B-Instruct-v0.3" in preferred_text:
            for fallback in [
                "/root/autodl-tmp/hf/models/Mistral-7B-Instruct-v0.3",
                "/data/hf/models/Mistral-7B-Instruct-v0.3",
                "mistralai/Mistral-7B-Instruct-v0.3",
            ]:
                if fallback not in ordered:
                    ordered.append(fallback)
        if "Qwen2.5-14B-Instruct" in preferred_text and "Qwen/Qwen2.5-14B-Instruct" not in ordered:
            ordered.append("Qwen/Qwen2.5-14B-Instruct")
        if "Qwen2.5-32B-Instruct" in preferred_text and "Qwen/Qwen2.5-32B-Instruct" not in ordered:
            ordered.append("Qwen/Qwen2.5-32B-Instruct")
        for path in candidates:
            if preferred.lower() in str(path).lower():
                ordered.append(path)
    ordered.extend(path for path in candidates if path not in ordered)
    ordered.extend(model_id for model_id in OPEN_WEIGHT_DEFAULT_MODEL_IDS if model_id not in ordered)
    return ordered


def _open_variant_to_schema_variant(variant: str) -> str:
    mapping = {
        "A4": "A4_full",
        "A5": "A5_plain",
        "A5_schema_posthoc": "A5_schema_posthoc_fields",
    }
    return mapping.get(variant, "A4_full")


def _top_up_sampled_events(sampled: pd.DataFrame, *, sample_size: int, random_seed: int = 2107) -> pd.DataFrame:
    if sampled.empty or "output_namespace" not in sampled.columns:
        return sampled
    rows: list[pd.DataFrame] = []
    for dataset_index, (namespace, frame) in enumerate(sampled.groupby("output_namespace", sort=True)):
        group = frame.head(sample_size).copy()
        group["open_weight_resampled_with_replacement"] = False
        group["open_weight_source_event_id"] = group.get("event_id", pd.Series("", index=group.index)).astype(str)
        missing = sample_size - len(group)
        if missing > 0 and not group.empty:
            extra = group.sample(n=missing, replace=True, random_state=random_seed + dataset_index).copy().reset_index(drop=True)
            extra["open_weight_resampled_with_replacement"] = True
            extra["open_weight_source_event_id"] = extra.get("event_id", pd.Series("", index=extra.index)).astype(str)
            extra["event_id"] = [
                f"{source}__open_weight_resample_{index + 1:03d}"
                for index, source in enumerate(extra["open_weight_source_event_id"].astype(str).tolist())
            ]
            group = pd.concat([group, extra], ignore_index=True)
        rows.append(group)
    return pd.concat(rows, ignore_index=True) if rows else sampled


def _load_open_weight_manifest(
    config: ProjectConfig,
    *,
    sample_size: int,
    dataset_ids: Iterable[str] | None,
) -> pd.DataFrame:
    from freewill.ai_matrix import build_sample_manifest

    sampled = _load_or_build_manifest(config, sample_size=sample_size, smoke=False, dataset_ids=dataset_ids)
    counts = sampled.groupby("output_namespace").size() if not sampled.empty and "output_namespace" in sampled.columns else pd.Series(dtype=int)
    if counts.empty or bool((counts < sample_size).any()):
        fresh = build_sample_manifest(config, dataset_ids=dataset_ids, sample_size=sample_size, smoke=False)
        fresh = fresh[fresh.get("sample_status", pd.Series(dtype=str)).astype(str).eq("sampled")].copy()
        fresh_counts = fresh.groupby("output_namespace").size() if not fresh.empty and "output_namespace" in fresh.columns else pd.Series(dtype=int)
        if int(fresh_counts.sum()) >= int(counts.sum()):
            sampled = fresh
    return _top_up_sampled_events(sampled, sample_size=sample_size)


def _default_payload(row: dict[str, Any], variant: str, seed: int) -> dict[str, Any]:
    payload = _schema_payload(row, _open_variant_to_schema_variant(variant), seed)
    payload["module_metadata"] = dict(payload["module_metadata"])
    payload["module_metadata"]["open_weight_variant"] = variant
    if variant == "A4_no_reason":
        payload["reason_graph"] = {"nodes": [], "weights": {}, "preferred_action": ""}
        payload["module_metadata"]["action_field_coupling"]["reason"] = False
        payload["module_metadata"]["field_generation"] = "reason_ablated"
    if variant == "A4_no_veto":
        payload["veto_state"] = {"applied": False, "condition_required": False, "strength": 0.0}
        if str(payload["final_action"]) == "veto_or_withhold":
            payload["final_action"] = payload["candidate_actions"][0]
        payload["module_metadata"]["action_field_coupling"]["veto"] = False
        payload["module_metadata"]["field_generation"] = "veto_ablated"
    payload["module_metadata"]["variant"] = variant
    return payload


def _prompt(row: dict[str, Any], variant: str) -> str:
    return (
        "Return exactly one compact JSON object and nothing else. "
        "Use short strings. No markdown. No explanations outside JSON. "
        "Close all braces. Required top-level keys: "
        f"{', '.join(SCHEMA_EQUALIZED_FIELDS)}.\n"
        "Use this compact shape: "
        '{"first_impulse":"a","candidate_actions":["a","b"],'
        '"reason_graph":{"nodes":["reason_supports::a"],"weights":{"top_reason":1.0},"preferred_action":"a"},'
        '"self_state":{"commitment":"a"},'
        '"memory_trace":{"relevant":"short"},'
        '"veto_state":{"applied":false,"condition_required":false,"strength":0.0,"vetoed_action":""},'
        '"final_action":"a","final_action_rationale":"short",'
        '"module_metadata":{"variant":"VARIANT","schema_equalized":true}}.\n'
        "Limit candidate_actions to two items and reason_graph.nodes to two items.\n"
        f"Variant: {variant}\n"
        f"Dataset: {row.get('dataset_id')}\n"
        f"Event id: {row.get('event_id')}\n"
        f"Context: {str(row.get('text', ''))[:900]}\n"
        "A4 should use reason, memory, self-state and veto when available. "
        "A5 should make a stochastic choice without structured action-field coupling. "
        "A4_no_reason removes reason coupling. A4_no_veto removes veto coupling. "
        "A5_schema_posthoc decides final_action first and then writes post-hoc fields."
    )


class _LocalGenerator:
    def __init__(self, model_identifier: str | Path):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model_identifier = model_identifier
        self.model_path = Path(model_identifier) if Path(str(model_identifier)).exists() else None
        model_ref = str(self.model_path or model_identifier)
        local_only = self.model_path is not None
        self.tokenizer = AutoTokenizer.from_pretrained(model_ref, local_files_only=local_only, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        kwargs: dict[str, Any] = {"local_files_only": local_only, "trust_remote_code": True}
        if torch.cuda.is_available():
            kwargs["device_map"] = str(os.environ.get("FREEWILL_OPEN_WEIGHT_DEVICE_MAP", "auto")).strip() or "auto"
            kwargs["torch_dtype"] = "auto"
            kwargs["low_cpu_mem_usage"] = True
            max_memory_gib = str(os.environ.get("FREEWILL_OPEN_WEIGHT_MAX_MEMORY_GIB", "")).strip()
            if max_memory_gib:
                kwargs["max_memory"] = {index: f"{max_memory_gib}GiB" for index in range(torch.cuda.device_count())}
            if str(os.environ.get("FREEWILL_OPEN_WEIGHT_LOAD_IN_4BIT", "")).strip().lower() in {"1", "true", "yes", "y"}:
                from transformers import BitsAndBytesConfig

                compute_dtype_name = str(os.environ.get("FREEWILL_OPEN_WEIGHT_BNB_4BIT_COMPUTE_DTYPE", "bfloat16"))
                compute_dtype = getattr(torch, compute_dtype_name, torch.bfloat16)
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type=str(os.environ.get("FREEWILL_OPEN_WEIGHT_BNB_4BIT_QUANT_TYPE", "nf4")),
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=str(
                        os.environ.get("FREEWILL_OPEN_WEIGHT_BNB_4BIT_USE_DOUBLE_QUANT", "1")
                    ).strip().lower()
                    in {"1", "true", "yes", "y"},
                )
        self.model = AutoModelForCausalLM.from_pretrained(model_ref, **kwargs)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            self.model.to("cpu")

    @property
    def identifier(self) -> str:
        return str(self.model_path or self.model_identifier)

    def _chat_texts(self, prompts: list[str]) -> list[str]:
        texts: list[str] = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            if hasattr(self.tokenizer, "apply_chat_template"):
                texts.append(self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            else:
                texts.append(prompt)
        return texts

    def prompt_token_counts(self, prompts: list[str]) -> list[int]:
        if not prompts:
            return []
        texts = self._chat_texts(prompts)
        encoded = self.tokenizer(texts, return_tensors=None, padding=False)
        input_ids = encoded.get("input_ids", [])
        return [int(len(item)) for item in input_ids]

    def text_token_count(self, text: str) -> int:
        encoded = self.tokenizer(str(text), return_tensors=None, padding=False)
        return int(len(encoded.get("input_ids", [])))

    def hash_hint(self) -> str:
        if self.model_path is not None:
            return _hash_tree_hint(self.model_path)
        return _hash_identifier_hint(str(self.model_identifier))

    def generate(self, prompt: str, *, seed: int, temperature: float = 0.2, top_p: float = 1.0, max_new_tokens: int = 320) -> tuple[dict[str, Any], str]:
        payload, raw_text = self.generate_many([prompt], seed=seed, temperature=temperature, top_p=top_p, max_new_tokens=max_new_tokens)[0]
        if "__open_weight_parse_error__" in payload:
            raise ValueError(str(payload["__open_weight_parse_error__"]))
        return payload, raw_text

    def generate_many(
        self,
        prompts: list[str],
        *,
        seed: int,
        temperature: float = 0.2,
        top_p: float = 1.0,
        max_new_tokens: int = 320,
    ) -> list[tuple[dict[str, Any], str]]:
        self.torch.manual_seed(int(seed))
        texts = self._chat_texts(prompts)
        encoded = self.tokenizer(texts, return_tensors="pt", padding=True)
        encoded = {key: value.to(self.model.device) for key, value in encoded.items()}
        with self.torch.no_grad():
            output = self.model.generate(
                **encoded,
                do_sample=temperature > 0,
                temperature=max(float(temperature), 1e-5),
                top_p=float(top_p),
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        results: list[tuple[dict[str, Any], str]] = []
        prompt_width = int(encoded["input_ids"].shape[-1])
        for item in output:
            generated = self.tokenizer.decode(item[prompt_width:], skip_special_tokens=True)
            start = generated.find("{")
            end = generated.rfind("}")
            if start >= 0 and end > start:
                try:
                    results.append((json.loads(generated[start : end + 1]), generated))
                except json.JSONDecodeError as exc:
                    results.append(({"__open_weight_parse_error__": f"json_decode_error:{exc.msg}"}, generated))
            else:
                results.append(({"__open_weight_parse_error__": "no_json_object"}, generated))
        return results


def run_open_weight_anchor(
    config: ProjectConfig,
    *,
    dataset_ids: Iterable[str] | None = None,
    smoke: bool = False,
    sample_size: int | None = None,
    seeds: Iterable[int] | None = None,
    variants: list[str] | None = None,
    model_path: str | None = None,
    hard_generation_cap: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    profile = "smoke" if smoke else "full"
    selected_variants = variants or (["A4", "A5", "A4_no_reason", "A4_no_veto"] if smoke else OPEN_WEIGHT_DEFAULT_VARIANTS)
    selected_seeds = [int(seed) for seed in (seeds or ([1, 2] if smoke else [1, 2, 3]))]
    chosen_sample_size = int(sample_size or (20 if smoke else 100))
    cap = int(hard_generation_cap or (1500 if smoke else 12000))
    selected_dataset_ids = list(dataset_ids) if dataset_ids else list(config.ai_matrix.datasets)
    # The open-weight smoke is intentionally a smoke profile over all seven Paper1
    # namespaces, not the smaller API smoke subset.
    sampled = _load_open_weight_manifest(config, sample_size=chosen_sample_size, dataset_ids=selected_dataset_ids)
    planned = int(len(sampled) * len(selected_variants) * len(selected_seeds))
    replacement_count = int(sampled.get("open_weight_resampled_with_replacement", pd.Series(False, index=sampled.index)).fillna(False).astype(bool).sum()) if not sampled.empty else 0
    source_event_count = max(0, int(len(sampled)) - replacement_count)
    result_dir = _result_dir(config, profile)
    plan = {
        "experiment": "open_weight_anchor",
        "profile": profile,
        "planned_generations": planned,
        "sampled_events": int(len(sampled)),
        "unique_source_events": source_event_count,
        "resampled_with_replacement_events": replacement_count,
        "dataset_count": int(sampled["output_namespace"].nunique()) if "output_namespace" in sampled.columns else 0,
        "sample_size_per_dataset": chosen_sample_size,
        "variants": selected_variants,
        "seeds": selected_seeds,
        "hard_generation_cap": cap,
        "dry_run": bool(dry_run),
        "model_candidates": [str(item) for item in _model_candidates(model_path)],
    }
    write_json(result_dir / "run_plan.json", plan)
    if planned > cap:
        raise RuntimeError(f"open_weight_anchor planned generations {planned} exceed hard cap {cap}")
    if dry_run:
        gate = {"status": "planned", **plan}
        write_json(result_dir / "gate_status.json", gate)
        return {"gate_status": gate}

    selected_model: str | Path | None = None
    generator: _LocalGenerator | None = None
    load_errors: list[dict[str, str]] = []
    for candidate in _model_candidates(model_path):
        try:
            generator = _LocalGenerator(candidate)
            selected_model = candidate
            break
        except Exception as exc:  # noqa: BLE001
            load_errors.append({"model": str(candidate), "error": f"{type(exc).__name__}: {exc}"[:500]})
            continue
    if generator is None or selected_model is None:
        gate = {"status": "blocked", **plan, "reason": "open_weight_model_load_failed", "model_load_errors": load_errors}
        write_json(result_dir / "gate_status.json", gate)
        return {"gate_status": gate}

    model_info = {
        "model_path": str(selected_model),
        "model_identifier": generator.identifier,
        "model_hash_hint": generator.hash_hint(),
        "device": generator.device,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "batch_size": int(os.environ.get("FREEWILL_OPEN_WEIGHT_BATCH_SIZE", "1")),
        "device_map": str(os.environ.get("FREEWILL_OPEN_WEIGHT_DEVICE_MAP", "auto")),
        "max_memory_gib": str(os.environ.get("FREEWILL_OPEN_WEIGHT_MAX_MEMORY_GIB", "")),
        "load_in_4bit": str(os.environ.get("FREEWILL_OPEN_WEIGHT_LOAD_IN_4BIT", "")).strip().lower()
        in {"1", "true", "yes", "y"},
        "model_load_errors_before_selected": load_errors,
    }
    write_json(result_dir / "model_info.json", model_info)

    trace_path = result_dir / "traces.parquet"
    behavior_path = result_dir / "behaviors.parquet"
    if trace_path.exists() and behavior_path.exists():
        existing_traces = pd.read_parquet(trace_path)
        existing_behaviors = pd.read_parquet(behavior_path)
        trace_rows: list[dict[str, Any]] = existing_traces.to_dict(orient="records")
        behavior_rows: list[dict[str, Any]] = existing_behaviors.to_dict(orient="records")
    else:
        trace_rows = []
        behavior_rows = []
    existing_keys = {
        (str(row.get("output_namespace", "")), str(row.get("variant", "")), int(row.get("seed", 0)), str(row.get("event_id", "")))
        for row in trace_rows
    }
    failure_count = 0
    batch_size = max(1, int(os.environ.get("FREEWILL_OPEN_WEIGHT_BATCH_SIZE", "1")))
    for namespace, frame in sampled.groupby("output_namespace", sort=True):
        records = frame.reset_index(drop=True).to_dict(orient="records")
        for seed in selected_seeds:
            for variant in selected_variants:
                pending: list[tuple[int, dict[str, Any], tuple[str, str, int, str], dict[str, Any], float]] = []
                for index, row in enumerate(records):
                    key = (str(namespace), str(variant), int(seed), str(row.get("event_id", "")))
                    if key in existing_keys:
                        continue
                    default = _default_payload(row, variant, seed)
                    temperature = 0.2 if variant.startswith("A4") else 0.9
                    pending.append((index, row, key, default, temperature))
                for batch_start in range(0, len(pending), batch_size):
                    batch = pending[batch_start : batch_start + batch_size]
                    prompts = [_prompt(row, variant) for index, row, key, default, temperature in batch]
                    temperature = float(batch[0][4]) if batch else 0.2
                    try:
                        generated_batch = generator.generate_many(
                            prompts,
                            seed=int(seed) * 100000 + batch_start,
                            temperature=temperature,
                            max_new_tokens=int(os.environ.get("FREEWILL_OPEN_WEIGHT_MAX_NEW_TOKENS", "512")),
                        )
                    except Exception:
                        generated_batch = []
                    for item_index, (index, row, key, default, temperature) in enumerate(batch):
                        try:
                            if generated_batch:
                                generated, raw_text = generated_batch[item_index]
                            else:
                                generated, raw_text = generator.generate(
                                    _prompt(row, variant),
                                    seed=seed + index,
                                    temperature=temperature,
                                    max_new_tokens=int(os.environ.get("FREEWILL_OPEN_WEIGHT_MAX_NEW_TOKENS", "512")),
                                )
                            if "__open_weight_parse_error__" in generated:
                                payload = default
                                error_type = f"generation_or_parse_error:{generated['__open_weight_parse_error__']}"
                                failure_count += 1
                            else:
                                payload = _normalize_payload(generated, default)
                                error_type = ""
                        except Exception as exc:  # noqa: BLE001
                            payload = default
                            raw_text = ""
                            error_type = f"generation_or_parse_error:{type(exc).__name__}"
                            failure_count += 1
                        provider_metadata = {
                            "model": str(selected_model),
                            "model_path": str(selected_model),
                            "model_hash_hint": model_info["model_hash_hint"],
                            "remote_used": False,
                            "open_weight_used": True,
                            "fallback_used": False,
                            "error_type": error_type,
                            "request_temperature": temperature,
                            "request_top_p": 1.0,
                            "sampling_seed": int(seed + index),
                            "batch_size": int(batch_size),
                        }
                        trace_rows.append(
                            {
                                "dataset_id": row.get("dataset_id", ""),
                                "variant": variant,
                                "episode_id": row.get("episode_id", ""),
                                "step_id": f"{row.get('episode_id', '')}:{variant}:{index + 1:03d}",
                                "event_id": row.get("event_id", ""),
                                "output_namespace": namespace,
                                "seed": seed,
                                "raw_generation": raw_text,
                                **payload,
                                "provider_metadata": provider_metadata,
                            }
                        )
                        behavior_rows.append(
                            {
                                "dataset_id": row.get("dataset_id", ""),
                                "variant": variant,
                                "episode_id": row.get("episode_id", ""),
                                "subject_id": variant,
                                "first_impulse": payload["first_impulse"],
                                "final_action": payload["final_action"],
                                "reasons": payload.get("reason_graph", {}).get("nodes", []),
                                "choice_metadata": {"event_id": row.get("event_id", ""), "provider_metadata": provider_metadata},
                                "output_namespace": namespace,
                                "seed": seed,
                            }
                        )
                        existing_keys.add(key)
                    print(
                        json.dumps(
                            {
                                "event": "open_weight_batch",
                                "profile": profile,
                                "dataset": str(namespace),
                                "seed": int(seed),
                                "variant": str(variant),
                                "batch_start": int(batch_start),
                                "batch_size": int(len(batch)),
                                "total_traces": int(len(trace_rows)),
                                "failure_count": int(failure_count),
                            }
                        ),
                        flush=True,
                    )
                if trace_rows:
                    write_parquet(pd.DataFrame(trace_rows), trace_path)
                    write_parquet(pd.DataFrame(behavior_rows), behavior_path)
    traces = pd.DataFrame(trace_rows)
    behaviors = pd.DataFrame(behavior_rows)
    afci = compute_afci_records(traces)
    summary = summarize_afci(afci)
    gate: dict[str, Any] = {
        "status": "needs_review",
        "pass_rule": "A4 AFCI > A5 in at least 5/7 datasets and parse failure < 5%",
    }
    if not traces.empty and "provider_metadata" in traces.columns:
        meta_rows = []
        for value in traces["provider_metadata"].tolist():
            if isinstance(value, dict):
                meta_rows.append(value)
            elif isinstance(value, str):
                try:
                    decoded = json.loads(value)
                except json.JSONDecodeError:
                    decoded = {}
                meta_rows.append(decoded if isinstance(decoded, dict) else {})
            else:
                meta_rows.append({})
        meta = pd.DataFrame(meta_rows)
        failure_count = int(meta.get("error_type", pd.Series("", index=meta.index)).fillna("").astype(str).ne("").sum()) if not meta.empty else 0
    parse_failure_rate = float(failure_count / max(1, planned))
    directional_count = 0
    for _, frame in summary.groupby("dataset_namespace", sort=True) if not summary.empty else []:
        a4 = frame[frame["variant"].astype(str).eq("A4")]
        a5 = frame[frame["variant"].astype(str).eq("A5")]
        if not a4.empty and not a5.empty and float(a4["AFCI"].mean()) > float(a5["AFCI"].mean()):
            directional_count += 1
    gate.update(
        {
            "status": "pass" if directional_count >= 5 and parse_failure_rate < 0.05 else "needs_review",
            "experiment": "open_weight_anchor",
            "profile": profile,
            "planned_generations": planned,
            "generation_count": int(len(traces)),
            "parse_failure_count": int(failure_count),
            "parse_failure_rate": parse_failure_rate,
            "a4_afci_gt_a5_dataset_count": int(directional_count),
            **model_info,
        }
    )
    write_parquet(traces, result_dir / "traces.parquet")
    write_parquet(behaviors, result_dir / "behaviors.parquet")
    write_parquet(afci, result_dir / "open_weight_anchor_metrics.parquet")
    write_parquet(summary, result_dir / "open_weight_anchor_summary.parquet")
    if profile == "full":
        write_parquet(afci, _runtime_root(config) / "results" / "paper1_revision" / "open_weight_anchor_metrics.parquet")
    write_json(result_dir / "gate_status.json", gate)
    _write_report(config, gate, summary)
    return {"gate_status": gate, "metrics": afci, "summary": summary}


def _write_report(config: ProjectConfig, gate: dict[str, Any], summary: pd.DataFrame) -> Path:
    path = _report_dir(config) / "open_weight_anchor.md"
    lines = [
        "# Paper1 Revision Open-Weight Anchor",
        "",
        f"- Gate: `{gate.get('status')}`",
        f"- Model path: `{gate.get('model_path', '')}`",
        f"- Model hash hint: `{gate.get('model_hash_hint', '')}`",
        f"- Device: `{gate.get('device', '')}`",
        f"- A4 AFCI > A5 datasets: `{gate.get('a4_afci_gt_a5_dataset_count')}`",
        f"- Parse failure rate: `{gate.get('parse_failure_rate')}`",
        "",
    ]
    if not summary.empty:
        lines.extend(["| dataset | variant | AFCI | reason | memory | veto | self |", "|---|---|---:|---:|---:|---:|---:|"])
        grouped = summary.groupby(["dataset_namespace", "variant"], sort=True).mean(numeric_only=True).reset_index()
        for row in grouped.head(120).to_dict(orient="records"):
            lines.append(
                f"| {row.get('dataset_namespace')} | {row.get('variant')} | {float(row.get('AFCI', 0.0)):.3f} | "
                f"{float(row.get('reason_action_alignment', 0.0)):.3f} | {float(row.get('memory_action_alignment', 0.0)):.3f} | "
                f"{float(row.get('veto_action_alignment', 0.0)):.3f} | {float(row.get('self_action_alignment', 0.0)):.3f} |"
            )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path

