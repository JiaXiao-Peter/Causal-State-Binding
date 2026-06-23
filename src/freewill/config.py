from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from freewill.simplemodel import Field, SimpleModel


class ProviderConfig(SimpleModel):
    mode: str = "openai_compatible"
    model: str = "gpt-5.4"
    base_url: str = "https://api.openai.com/v1"
    endpoint_path: str = "/chat/completions"
    api_key_env: str = "OPENAI_API_KEY"
    timeout_seconds: int = 60
    temperature: float = 0.2
    top_p: float = 1.0
    json_mode: str = "json_object"
    provider_role: str = "generic"
    fallback_model: str = "gpt-4.1"
    fallback_base_url: str = ""
    fallback_endpoint_path: str = ""
    fallback_api_key_env: str = ""
    fallback_temperature: float = -1.0
    fallback_top_p: float = -1.0
    fallback_json_mode: str = ""


class PathsConfig(SimpleModel):
    raw_root: str
    narratives_root: str
    narratives_mirror_root: str
    openneuro_stage_root: str = "./data/openneuro-stage"
    notebook_ready_root: str
    project_root: str
    runtime_root: str
    artifacts_root: str
    results_root: str
    reports_root: str

    def resolve(self) -> "PathsConfig":
        data = {name: str(Path(value)) for name, value in self.model_dump().items()}
        return PathsConfig(**data)


class IngestConfig(SimpleModel):
    event_group_size: int = 2


class AnnotationConfig(SimpleModel):
    adjudication_fraction: float = 0.15
    llm_weight: float = 0.6
    icc_threshold: float = 0.6
    alpha_threshold: float = 0.7
    max_remote_annotations: int = 64
    provider: ProviderConfig = Field(default_factory=ProviderConfig)


class FeatureConfig(SimpleModel):
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    surprisal_model: str = "gpt2-medium"
    semantic_dims: int = 16
    use_transformer_models: bool = False
    min_duration_seconds: float = 0.25


class ModelingConfig(SimpleModel):
    lags: list[int] = Field(default_factory=lambda: [0, 1, 2])
    alphas: list[float] = Field(default_factory=lambda: [0.1, 1.0, 10.0, 100.0])
    permutation_count: int = 25
    random_seed: int = 7
    max_units_per_dataset: int = 64
    max_subject_unit_groups: int = 512


class AgentConfig(SimpleModel):
    variants: list[str] = Field(default_factory=lambda: ["A0", "A1", "A2", "A3", "A4", "A5"])
    max_remote_decisions: int = 64
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    decoding_overrides: dict[str, dict[str, float]] = Field(default_factory=dict)


class AiMatrixConfig(SimpleModel):
    datasets: list[str] = Field(
        default_factory=lambda: [
            "narratives_formal",
            "ds000210_self_loop",
            "ds001618_self_loop",
            "ds001882_choice",
            "ds000212_choice",
            "ds004917_uncertainty",
            "ds004042_recall",
        ]
    )
    smoke_datasets: list[str] = Field(default_factory=lambda: ["ds001882_choice", "narratives_formal"])
    variants: list[str] = Field(default_factory=lambda: ["A0", "A1", "A2", "A3", "A4", "A5", "A4_no_reason", "A4_no_veto"])
    smoke_variants: list[str] = Field(default_factory=lambda: ["A4", "A5", "A4_no_reason", "A4_no_veto"])
    sample_size: int = 500
    smoke_sample_size: int = 10
    seeds: list[int] = Field(default_factory=lambda: [1, 2, 3])
    smoke_seeds: list[int] = Field(default_factory=lambda: [1])
    max_workers: int = 16
    smoke_max_workers: int = 4
    chunk_size: int = 50
    smoke_chunk_size: int = 10
    hard_call_cap: int = 84000
    hard_token_cap: int = 40000000
    estimated_tokens_per_call: int = 300
    bootstrap_samples: int = 1000
    random_seed: int = 7


class TrackBConfig(SimpleModel):
    anchor_tasks: list[str] = Field(default_factory=lambda: ["pieman", "sherlock", "merlin", "milkyway"])
    max_subjects_per_task: int = 20
    annex_timeout_seconds: int = 90
    roi_names: list[str] = Field(
        default_factory=lambda: [
            "mPFC",
            "PCC_precuneus",
            "TPJ",
            "STS",
            "IFG",
            "dACC_preSMA",
            "anterior_insula",
            "hippocampus",
        ]
        )


class DatasetManifest(SimpleModel):
    dataset_id: str
    adapter: str
    source_root: str
    task_family: str
    modality: str
    enabled: bool = True
    output_namespace: str
    split_strategy: str = "default"
    acquisition_mode: str = "local_existing"
    storage_root: str = ""
    download_strategy: str = "manual"
    required_markers: list[str] = Field(default_factory=list)
    active_phase: str = "active"
    derivatives_root: str = ""
    formal_enabled: bool = False
    formal_priority: int = 0
    preproc_strategy: str = "minimal_local_preproc"
    formal_derivatives_root: str = ""
    negative_control_tasks: list[str] = Field(default_factory=list)
    source_dataset_id: str = ""
    exploratory_enabled: bool = False
    exploratory_tier: str = ""
    scan_role: str = ""
    scan_mode: str = ""
    public_source: str = ""
    requires_preproc: bool = False
    max_scan_subjects: int = 0
    terminal_status_policy: str = "explicit_terminal_state"


class FormalMainlineConfig(SimpleModel):
    datasets: list[str] = Field(default_factory=lambda: ["narratives_full", "ds001882", "ds004042_recall"])
    remote_backend: str = "slurm_fmriprep"
    poll_interval_seconds: int = 300
    stall_threshold_minutes: int = 30
    max_retries: int = 2
    batch_size: int = 8
    max_active_jobs: int = 0
    refresh_on_dataset_completion: bool = True
    formal_report_name: str = "formal_mainline_report.md"
    monitor_report_name: str = "formal_mainline_monitor.md"
    figures_dir_name: str = "formal_figures"
    ssh_host: str = ""
    ssh_user: str = ""
    ssh_port: int = 22
    ssh_identity_file: str = ""
    ssh_remote_root: str = "/path/to/freewill-data"
    ssh_data_root: str = "/path/to/freewill-data/data"
    ssh_work_root: str = "/path/to/freewill-data/work"
    ssh_output_root: str = "/path/to/freewill-data/outputs"
    ssh_log_root: str = "/path/to/freewill-data/logs"
    ssh_fmriprep_image: str = "nipreps/fmriprep:latest"
    ssh_fs_license_file: str = "/path/to/freewill-data/licenses/freesurfer/license.txt"
    ssh_nthreads: int = 8
    ssh_omp_nthreads: int = 2
    ssh_mem_mb: int = 28000
    ssh_sync_raw: bool = True
    ssh_rsync_delete: bool = False
    ssh_command_timeout_seconds: int = 600
    ssh_rsync_timeout_seconds: int = 7200


class Tier1Config(SimpleModel):
    minimal_local_preproc_enabled: bool = True
    minimal_local_grid_shape: list[int] = Field(default_factory=lambda: [4, 4, 4])
    minimal_local_max_bold_files_per_dataset: int = 12
    minimal_local_max_timepoints_per_run: int = 180
    minimal_local_time_stride: int = 2
    minimal_local_trim_initial_trs: int = 0
    minimal_local_min_voxels_per_unit: int = 16
    datasets: list[DatasetManifest] = Field(default_factory=list)
    formal_report_name: str = "tier1_report.md"
    narratives_manifest_name: str = "anchor_manifest.tsv"
    parcel_output_name: str = "anchor_schaefer400_timeseries.parquet"
    openneuro_bootstrap_name: str = "openneuro_bootstrap.json"
    openneuro_prepare_name: str = "openneuro_prepare.parquet"
    openneuro_download_name: str = "openneuro_download.parquet"
    openneuro_verify_name: str = "openneuro_verify.parquet"
    openneuro_import_name: str = "openneuro_import.parquet"
    dataset_cards_name: str = "dataset_cards.parquet"
    missing_preproc_name: str = "missing_preproc.parquet"
    roi_output_name: str = "anchor_roi_summary.parquet"
    formal_mainline: FormalMainlineConfig = Field(default_factory=FormalMainlineConfig)
    formal_mainline_status_name: str = "formal_mainline_status.parquet"
    remote_preproc_jobs_name: str = "remote_preproc_jobs.parquet"
    formal_mainline_anomalies_name: str = "formal_mainline_anomalies.parquet"


class ExploratoryScanConfig(SimpleModel):
    performance_mode: str = "max_throughput"
    keep_remote_vm_running: bool = True
    max_parallel_downloads: int = 5
    max_parallel_imports: int = 8
    max_parallel_light_scans: int = 8
    max_parallel_fmriprep_jobs: int = 1
    reuse_existing_model_outputs: bool = True
    max_agent_events_per_dataset: int = 5000
    poll_interval_seconds: int = 300
    stall_threshold_minutes: int = 30
    max_retries: int = 2
    top_candidate_count: int = 3
    report_name: str = "exploratory_scan_report.md"
    candidate_report_name: str = "candidate_validation_report.md"
    figures_dir_name: str = "exploratory_figures"
    registry_name: str = "exploratory_dataset_registry.parquet"
    download_status_name: str = "exploratory_download_status.parquet"
    import_status_name: str = "exploratory_import_status.parquet"
    scan_results_name: str = "exploratory_scan_results.parquet"
    candidate_rankings_name: str = "exploratory_candidate_rankings.parquet"
    anomalies_name: str = "exploratory_anomalies.parquet"
    performance_status_name: str = "exploratory_performance_status.parquet"


class ProjectConfig(SimpleModel):
    run_name: str = "default"
    split_map: dict[str, str] = Field(default_factory=dict)
    paths: PathsConfig
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    annotation: AnnotationConfig = Field(default_factory=AnnotationConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    modeling: ModelingConfig = Field(default_factory=ModelingConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    ai_matrix: AiMatrixConfig = Field(default_factory=AiMatrixConfig)
    track_b: TrackBConfig = Field(default_factory=TrackBConfig)
    tier1: Tier1Config = Field(default_factory=Tier1Config)
    exploratory_scan: ExploratoryScanConfig = Field(default_factory=ExploratoryScanConfig)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _load_env_file_into_environment(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'").strip('"'))


def _coerce_env_override(value: str, current: Any) -> Any:
    if isinstance(current, bool):
        return value.lower() in {"1", "true", "yes", "on"}
    if isinstance(current, int) and not isinstance(current, bool):
        return int(value)
    if isinstance(current, float):
        return float(value)
    return value


def _apply_provider_env_overrides(merged: dict[str, Any], role: str) -> None:
    section = merged.setdefault(role, {})
    provider = section.setdefault("provider", {})
    field_names = [
        "mode",
        "model",
        "base_url",
        "endpoint_path",
        "api_key_env",
        "timeout_seconds",
        "temperature",
        "top_p",
        "json_mode",
        "fallback_model",
        "fallback_base_url",
        "fallback_endpoint_path",
        "fallback_api_key_env",
        "fallback_temperature",
        "fallback_top_p",
        "fallback_json_mode",
    ]
    for field_name in field_names:
        env_name = f"FREEWILL_{role.upper()}_{field_name.upper()}"
        if env_name not in os.environ:
            continue
        current = provider.get(field_name, getattr(ProviderConfig, field_name))
        provider[field_name] = _coerce_env_override(os.environ[env_name], current)
    provider["provider_role"] = role


def _apply_formal_mainline_env_overrides(merged: dict[str, Any]) -> None:
    formal = merged.setdefault("tier1", {}).setdefault("formal_mainline", {})
    defaults = FormalMainlineConfig()
    for field_name, current in defaults.model_dump().items():
        env_name = f"FREEWILL_FORMAL_{field_name.upper()}"
        if env_name in os.environ:
            formal[field_name] = _coerce_env_override(os.environ[env_name], current)


def _load_provider_layers(project_root: Path) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    example_path = project_root / "configs" / "providers.example.yaml"
    if example_path.exists():
        merged = deep_merge(merged, load_yaml(example_path))
    return merged


def _default_derivatives_root(paths: PathsConfig, dataset: DatasetManifest) -> str:
    raw_root = Path(paths.raw_root)
    if dataset.dataset_id == "narratives_full":
        return str(raw_root / "derivatives")
    return str(raw_root / "derivatives" / dataset.source_dataset_id)


def load_config(config_path: str | Path | None = None, paths_path: str | Path | None = None) -> ProjectConfig:
    project_root = Path(__file__).resolve().parents[2]
    default_path = project_root / "configs" / "default.yaml"
    default_paths = project_root / "configs" / "paths.yaml"
    runtime_paths = project_root / "configs" / "runtime_paths.yaml"
    datasets_path = project_root / "configs" / "datasets.yaml"
    provider_local_path = project_root / "configs" / "providers.local.yaml"
    provider_env_path = Path.home() / ".config" / "freewill" / "providers.env"

    _load_env_file_into_environment(provider_env_path)

    merged = _load_provider_layers(project_root)
    merged = deep_merge(merged, load_yaml(default_path))
    if provider_local_path.exists():
        merged = deep_merge(merged, load_yaml(provider_local_path))
    merged["paths"] = load_yaml(default_paths)
    if runtime_paths.exists():
        merged["paths"] = deep_merge(merged["paths"], load_yaml(runtime_paths))
    if datasets_path.exists():
        merged = deep_merge(merged, load_yaml(datasets_path))

    if paths_path:
        merged["paths"] = deep_merge(merged["paths"], load_yaml(paths_path))
    if config_path:
        merged = deep_merge(merged, load_yaml(config_path))

    env_project_root = os.getenv("FREEWILL_PROJECT_ROOT")
    if env_project_root:
        merged["paths"]["project_root"] = env_project_root

    _apply_provider_env_overrides(merged, "annotation")
    _apply_provider_env_overrides(merged, "agent")
    _apply_formal_mainline_env_overrides(merged)

    config = ProjectConfig(**merged)
    config.paths = config.paths.resolve()
    config.annotation.provider.provider_role = "annotation"
    config.agent.provider.provider_role = "agent"
    for dataset in config.tier1.datasets:
        if not dataset.source_dataset_id:
            dataset.source_dataset_id = dataset.dataset_id
        if not dataset.storage_root:
            dataset.storage_root = dataset.source_root
        if not dataset.derivatives_root:
            dataset.derivatives_root = _default_derivatives_root(config.paths, dataset)
        if not dataset.formal_derivatives_root:
            dataset.formal_derivatives_root = dataset.derivatives_root
        if not dataset.required_markers:
            dataset.required_markers = [
                "dataset_description.json",
                "sub-*",
                "*_bold.nii.gz",
                "*.json",
            ]
    return config

