from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _json_default(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, pd.Series):
            return value.tolist()
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=_json_default)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def write_parquet(frame: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _json_default(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, pd.Series):
            return value.tolist()
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if isinstance(value, set):
            return sorted(value)
        return str(value)

    def _flatten_nested_columns(source: pd.DataFrame) -> pd.DataFrame:
        converted = source.copy()
        for column in converted.columns:
            if converted[column].dtype != "object":
                continue
            has_nested = converted[column].map(lambda value: isinstance(value, (dict, list, tuple, set, np.ndarray))).any()
            if has_nested:
                converted[column] = converted[column].map(
                    lambda value: json.dumps(value, ensure_ascii=False, default=_json_default)
                    if isinstance(value, (dict, list, tuple, set, np.ndarray))
                    else value
                )
        return converted

    def _prepare_target() -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.is_file():
            path.unlink()

    def _write_frame_json(source: pd.DataFrame) -> None:
        json_path = path.with_suffix(".json")
        write_json(json_path, {"rows": source.to_dict(orient="records")})
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"json_fallback={json_path.name}\n", encoding="utf-8")
        except OSError:
            pass

    try:
        _prepare_target()
        frame.to_parquet(path, index=False)
    except Exception:
        try:
            _prepare_target()
            _flatten_nested_columns(frame).to_parquet(path, index=False)
        except Exception:
            try:
                _prepare_target()
                frame.to_pickle(path)
            except Exception:
                _write_frame_json(_flatten_nested_columns(frame))


def read_parquet(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    try:
        return pd.read_parquet(path)
    except Exception as parquet_error:
        try:
            return pd.read_pickle(path)
        except Exception:
            fallback_candidates = [path.with_suffix(".json")]
            if path.suffix:
                fallback_candidates.append(path.with_name(f"{path.stem}.json"))
            for candidate in fallback_candidates:
                if not candidate.exists():
                    continue
                payload = read_json(candidate)
                if isinstance(payload, dict) and "rows" in payload:
                    return pd.DataFrame(payload["rows"])
                if isinstance(payload, list):
                    return pd.DataFrame(payload)
            raise parquet_error


def zscore(values: pd.Series | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    std = array.std()
    if std == 0:
        return np.zeros_like(array)
    return (array - array.mean()) / std


def safe_json_loads(payload: str) -> dict[str, Any]:
    payload = payload.strip()
    if payload.startswith("```"):
        payload = payload.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(payload)

