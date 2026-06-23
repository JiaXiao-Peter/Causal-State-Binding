from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

import requests

from freewill.config import ProviderConfig
from freewill.utils import safe_json_loads


@dataclass(frozen=True)
class ProviderResult:
    payload: dict[str, Any]
    metadata: dict[str, Any]


class ProviderClient:
    def __init__(self, config: ProviderConfig):
        self.config = config

    def _resolved_endpoint(self, base_url: str, endpoint_path: str) -> str:
        cleaned_base = (base_url or "").strip().rstrip("/")
        cleaned_endpoint = endpoint_path or "/chat/completions"
        if not cleaned_endpoint.startswith("/"):
            cleaned_endpoint = f"/{cleaned_endpoint}"
        if cleaned_base.endswith("/chat/completions") or cleaned_base.endswith("/responses"):
            return cleaned_base
        if cleaned_base.endswith("/v1"):
            return f"{cleaned_base}{cleaned_endpoint}"
        if "/v1/" in cleaned_base:
            return cleaned_base
        return f"{cleaned_base}/v1{cleaned_endpoint}" if cleaned_base else cleaned_endpoint

    def _request_settings(self, use_fallback: bool) -> dict[str, Any]:
        if not use_fallback:
            return {
                "mode": self.config.mode,
                "model": self.config.model,
                "base_url": self.config.base_url,
                "endpoint_path": self.config.endpoint_path,
                "api_key_env": self.config.api_key_env,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "json_mode": self.config.json_mode,
            }
        return {
            "mode": self.config.mode,
            "model": self.config.fallback_model or self.config.model,
            "base_url": self.config.fallback_base_url or self.config.base_url,
            "endpoint_path": self.config.fallback_endpoint_path or self.config.endpoint_path,
            "api_key_env": self.config.fallback_api_key_env or self.config.api_key_env,
            "temperature": self.config.fallback_temperature if self.config.fallback_temperature >= 0 else self.config.temperature,
            "top_p": self.config.fallback_top_p if self.config.fallback_top_p >= 0 else self.config.top_p,
            "json_mode": self.config.fallback_json_mode or self.config.json_mode,
        }

    def _metadata_template(self, *, use_fallback: bool, request_settings: dict[str, Any]) -> dict[str, Any]:
        return {
            "provider_role": self.config.provider_role,
            "model": request_settings["model"],
            "base_url": self._resolved_endpoint(request_settings["base_url"], request_settings["endpoint_path"]),
            "json_mode": request_settings["json_mode"],
            "request_temperature": request_settings.get("temperature"),
            "request_top_p": request_settings.get("top_p"),
            "fallback_used": use_fallback,
            "fallback_reason": "",
            "error_type": "",
            "latency_ms": 0,
            "remote_used": False,
        }

    def _is_remote_ready(self, request_settings: dict[str, Any]) -> bool:
        api_key_env = request_settings["api_key_env"]
        return request_settings["mode"] == "openai_compatible" and bool(api_key_env) and bool(os.getenv(api_key_env))

    def is_remote_ready(self, use_fallback: bool = False) -> bool:
        return self._is_remote_ready(self._request_settings(use_fallback))

    def _post_chat_completion(
        self,
        request_settings: dict[str, Any],
        *,
        system_prompt: str,
        user_prompt: str,
        temperature_override: float | None = None,
        top_p_override: float | None = None,
    ) -> ProviderResult:
        effective_settings = dict(request_settings)
        if temperature_override is not None:
            effective_settings["temperature"] = float(temperature_override)
        if top_p_override is not None:
            effective_settings["top_p"] = float(top_p_override)
        metadata = self._metadata_template(use_fallback=False, request_settings=effective_settings)
        endpoint = metadata["base_url"]
        headers = {
            "Authorization": f"Bearer {os.environ[effective_settings['api_key_env']]}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": effective_settings["model"],
            "temperature": effective_settings["temperature"],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if effective_settings.get("top_p") is not None:
            payload["top_p"] = effective_settings["top_p"]
        if effective_settings["json_mode"] == "json_object":
            payload["response_format"] = {"type": "json_object"}

        start = time.perf_counter()
        response = requests.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=self.config.timeout_seconds,
        )
        metadata["latency_ms"] = int((time.perf_counter() - start) * 1000)
        metadata["remote_used"] = True
        response.raise_for_status()
        response_payload = response.json()
        usage = response_payload.get("usage", {}) if isinstance(response_payload, dict) else {}
        if isinstance(usage, dict):
            metadata["usage"] = usage
            for key in ["prompt_tokens", "completion_tokens", "total_tokens"]:
                if key in usage:
                    metadata[key] = usage[key]
        content = response_payload["choices"][0]["message"]["content"]
        parsed = safe_json_loads(content) if isinstance(content, str) else json.loads(json.dumps(content))
        return ProviderResult(payload=parsed, metadata=metadata)

    def chat_json_with_metadata(
        self,
        system_prompt: str,
        user_prompt: str,
        default: dict[str, Any],
        *,
        temperature_override: float | None = None,
        top_p_override: float | None = None,
    ) -> ProviderResult:
        attempts = [(False, self._request_settings(False))]
        if self.config.fallback_model or self.config.fallback_base_url:
            attempts.append((True, self._request_settings(True)))

        primary_error = ""
        for use_fallback, request_settings in attempts:
            metadata = self._metadata_template(use_fallback=use_fallback, request_settings=request_settings)
            if use_fallback:
                metadata["fallback_reason"] = primary_error
            if not self._is_remote_ready(request_settings):
                metadata["error_type"] = "missing_api_key"
                if not use_fallback:
                    primary_error = metadata["error_type"]
                continue
            retryable_http = {408, 409, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524, 525}
            for retry_index in range(10):
                try:
                    result = self._post_chat_completion(
                        request_settings,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        temperature_override=temperature_override,
                        top_p_override=top_p_override,
                    )
                    if use_fallback:
                        result.metadata["fallback_used"] = True
                        result.metadata["fallback_reason"] = primary_error
                    result.metadata["provider_role"] = self.config.provider_role
                    result.metadata["retry_count"] = retry_index
                    return result
                except requests.HTTPError as exc:
                    status_code = exc.response.status_code if exc.response is not None else 0
                    metadata["error_type"] = f"http_{status_code if status_code else 'error'}"
                    if status_code not in retryable_http:
                        break
                except requests.RequestException:
                    metadata["error_type"] = "request_error"
                except (KeyError, ValueError, TypeError, json.JSONDecodeError):
                    metadata["error_type"] = "parse_error"
                    break
                if retry_index < 9:
                    time.sleep(min(16.0, 0.75 * (2**retry_index)))
            if not use_fallback:
                primary_error = metadata["error_type"]

        fallback_settings = self._request_settings(bool(primary_error and len(attempts) > 1))
        metadata = self._metadata_template(
            use_fallback=bool(primary_error and len(attempts) > 1),
            request_settings=fallback_settings,
        )
        metadata["error_type"] = primary_error or "offline_default"
        metadata["fallback_reason"] = primary_error if metadata["fallback_used"] else ""
        return ProviderResult(payload=default, metadata=metadata)

    def chat_json(self, system_prompt: str, user_prompt: str, default: dict[str, Any]) -> dict[str, Any]:
        return self.chat_json_with_metadata(system_prompt, user_prompt, default).payload

