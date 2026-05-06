# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration helpers for latent reasoning backends.

This module is intentionally small and runtime-only: public entrypoints can
expose model aliases, while workers consume a normalized backend config without
knowing which CLI flag or legacy key produced it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LATENT_REASONING_EXTRA_ARGS_KEY = "latent_reasoning"
LEGACY_QWEN35_EXTRA_ARGS_KEY = "latent_qwen35"
QWEN35_MTP_BACKEND = "qwen35_mtp"


@dataclass(frozen=True)
class LatentReasoningBackendSpec:
    name: str
    default_think_close_token_id: int
    default_max_internal_tokens: int
    capture_inputs_embeds_env: str | None = None
    adapter_module: str | None = None
    adapter_class: str | None = None
    adapter_prefix: str | None = None
    supported_model_types: tuple[str, ...] = ()
    supports_async_scheduling: bool = False

    @property
    def adapter_qualname(self) -> str | None:
        if not self.adapter_module or not self.adapter_class:
            return None
        return f"{self.adapter_module}.{self.adapter_class}"


LATENT_REASONING_BACKENDS: dict[str, LatentReasoningBackendSpec] = {
    QWEN35_MTP_BACKEND: LatentReasoningBackendSpec(
        name=QWEN35_MTP_BACKEND,
        default_think_close_token_id=248069,
        default_max_internal_tokens=1200,
        capture_inputs_embeds_env="LATENT_QWEN35_CAPTURE_INPUTS_EMBEDS",
        adapter_module="vllm.model_executor.models.qwen3_5_latent_mtp",
        adapter_class="Qwen3_5LatentMTP",
        adapter_prefix="latent_qwen35",
        supported_model_types=("qwen3_5", "qwen3_5_text"),
    )
}
SUPPORTED_LATENT_REASONING_BACKENDS = frozenset(LATENT_REASONING_BACKENDS)


def get_latent_reasoning_backend_spec(
    backend: str,
) -> LatentReasoningBackendSpec:
    try:
        return LATENT_REASONING_BACKENDS[backend]
    except KeyError as exc:
        raise ValueError(
            "Unsupported latent reasoning backend "
            f"{backend!r}; supported backends: "
            f"{sorted(SUPPORTED_LATENT_REASONING_BACKENDS)}."
        ) from exc


def latent_reasoning_capture_env_names(
    backends: set[str] | None = None,
) -> set[str]:
    names: set[str] = set()
    for backend, spec in LATENT_REASONING_BACKENDS.items():
        if backends is not None and backend not in backends:
            continue
        if spec.capture_inputs_embeds_env:
            names.add(spec.capture_inputs_embeds_env)
    return names


def latent_reasoning_requires_sync_scheduling(backends: set[str]) -> bool:
    return any(
        not get_latent_reasoning_backend_spec(backend).supports_async_scheduling
        for backend in backends
    )


def normalize_latent_reasoning_config(
    extra_args: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if not extra_args:
        return None

    latent_cfg = extra_args.get(LATENT_REASONING_EXTRA_ARGS_KEY)
    if latent_cfg is None:
        latent_cfg = extra_args.get(LEGACY_QWEN35_EXTRA_ARGS_KEY)
    if latent_cfg is None:
        return None
    if not isinstance(latent_cfg, dict):
        raise ValueError(
            "SamplingParams.extra_args['latent_reasoning'] must be a dict "
            "with a 'checkpoint' path."
        )

    backend = latent_cfg.get("backend", QWEN35_MTP_BACKEND)
    spec = get_latent_reasoning_backend_spec(str(backend))

    checkpoint = latent_cfg.get("checkpoint")
    if not checkpoint:
        raise ValueError(
            "SamplingParams.extra_args['latent_reasoning'] must include "
            "a non-empty 'checkpoint' path."
        )

    normalized = dict(latent_cfg)
    normalized["backend"] = str(backend)
    normalized["checkpoint"] = str(Path(str(checkpoint)).expanduser().resolve())
    normalized["think_close_token_id"] = int(
        normalized.get(
            "think_close_token_id",
            spec.default_think_close_token_id,
        )
    )
    normalized["max_internal_tokens"] = int(
        normalized.get("max_internal_tokens", spec.default_max_internal_tokens)
    )
    return normalized


def latent_reasoning_checkpoint(extra_args: Mapping[str, Any] | None) -> str | None:
    config = normalize_latent_reasoning_config(extra_args)
    return None if config is None else str(config["checkpoint"])
