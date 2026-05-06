# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration helpers for latent reasoning backends.

This module is intentionally small and runtime-only: public entrypoints can
expose model aliases, while workers consume a normalized backend config without
knowing which CLI flag or legacy key produced it.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


LATENT_REASONING_EXTRA_ARGS_KEY = "latent_reasoning"
LEGACY_QWEN35_EXTRA_ARGS_KEY = "latent_qwen35"
QWEN35_MTP_BACKEND = "qwen35_mtp"
SUPPORTED_LATENT_REASONING_BACKENDS = frozenset({QWEN35_MTP_BACKEND})


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
    if backend not in SUPPORTED_LATENT_REASONING_BACKENDS:
        raise ValueError(
            "Unsupported latent reasoning backend "
            f"{backend!r}; supported backends: "
            f"{sorted(SUPPORTED_LATENT_REASONING_BACKENDS)}."
        )

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
        normalized.get("think_close_token_id", 248069)
    )
    normalized["max_internal_tokens"] = int(
        normalized.get("max_internal_tokens", 1200)
    )
    return normalized


def latent_reasoning_checkpoint(extra_args: Mapping[str, Any] | None) -> str | None:
    config = normalize_latent_reasoning_config(extra_args)
    return None if config is None else str(config["checkpoint"])
