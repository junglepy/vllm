# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Latent reasoning utilities for experimental model forks."""

from vllm.latent.qwen3_5_mtp import (
    LatentGenerationConfig,
    LatentGenerationOutput,
    Qwen35LatentMTPRuntime,
)

__all__ = [
    "LatentGenerationConfig",
    "LatentGenerationOutput",
    "Qwen35LatentMTPRuntime",
]
