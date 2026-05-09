# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Latent reasoning utilities for experimental model forks."""

__all__ = [
    "LatentGenerationConfig",
    "LatentGenerationOutput",
    "Qwen35LatentMTPRuntime",
]


def __getattr__(name: str):
    if name in __all__:
        from vllm.latent.qwen3_5_mtp import (
            LatentGenerationConfig,
            LatentGenerationOutput,
            Qwen35LatentMTPRuntime,
        )

        values = {
            "LatentGenerationConfig": LatentGenerationConfig,
            "LatentGenerationOutput": LatentGenerationOutput,
            "Qwen35LatentMTPRuntime": Qwen35LatentMTPRuntime,
        }
        return values[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
