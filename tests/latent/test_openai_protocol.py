# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.entrypoints.openai.models.protocol import LatentReasoningModulePath


def test_latent_reasoning_module_uses_backend_defaults():
    module = LatentReasoningModulePath(name="latent", path="/tmp/head.pt")

    assert module.to_extra_args() == {
        "backend": "qwen35_mtp",
        "checkpoint": "/tmp/head.pt",
        "think_close_token_id": 248069,
        "max_internal_tokens": 1200,
    }


def test_latent_reasoning_module_allows_backend_overrides():
    module = LatentReasoningModulePath(
        name="latent",
        path="/tmp/head.pt",
        backend="qwen35_mtp",
        think_close_token_id=1,
        max_internal_tokens=2,
    )

    assert module.to_extra_args()["think_close_token_id"] == 1
    assert module.to_extra_args()["max_internal_tokens"] == 2
