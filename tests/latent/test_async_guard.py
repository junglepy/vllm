# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


def test_latent_reasoning_rejects_async_scheduling():
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.use_async_scheduling = True
    params = SamplingParams(
        max_tokens=1,
        extra_args={
            "latent_reasoning": {
                "checkpoint": "/tmp/head.pt",
            }
        },
    )

    with pytest.raises(ValueError, match="async_scheduling=False"):
        runner._get_latent_qwen35_config(params)


def test_latent_reasoning_config_allowed_for_sync_scheduling():
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.use_async_scheduling = False
    params = SamplingParams(
        max_tokens=1,
        extra_args={
            "latent_reasoning": {
                "checkpoint": "/tmp/head.pt",
            }
        },
    )

    config = runner._get_latent_qwen35_config(params)

    assert config is not None
    assert config["backend"] == "qwen35_mtp"
    assert config["checkpoint"] == "/tmp/head.pt"
