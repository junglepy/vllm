# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from argparse import Namespace

from vllm.entrypoints.openai.api_server import _enable_latent_reasoning_defaults
from vllm.entrypoints.openai.models.protocol import LatentReasoningModulePath


def test_latent_reasoning_server_defaults_set_capture_env_and_sync_scheduler(
    monkeypatch,
):
    monkeypatch.delenv("LATENT_QWEN35_CAPTURE_INPUTS_EMBEDS", raising=False)
    args = Namespace(
        latent_reasoning_modules=[
            LatentReasoningModulePath(name="latent", path="/tmp/head.pt")
        ],
        latent_qwen35_modules=None,
        async_scheduling=True,
    )

    _enable_latent_reasoning_defaults(args)

    assert args.async_scheduling is False
    assert os.environ["LATENT_QWEN35_CAPTURE_INPUTS_EMBEDS"] == "1"


def test_latent_reasoning_server_defaults_preserve_explicit_capture_env(
    monkeypatch,
):
    monkeypatch.setenv("LATENT_QWEN35_CAPTURE_INPUTS_EMBEDS", "0")
    args = Namespace(
        latent_reasoning_modules=[
            LatentReasoningModulePath(name="latent", path="/tmp/head.pt")
        ],
        latent_qwen35_modules=None,
        async_scheduling=False,
    )

    _enable_latent_reasoning_defaults(args)

    assert args.async_scheduling is False
    assert os.environ["LATENT_QWEN35_CAPTURE_INPUTS_EMBEDS"] == "0"
