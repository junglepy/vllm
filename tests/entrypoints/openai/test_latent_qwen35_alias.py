# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from argparse import Namespace
from types import SimpleNamespace

import pytest

from vllm.entrypoints.openai.api_server import _enable_latent_qwen35_defaults
from vllm.entrypoints.openai.engine.serving import OpenAIServing
from vllm.entrypoints.openai.models.protocol import LatentQwen35ModulePath
from vllm.sampling_params import BeamSearchParams, SamplingParams


class _FakeModels:
    def __init__(self, cfg: dict[str, int | str] | None):
        self.cfg = cfg

    def latent_qwen35_extra_args(
        self,
        model_name: str | None,
    ) -> dict[str, int | str] | None:
        return self.cfg if model_name == "latent-step5500" else None


def _serving_with_latent_cfg(cfg: dict[str, int | str] | None) -> OpenAIServing:
    serving = OpenAIServing.__new__(OpenAIServing)
    serving.models = _FakeModels(cfg)
    return serving


def test_latent_qwen35_alias_injects_sampling_extra_args():
    cfg = {
        "checkpoint": "/path/to/step5500.pt",
        "think_close_token_id": 248069,
        "max_internal_tokens": 3000,
    }
    serving = _serving_with_latent_cfg(cfg)
    request = SimpleNamespace(model="latent-step5500")
    params = SamplingParams(max_tokens=8)

    out = serving._apply_latent_qwen35_alias(request, params)

    assert out is params
    assert out.extra_args == {"latent_qwen35": cfg}


def test_latent_qwen35_alias_preserves_non_conflicting_extra_args():
    cfg = {
        "checkpoint": "/path/to/step5500.pt",
        "think_close_token_id": 248069,
        "max_internal_tokens": 1200,
    }
    serving = _serving_with_latent_cfg(cfg)
    request = SimpleNamespace(model="latent-step5500")
    params = SamplingParams(
        max_tokens=8,
        extra_args={"trace_id": "abc", "latent_qwen35": dict(cfg)},
    )

    out = serving._apply_latent_qwen35_alias(request, params)

    assert out.extra_args == {"trace_id": "abc", "latent_qwen35": cfg}


def test_latent_qwen35_alias_rejects_conflicting_extra_args():
    cfg = {
        "checkpoint": "/path/to/step5500.pt",
        "think_close_token_id": 248069,
        "max_internal_tokens": 1200,
    }
    serving = _serving_with_latent_cfg(cfg)
    request = SimpleNamespace(model="latent-step5500")
    params = SamplingParams(
        max_tokens=8,
        extra_args={
            "latent_qwen35": {
                "checkpoint": "/other.pt",
                "think_close_token_id": 248069,
                "max_internal_tokens": 1200,
            }
        },
    )

    with pytest.raises(ValueError, match="conflicts with model alias"):
        serving._apply_latent_qwen35_alias(request, params)


def test_latent_qwen35_alias_rejects_beam_search():
    cfg = {
        "checkpoint": "/path/to/step5500.pt",
        "think_close_token_id": 248069,
        "max_internal_tokens": 1200,
    }
    serving = _serving_with_latent_cfg(cfg)
    request = SimpleNamespace(model="latent-step5500")

    with pytest.raises(ValueError, match="do not support beam search"):
        serving._apply_latent_qwen35_alias(
            request,
            BeamSearchParams(beam_width=2, max_tokens=8),
        )


def test_latent_qwen35_module_to_extra_args():
    module = LatentQwen35ModulePath(
        name="latent-step5500",
        path="/path/to/step5500.pt",
        think_close_token_id=248069,
        max_internal_tokens=3000,
    )

    assert module.to_extra_args() == {
        "checkpoint": "/path/to/step5500.pt",
        "think_close_token_id": 248069,
        "max_internal_tokens": 3000,
    }


def test_latent_qwen35_server_sets_capture_env(monkeypatch):
    monkeypatch.delenv("LATENT_QWEN35_CAPTURE_INPUTS_EMBEDS", raising=False)
    args = Namespace(
        async_scheduling=True,
        latent_qwen35_modules=[
            LatentQwen35ModulePath(
                name="latent-step5500",
                path="/path/to/step5500.pt",
            )
        ],
    )

    _enable_latent_qwen35_defaults(args)

    assert os.environ["LATENT_QWEN35_CAPTURE_INPUTS_EMBEDS"] == "1"
    assert args.async_scheduling is False


def test_latent_qwen35_server_preserves_existing_capture_env(monkeypatch):
    monkeypatch.setenv("LATENT_QWEN35_CAPTURE_INPUTS_EMBEDS", "0")
    args = Namespace(
        async_scheduling=None,
        latent_qwen35_modules=[
            LatentQwen35ModulePath(
                name="latent-step5500",
                path="/path/to/step5500.pt",
            )
        ],
    )

    _enable_latent_qwen35_defaults(args)

    assert os.environ["LATENT_QWEN35_CAPTURE_INPUTS_EMBEDS"] == "0"
    assert args.async_scheduling is False
