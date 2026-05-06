# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.latent.config import (
    get_latent_reasoning_backend_spec,
    latent_reasoning_capture_env_names,
    normalize_latent_reasoning_config,
)


def test_backend_registry_exposes_qwen35_defaults():
    spec = get_latent_reasoning_backend_spec("qwen35_mtp")

    assert spec.default_think_close_token_id == 248069
    assert spec.default_max_internal_tokens == 1200
    assert spec.capture_inputs_embeds_env == "LATENT_QWEN35_CAPTURE_INPUTS_EMBEDS"
    assert (
        spec.adapter_qualname
        == "vllm.model_executor.models.qwen3_5_latent_mtp.Qwen3_5LatentMTP"
    )
    assert latent_reasoning_capture_env_names({"qwen35_mtp"}) == {
        "LATENT_QWEN35_CAPTURE_INPUTS_EMBEDS"
    }


def test_normalize_latent_reasoning_config():
    config = normalize_latent_reasoning_config(
        {
            "latent_reasoning": {
                "backend": "qwen35_mtp",
                "checkpoint": "/tmp/head.pt",
                "think_close_token_id": "248069",
                "max_internal_tokens": "3000",
            }
        }
    )

    assert config == {
        "backend": "qwen35_mtp",
        "checkpoint": "/tmp/head.pt",
        "think_close_token_id": 248069,
        "max_internal_tokens": 3000,
    }


def test_normalize_legacy_qwen35_config():
    config = normalize_latent_reasoning_config(
        {
            "latent_qwen35": {
                "checkpoint": "/tmp/head.pt",
            }
        }
    )

    assert config == {
        "backend": "qwen35_mtp",
        "checkpoint": "/tmp/head.pt",
        "think_close_token_id": 248069,
        "max_internal_tokens": 1200,
    }


def test_normalize_latent_reasoning_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unsupported latent reasoning backend"):
        normalize_latent_reasoning_config(
            {
                "latent_reasoning": {
                    "backend": "other",
                    "checkpoint": "/tmp/head.pt",
                }
            }
        )
