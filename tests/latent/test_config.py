# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.latent.config import normalize_latent_reasoning_config


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
