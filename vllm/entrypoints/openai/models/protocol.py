# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from dataclasses import dataclass

from vllm.latent.config import QWEN35_MTP_BACKEND, get_latent_reasoning_backend_spec


@dataclass
class BaseModelPath:
    name: str
    model_path: str


@dataclass
class LoRAModulePath:
    name: str
    path: str
    base_model_name: str | None = None


@dataclass
class LatentReasoningModulePath:
    name: str
    path: str
    base_model_name: str | None = None
    backend: str = QWEN35_MTP_BACKEND
    think_close_token_id: int | None = None
    max_internal_tokens: int | None = None

    def __post_init__(self) -> None:
        spec = get_latent_reasoning_backend_spec(self.backend)
        if self.think_close_token_id is None:
            self.think_close_token_id = spec.default_think_close_token_id
        if self.max_internal_tokens is None:
            self.max_internal_tokens = spec.default_max_internal_tokens

    def to_extra_args(self) -> dict[str, int | str]:
        assert self.think_close_token_id is not None
        assert self.max_internal_tokens is not None
        return {
            "backend": self.backend,
            "checkpoint": self.path,
            "think_close_token_id": self.think_close_token_id,
            "max_internal_tokens": self.max_internal_tokens,
        }


@dataclass
class LatentQwen35ModulePath:
    """Backward-compatible CLI shape for the first latent reasoning backend."""

    name: str
    path: str
    base_model_name: str | None = None
    think_close_token_id: int = 248069
    max_internal_tokens: int = 1200

    def to_extra_args(self) -> dict[str, int | str]:
        return LatentReasoningModulePath(
            name=self.name,
            path=self.path,
            base_model_name=self.base_model_name,
            backend=QWEN35_MTP_BACKEND,
            think_close_token_id=self.think_close_token_id,
            max_internal_tokens=self.max_internal_tokens,
        ).to_extra_args()
