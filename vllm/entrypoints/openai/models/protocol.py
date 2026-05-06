# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from dataclasses import dataclass


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
class LatentQwen35ModulePath:
    name: str
    path: str
    base_model_name: str | None = None
    think_close_token_id: int = 248069
    max_internal_tokens: int = 1200

    def to_extra_args(self) -> dict[str, int | str]:
        return {
            "checkpoint": self.path,
            "think_close_token_id": self.think_close_token_id,
            "max_internal_tokens": self.max_internal_tokens,
        }
