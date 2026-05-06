# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Qwen3.5 latent-MTP head.

This module hosts the latent-mimo Qwen3.5 head in the same model-executor layer
as vLLM's built-in Qwen3.5 MTP speculative head. The head consumes the previous
token id plus the target model hidden state, then projects the resulting MTP
hidden state back into the target embedding space for an internal latent decode
step.
"""

from collections.abc import Iterable

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
    _require_is_multimodal,
)
from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MultiTokenPredictor
from vllm.model_executor.models.utils import (
    _merge_multimodal_embeddings,
    is_pp_missing_parameter,
)
from vllm.sequence import IntermediateTensors

logger = init_logger(__name__)


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
        "hidden_states": 0,
    }
)
class Qwen3_5LatentMTP(nn.Module, SupportsMultiModal):
    """Qwen3.5 latent-MTP head used by latent-mimo checkpoints.

    The training checkpoint stores weights under ``core.*`` and ``to_embed.*``.
    The vLLM executor keeps the one-layer MTP core under ``model.*`` to match the
    existing Qwen3.5 MTP implementation and exposes ``compute_latent_embeds`` for
    the latent decode loop.
    """

    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.config = config
        self.vllm_config = vllm_config
        self.model = Qwen3_5MultiTokenPredictor(
            vllm_config=vllm_config,
            prefix=f"{prefix}.model" if prefix else "model",
        )
        self.to_embed = ColumnParallelLinear(
            config.hidden_size,
            config.hidden_size,
            gather_output=True,
            bias=False,
            return_bias=False,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.to_embed" if prefix else "to_embed",
        )

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self.model.embed_input_ids(input_ids)

        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds

        is_multimodal = _require_is_multimodal(is_multimodal)
        return _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(
            input_ids=input_ids,
            positions=positions,
            hidden_states=hidden_states,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_latent_embeds(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        if not get_pp_group().is_last_rank:
            return None
        return self.to_embed(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in self._remap_latent_checkpoint_weights(weights):
            if "rotary_emb.inv_freq" in name:
                continue
            if name.endswith(".bias") and name not in params_dict:
                continue
            if is_pp_missing_parameter(name, self):
                continue
            if name not in params_dict:
                logger.warning_once(
                    "Parameter %s not found in Qwen3_5LatentMTP, skip loading",
                    name,
                )
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)

        return loaded_params

    @staticmethod
    def _remap_latent_checkpoint_weights(
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterable[tuple[str, torch.Tensor]]:
        for name, weight in weights:
            if name.startswith("core."):
                name = "model." + name[len("core.") :]
                name = name.replace("model.layer.", "model.layers.0.")
            elif name.startswith("to_embed."):
                pass
            elif name.startswith("latent_head.core."):
                name = "model." + name[len("latent_head.core.") :]
                name = name.replace("model.layer.", "model.layers.0.")
            elif name.startswith("latent_head.to_embed."):
                name = "to_embed." + name[len("latent_head.to_embed.") :]
            else:
                continue
            yield name, weight
