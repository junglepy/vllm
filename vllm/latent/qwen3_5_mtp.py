# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.5 latent-MTP runtime.

This module hosts checkpoints trained with the latent MTP head used by the
latent-mimo Qwen3.5 experiments. It intentionally mirrors the model-side MTP
structure already present in :mod:`vllm.model_executor.models.qwen3_5_mtp`:

* a one-layer Qwen3.5 MTP core consumes the previous token id and previous
  target hidden state;
* a learned ``to_embed`` projection maps the MTP hidden state into the target
  model embedding space;
* while latent mode is active, the projected embedding is fed back to the
  target backbone as an internal, non-visible reasoning step;
* when the target logits predict ``</think>``, generation switches back to
  ordinary token generation.

The runtime is kept separate from the production scheduler for now because the
public vLLM request scheduler assumes each decode slot corresponds to a token id
owned by the request. Latent steps deliberately advance the target KV cache with
continuous embeddings that must not be surfaced as request tokens. Keeping this
as a named vLLM latent runtime gives a stable integration surface while avoiding
ad-hoc monkey patches in downstream repos.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.masking_utils import create_causal_mask
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5RMSNorm,
)
from transformers.utils import logging as hf_logging


@dataclass(slots=True)
class LatentGenerationConfig:
    """Configuration for Qwen3.5 latent-MTP generation."""

    model: str | Path
    checkpoint: str | Path
    dtype: str = "bfloat16"
    device_map: str = "cuda"
    attn_implementation: str | None = None
    trust_remote_code: bool = True
    think_close_token_id: int = 248069
    max_total_steps: int = 1200
    enable_thinking: bool = True
    system_prompt: str = ""


@dataclass(slots=True)
class LatentGenerationOutput:
    """One latent generation result."""

    prompt: str
    generated_token_ids: list[int]
    generated_text: str
    mode_exit: str
    steps_total: int
    latent_steps: int
    normal_steps: int
    think_close_predicted: bool
    elapsed_s: float

    @property
    def visible_tokens_per_s(self) -> float:
        return len(self.generated_token_ids) / self.elapsed_s if self.elapsed_s > 0 else 0.0

    @property
    def latent_steps_per_s(self) -> float:
        return self.latent_steps / self.elapsed_s if self.elapsed_s > 0 else 0.0

    @property
    def total_steps_per_s(self) -> float:
        return self.steps_total / self.elapsed_s if self.elapsed_s > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["visible_tokens_per_s"] = self.visible_tokens_per_s
        out["latent_steps_per_s"] = self.latent_steps_per_s
        out["total_steps_per_s"] = self.total_steps_per_s
        return out


def torch_dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def get_text_backbone(causal_lm: nn.Module) -> nn.Module:
    if hasattr(causal_lm, "model") and hasattr(causal_lm.model, "language_model"):
        return causal_lm.model.language_model
    if hasattr(causal_lm, "model") and hasattr(causal_lm.model, "layers"):
        return causal_lm.model
    raise RuntimeError("Unable to find Qwen3.5 text backbone on loaded model")


class Qwen35MTPCore(nn.Module):
    """HF-compatible Qwen3.5 one-layer MTP core used by latent checkpoints."""

    def __init__(self, text_config, rotary_emb, embed_tokens: nn.Module):
        super().__init__()
        import copy

        cfg = copy.deepcopy(text_config)
        cfg.num_hidden_layers = 1
        cfg.full_attention_interval = 1
        cfg.layer_types = ["full_attention"]
        self.config = cfg

        self.embed_tokens = embed_tokens
        self.rotary_emb = rotary_emb
        self.fc = nn.Linear(2 * cfg.hidden_size, cfg.hidden_size, bias=False)
        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.pre_fc_norm_hidden = Qwen3_5RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.layer = Qwen3_5DecoderLayer(cfg, layer_idx=0)
        self.norm = Qwen3_5RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: DynamicCache | None = None,
        cache_position: torch.Tensor | None = None,
    ) -> torch.Tensor:
        input_embeds = self.embed_tokens(input_ids)
        x = torch.cat(
            [
                self.pre_fc_norm_embedding(input_embeds),
                self.pre_fc_norm_hidden(hidden_states),
            ],
            dim=-1,
        )
        x = self.fc(x)

        if cache_position is None:
            past_seen = 0 if past_key_values is None else past_key_values.get_seq_length()
            cache_position = torch.arange(past_seen, past_seen + x.shape[1], device=x.device)

        if attention_mask is not None and attention_mask.dim() == 2:
            text_position_ids = attention_mask.to(dtype=torch.long).cumsum(dim=-1) - 1
            text_position_ids = text_position_ids.clamp_min_(0)
            text_position_ids = text_position_ids[:, -x.shape[1] :]
        else:
            text_position_ids = cache_position.view(1, -1).expand(x.shape[0], -1)

        position_ids = text_position_ids.unsqueeze(0).expand(3, -1, -1)
        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=x,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )
        if isinstance(causal_mask, torch.Tensor):
            causal_mask = causal_mask.contiguous()
        position_embeddings = self.rotary_emb(x, position_ids)

        x = self.layer(
            hidden_states=x,
            position_embeddings=position_embeddings,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
        )
        return self.norm(x)


class Qwen35LatentMTPHead(nn.Module):
    """Latent head checkpoint module."""

    def __init__(self, text_config, rotary_emb, embed_tokens: nn.Module):
        super().__init__()
        self.core = Qwen35MTPCore(text_config, rotary_emb, embed_tokens)
        hidden_size = int(text_config.hidden_size)
        emb_dim = int(embed_tokens.weight.shape[1])
        self.to_embed = nn.Linear(hidden_size, emb_dim, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.to_embed(self.core(input_ids, hidden_states, attention_mask))


def _chat_input_ids(
    tokenizer,
    question: str,
    system: str,
    enable_thinking: bool,
    device: torch.device,
) -> torch.Tensor:
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": question})
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        return_tensors="pt",
    )
    if isinstance(rendered, dict):
        input_ids = rendered["input_ids"]
    else:
        input_ids = rendered
    return input_ids.to(device)


class Qwen35LatentMTPRuntime:
    """Convenience runtime for local latent-MTP Qwen3.5 checkpoints."""

    def __init__(self, config: LatentGenerationConfig):
        hf_logging.disable_progress_bar()
        hf_logging.set_verbosity_error()
        self.config = config
        self.model_path = Path(config.model)
        self.checkpoint_path = Path(config.checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=config.trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch_dtype(config.dtype)
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": config.trust_remote_code,
            "device_map": config.device_map,
        }
        if config.attn_implementation:
            model_kwargs["attn_implementation"] = config.attn_implementation
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                dtype=dtype,
                **model_kwargs,
            )
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=dtype,
                **model_kwargs,
            )
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.text_backbone = get_text_backbone(self.model)
        self.latent_head = self._build_latent_head()
        self._load_latent_checkpoint()
        self.latent_head.eval()

    def _build_latent_head(self) -> Qwen35LatentMTPHead:
        head = Qwen35LatentMTPHead(
            text_config=self.text_backbone.config,
            rotary_emb=self.text_backbone.rotary_emb,
            embed_tokens=self.text_backbone.embed_tokens,
        )
        return head.to(device=self.device, dtype=next(self.model.parameters()).dtype)

    def _load_latent_checkpoint(self) -> None:
        ckpt = torch.load(self.checkpoint_path, map_location="cpu")
        state = ckpt.get("head_state_dict", ckpt)
        missing, unexpected = self.latent_head.load_state_dict(state, strict=False)
        missing = [x for x in missing if not x.startswith("core.embed_tokens.")]
        if missing or unexpected:
            raise RuntimeError(
                "Latent checkpoint key mismatch: "
                f"missing={missing[:20]} unexpected={unexpected[:20]}"
            )

    @torch.inference_mode()
    def generate(self, prompt: str, *, system: str | None = None) -> LatentGenerationOutput:
        start = time.perf_counter()
        cfg = self.config
        system_prompt = cfg.system_prompt if system is None else system
        input_ids = _chat_input_ids(
            self.tokenizer,
            prompt,
            system_prompt,
            cfg.enable_thinking,
            self.device,
        )
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=self.device)

        prefill = self.text_backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
        hidden = prefill.last_hidden_state
        past_kv = prefill.past_key_values
        hidden_last = hidden[:, -1:, :]
        logits_last = self.model.lm_head(hidden_last)[:, -1, :]

        mtp_cache = DynamicCache(config=self.text_backbone.config)
        cache_pos = torch.arange(input_ids.shape[1], device=self.device)
        _ = self.latent_head.core(
            input_ids=input_ids,
            hidden_states=hidden,
            attention_mask=attention_mask,
            past_key_values=mtp_cache,
            cache_position=cache_pos,
        )

        generated_ids: list[int] = []
        latent_steps = 0
        normal_steps = 0
        steps_total = 0
        think_close_predicted = False
        mode = "latent"
        cur_pos = int(input_ids.shape[1])
        eos_token_id = self.tokenizer.eos_token_id

        while steps_total < int(cfg.max_total_steps):
            next_id = int(torch.argmax(logits_last, dim=-1).item())
            next_ids = torch.tensor([[next_id]], dtype=torch.long, device=self.device)

            if mode == "latent" and next_id != int(cfg.think_close_token_id):
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.ones(
                            (1, 1),
                            dtype=attention_mask.dtype,
                            device=self.device,
                        ),
                    ],
                    dim=1,
                )
                mtp_hidden = self.latent_head.core(
                    input_ids=next_ids,
                    hidden_states=hidden_last,
                    attention_mask=attention_mask,
                    past_key_values=mtp_cache,
                    cache_position=torch.tensor([cur_pos], dtype=torch.long, device=self.device),
                )
                latent_embed = self.latent_head.to_embed(mtp_hidden)
                out_step = self.text_backbone(
                    inputs_embeds=latent_embed,
                    attention_mask=attention_mask,
                    past_key_values=past_kv,
                    use_cache=True,
                )
                past_kv = out_step.past_key_values
                hidden_last = out_step.last_hidden_state[:, -1:, :]
                logits_last = self.model.lm_head(hidden_last)[:, -1, :]
                latent_steps += 1
                steps_total += 1
                cur_pos += 1
                continue

            if mode == "latent":
                think_close_predicted = True
                mode = "normal"

            generated_ids.append(next_id)
            normal_steps += 1
            steps_total += 1
            if eos_token_id is not None and next_id == int(eos_token_id):
                break

            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (1, 1),
                        dtype=attention_mask.dtype,
                        device=self.device,
                    ),
                ],
                dim=1,
            )
            out_step = self.text_backbone(
                input_ids=next_ids,
                attention_mask=attention_mask,
                past_key_values=past_kv,
                use_cache=True,
            )
            past_kv = out_step.past_key_values
            hidden_last = out_step.last_hidden_state[:, -1:, :]
            logits_last = self.model.lm_head(hidden_last)[:, -1, :]
            cur_pos += 1

        elapsed = time.perf_counter() - start
        mode_exit = (
            "eos"
            if generated_ids
            and eos_token_id is not None
            and generated_ids[-1] == int(eos_token_id)
            else "max_steps"
        )
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
        return LatentGenerationOutput(
            prompt=prompt,
            generated_token_ids=generated_ids,
            generated_text=text,
            mode_exit=mode_exit,
            steps_total=steps_total,
            latent_steps=latent_steps,
            normal_steps=normal_steps,
            think_close_predicted=think_close_predicted,
            elapsed_s=elapsed,
        )

    def generate_many(self, prompts: Sequence[str]) -> list[LatentGenerationOutput]:
        return [self.generate(prompt) for prompt in prompts]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
