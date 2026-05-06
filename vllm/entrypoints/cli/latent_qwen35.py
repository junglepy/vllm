# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
import time
from pathlib import Path
from typing import Any

from vllm.entrypoints.cli.types import CLISubcommand


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _row_prompt(row: dict[str, Any], field: str) -> str:
    if field in row:
        return str(row[field])
    for candidate in ("question", "prompt", "input", "text"):
        if candidate in row:
            return str(row[candidate])
    raise KeyError(
        f"No prompt field '{field}' or fallback question/prompt/input/text in row"
    )


class LatentQwen35Subcommand(CLISubcommand):
    """Run Qwen3.5 latent-MTP generation through the vLLM engine."""

    name = "latent-qwen35"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        from transformers import AutoTokenizer

        from vllm import LLM, SamplingParams

        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            trust_remote_code=not args.no_trust_remote_code,
        )

        if args.input_jsonl:
            rows = _read_jsonl(Path(args.input_jsonl))
            if args.limit and args.limit > 0:
                rows = rows[: args.limit]
        else:
            rows = [{"prompt": args.prompt}]

        prompts: list[str] = []
        for row in rows:
            user_prompt = _row_prompt(row, args.prompt_field)
            messages = []
            if args.system_prompt:
                messages.append({"role": "system", "content": args.system_prompt})
            messages.append({"role": "user", "content": user_prompt})
            prompts.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=not args.no_thinking,
                )
            )

        llm = LLM(
            model=args.model,
            dtype=args.dtype,
            trust_remote_code=not args.no_trust_remote_code,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
            async_scheduling=False,
            enforce_eager=args.enforce_eager,
        )
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_visible_tokens,
            extra_args={
                "latent_qwen35": {
                    "checkpoint": args.checkpoint,
                    "think_close_token_id": args.think_close_token_id,
                    "max_internal_tokens": args.max_internal_tokens,
                }
            },
        )

        started = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params, use_tqdm=not args.no_tqdm)
        elapsed = time.perf_counter() - started

        out_rows: list[dict[str, Any]] = []
        total_visible = 0
        total_internal = 0
        for row, out in zip(rows, outputs, strict=True):
            completion = out.outputs[0]
            visible_tokens = len(completion.token_ids)
            internal_tokens = int(getattr(out, "latent_internal_token_count", 0))
            total_visible += visible_tokens
            total_internal += internal_tokens
            rec = dict(row)
            rec["latent_qwen35"] = {
                "text": completion.text,
                "token_ids": list(completion.token_ids),
                "visible_tokens": visible_tokens,
                "internal_tokens": internal_tokens,
                "total_steps": visible_tokens + internal_tokens,
                "finish_reason": completion.finish_reason,
                "stop_reason": completion.stop_reason,
            }
            out_rows.append(rec)

        summary = {
            "rows": len(out_rows),
            "elapsed_s": elapsed,
            "visible_tokens": total_visible,
            "internal_tokens": total_internal,
            "total_steps": total_visible + total_internal,
            "visible_tokens_per_s": total_visible / elapsed if elapsed > 0 else 0.0,
            "total_steps_per_s": (
                (total_visible + total_internal) / elapsed if elapsed > 0 else 0.0
            ),
        }
        print(json.dumps(summary, ensure_ascii=False), flush=True)

        if args.output_jsonl:
            _write_jsonl(Path(args.output_jsonl), out_rows)
        else:
            print(json.dumps(out_rows[0], ensure_ascii=False, indent=2))

    def subparser_init(self, subparsers: argparse._SubParsersAction):
        from vllm.entrypoints.utils import VLLM_SUBCMD_PARSER_EPILOG

        parser = subparsers.add_parser(
            self.name,
            help="Run Qwen3.5 latent-MTP generation through the vLLM engine.",
            description="Run latent-switch generation for latent-mimo Qwen3.5 checkpoints.",
            usage="vllm latent-qwen35 --model MODEL --checkpoint CKPT --prompt '...'",
        )
        parser.add_argument("--model", required=True, help="Path or HF id for Qwen3.5")
        parser.add_argument("--checkpoint", required=True, help="Latent head checkpoint .pt")
        parser.add_argument("--prompt", default="", help="Single prompt when --input-jsonl is omitted")
        parser.add_argument("--input-jsonl", default=None, help="Optional input JSONL")
        parser.add_argument("--output-jsonl", default=None, help="Optional output JSONL")
        parser.add_argument("--prompt-field", default="question", help="Prompt field for JSONL rows")
        parser.add_argument("--limit", type=int, default=0, help="Optional row limit")
        parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
        parser.add_argument("--max-model-len", type=int, default=4096)
        parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
        parser.add_argument("--tensor-parallel-size", type=int, default=1)
        parser.add_argument("--temperature", type=float, default=0.0)
        parser.add_argument("--top-p", type=float, default=1.0)
        parser.add_argument("--max-visible-tokens", type=int, default=1024)
        parser.add_argument("--max-internal-tokens", type=int, default=1200)
        parser.add_argument("--think-close-token-id", type=int, default=248069)
        parser.add_argument("--system-prompt", default="")
        parser.add_argument("--no-thinking", action="store_true", help="Disable Qwen thinking chat template")
        parser.add_argument("--no-trust-remote-code", action="store_true")
        parser.add_argument("--enforce-eager", action="store_true")
        parser.add_argument("--no-tqdm", action="store_true")
        parser.epilog = VLLM_SUBCMD_PARSER_EPILOG.format(subcmd=self.name)
        return parser


def cmd_init() -> list[CLISubcommand]:
    return [LatentQwen35Subcommand()]
