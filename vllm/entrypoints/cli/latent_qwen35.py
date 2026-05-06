# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
from pathlib import Path

from vllm.entrypoints.cli.types import CLISubcommand


def _row_prompt(row: dict, field: str) -> str:
    if field in row:
        return str(row[field])
    for candidate in ("question", "prompt", "input", "text"):
        if candidate in row:
            return str(row[candidate])
    raise KeyError(f"No prompt field '{field}' or fallback question/prompt/input/text in row")


class LatentQwen35Subcommand(CLISubcommand):
    """Run Qwen3.5 latent-MTP generation from a local checkpoint."""

    name = "latent-qwen35"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        from vllm.latent.qwen3_5_mtp import (
            LatentGenerationConfig,
            Qwen35LatentMTPRuntime,
            read_jsonl,
            write_jsonl,
        )

        cfg = LatentGenerationConfig(
            model=args.model,
            checkpoint=args.checkpoint,
            dtype=args.dtype,
            device_map=args.device_map,
            attn_implementation=args.attn_implementation,
            trust_remote_code=not args.no_trust_remote_code,
            think_close_token_id=args.think_close_token_id,
            max_total_steps=args.max_total_steps,
            enable_thinking=not args.no_thinking,
            system_prompt=args.system_prompt,
        )
        runtime = Qwen35LatentMTPRuntime(cfg)

        rows: list[dict]
        if args.input_jsonl:
            rows = read_jsonl(Path(args.input_jsonl))
            if args.limit and args.limit > 0:
                rows = rows[: args.limit]
            prompts = [_row_prompt(row, args.prompt_field) for row in rows]
        else:
            rows = [{"prompt": args.prompt}]
            prompts = [args.prompt]

        out_rows: list[dict] = []
        for idx, (row, prompt) in enumerate(zip(rows, prompts, strict=True), start=1):
            result = runtime.generate(prompt)
            rec = dict(row)
            rec["latent_qwen35"] = result.to_dict()
            out_rows.append(rec)
            print(
                json.dumps(
                    {
                        "idx": idx,
                        "steps_total": result.steps_total,
                        "latent_steps": result.latent_steps,
                        "normal_steps": result.normal_steps,
                        "total_steps_per_s": result.total_steps_per_s,
                        "visible_tokens_per_s": result.visible_tokens_per_s,
                        "mode_exit": result.mode_exit,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

        if args.output_jsonl:
            write_jsonl(Path(args.output_jsonl), out_rows)
        else:
            print(json.dumps(out_rows[0], ensure_ascii=False, indent=2))

    def subparser_init(self, subparsers: argparse._SubParsersAction):
        from vllm.entrypoints.utils import VLLM_SUBCMD_PARSER_EPILOG

        parser = subparsers.add_parser(
            self.name,
            help="Run Qwen3.5 latent-MTP local generation.",
            description="Run latent-switch generation for latent-mimo Qwen3.5 checkpoints.",
            usage="vllm latent-qwen35 --model MODEL --checkpoint CKPT --prompt '...'",
        )
        parser.add_argument("--model", required=True, help="Path or HF id for Qwen3.5 base model")
        parser.add_argument("--checkpoint", required=True, help="Latent head checkpoint .pt")
        parser.add_argument(
            "--prompt",
            default="",
            help="Single prompt when --input-jsonl is omitted",
        )
        parser.add_argument("--input-jsonl", default=None, help="Optional input JSONL")
        parser.add_argument("--output-jsonl", default=None, help="Optional output JSONL")
        parser.add_argument(
            "--prompt-field",
            default="question",
            help="Prompt field for JSONL rows",
        )
        parser.add_argument("--limit", type=int, default=0, help="Optional row limit")
        parser.add_argument(
            "--dtype",
            choices=["bfloat16", "float16", "float32"],
            default="bfloat16",
        )
        parser.add_argument("--device-map", default="cuda")
        parser.add_argument(
            "--attn-implementation",
            default=None,
            help="HF attention implementation, e.g. flash_attention_2",
        )
        parser.add_argument("--think-close-token-id", type=int, default=248069)
        parser.add_argument("--max-total-steps", type=int, default=1200)
        parser.add_argument("--system-prompt", default="")
        parser.add_argument(
            "--no-thinking",
            action="store_true",
            help="Disable Qwen thinking chat template",
        )
        parser.add_argument("--no-trust-remote-code", action="store_true")
        parser.epilog = VLLM_SUBCMD_PARSER_EPILOG.format(subcmd=self.name)
        return parser


def cmd_init() -> list[CLISubcommand]:
    return [LatentQwen35Subcommand()]
