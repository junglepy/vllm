# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run an OpenAI-compatible latent reasoning server smoke/mini-benchmark.

This script intentionally uses the HTTP API path instead of Python
``LLM.generate()`` so regressions in model aliases, request translation, and
usage accounting are caught in the production-facing path.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_PROMPTS = [
    "What is 12 + 30? Answer briefly.",
    "Janet has 9 apples and buys 7 more. How many apples does she have?",
]


def _http_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urlopen(req, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    return json.loads(body) if body else {}


def _wait_for_health(base_url: str, timeout_s: float, log_path: Path) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: str | None = None
    while time.monotonic() < deadline:
        try:
            with urlopen(f"{base_url}/health", timeout=5.0) as response:
                if response.status == 200:
                    return
        except (HTTPError, URLError, TimeoutError, ConnectionError) as exc:
            last_error = repr(exc)
        time.sleep(2.0)
    tail = ""
    if log_path.exists():
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-80:])
    raise RuntimeError(
        f"Server did not become healthy within {timeout_s:.0f}s. "
        f"Last error: {last_error}\nLast log lines:\n{tail}"
    )


def _load_prompts(path: Path | None, limit: int) -> list[str]:
    if path is None:
        prompts = list(DEFAULT_PROMPTS)
    else:
        prompts = []
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                prompt = row.get("prompt") or row.get("question")
                if not isinstance(prompt, str) or not prompt:
                    raise ValueError(
                        "Each JSONL row must contain a non-empty 'prompt' "
                        "or 'question' string."
                    )
                prompts.append(prompt)
                if len(prompts) >= limit:
                    break
    return prompts[:limit]


def _terminate(proc: subprocess.Popen[Any]) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=30)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-alias", default="qwen35-base")
    parser.add_argument("--latent-alias", default="qwen35-latent")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--max-internal-tokens", type=int, default=1200)
    parser.add_argument("--wait-timeout-s", type=float, default=600.0)
    parser.add_argument("--prompts-jsonl", type=Path)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / "server.log"
    models_path = args.out_dir / "models.json"
    predictions_path = args.out_dir / "predictions.jsonl"
    summary_path = args.out_dir / "summary.json"

    module = {
        "name": args.latent_alias,
        "path": str(Path(args.checkpoint).expanduser().resolve()),
        "backend": "qwen35_mtp",
        "max_internal_tokens": args.max_internal_tokens,
    }
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        args.model,
        "--served-model-name",
        args.base_alias,
        "--latent-reasoning-modules",
        json.dumps(module),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--dtype",
        args.dtype,
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
    ]
    env = os.environ.copy()
    env.setdefault("LATENT_QWEN35_CAPTURE_INPUTS_EMBEDS", "1")
    base_url = f"http://{args.host}:{args.port}"

    started_at = time.monotonic()
    with log_path.open("w") as log_f:
        proc = subprocess.Popen(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
        )
        try:
            _wait_for_health(base_url, args.wait_timeout_s, log_path)
            models = _http_json("GET", f"{base_url}/v1/models")
            models_path.write_text(json.dumps(models, indent=2) + "\n")
            model_ids = {item["id"] for item in models.get("data", [])}
            if args.latent_alias not in model_ids:
                raise RuntimeError(
                    f"Latent alias {args.latent_alias!r} not listed in /v1/models."
                )

            prompts = _load_prompts(args.prompts_jsonl, args.limit)
            rows: list[dict[str, Any]] = []
            with predictions_path.open("w") as out_f:
                for idx, prompt in enumerate(prompts):
                    req_started = time.monotonic()
                    response = _http_json(
                        "POST",
                        f"{base_url}/v1/chat/completions",
                        {
                            "model": args.latent_alias,
                            "messages": [{"role": "user", "content": prompt}],
                            "temperature": 0,
                            "max_tokens": args.max_tokens,
                        },
                        timeout=300.0,
                    )
                    elapsed = time.monotonic() - req_started
                    usage = response.get("usage") or {}
                    details = usage.get("completion_tokens_details") or {}
                    reasoning_tokens = int(details.get("reasoning_tokens") or 0)
                    content = (
                        response.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content")
                    )
                    row = {
                        "index": idx,
                        "prompt": prompt,
                        "content": content,
                        "usage": usage,
                        "reasoning_tokens": reasoning_tokens,
                        "elapsed_s": elapsed,
                    }
                    rows.append(row)
                    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out_f.flush()
                    if reasoning_tokens <= 0:
                        raise RuntimeError(
                            "Latent request returned zero reasoning_tokens; "
                            f"row={idx}, usage={usage}"
                        )

            elapsed_total = time.monotonic() - started_at
            summary = {
                "num_prompts": len(rows),
                "elapsed_s": elapsed_total,
                "mean_reasoning_tokens": sum(r["reasoning_tokens"] for r in rows)
                / max(1, len(rows)),
                "mean_request_elapsed_s": sum(r["elapsed_s"] for r in rows)
                / max(1, len(rows)),
                "latent_alias": args.latent_alias,
                "base_alias": args.base_alias,
                "checkpoint": module["path"],
            }
            summary_path.write_text(json.dumps(summary, indent=2) + "\n")
            print(json.dumps(summary, indent=2))
        finally:
            _terminate(proc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
