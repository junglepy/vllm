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
    parser.add_argument(
        "--skip-base",
        action="store_true",
        help="Only check the latent alias. By default both base and latent "
        "aliases are exercised through the HTTP API.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def _run_chat(
    *,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
) -> tuple[dict[str, Any], float]:
    req_started = time.monotonic()
    response = _http_json(
        "POST",
        f"{base_url}/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": max_tokens,
        },
        timeout=300.0,
    )
    return response, time.monotonic() - req_started


def _row_from_response(
    *,
    index: int,
    model: str,
    mode: str,
    prompt: str,
    response: dict[str, Any],
    elapsed_s: float,
) -> dict[str, Any]:
    usage = response.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    reasoning_tokens = int(details.get("reasoning_tokens") or 0)
    content = response.get("choices", [{}])[0].get("message", {}).get("content")
    return {
        "index": index,
        "mode": mode,
        "model": model,
        "prompt": prompt,
        "content": content,
        "usage": usage,
        "reasoning_tokens": reasoning_tokens,
        "elapsed_s": elapsed_s,
    }


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
            if args.base_alias not in model_ids:
                raise RuntimeError(
                    f"Base alias {args.base_alias!r} not listed in /v1/models."
                )
            if args.latent_alias not in model_ids:
                raise RuntimeError(
                    f"Latent alias {args.latent_alias!r} not listed in /v1/models."
                )

            prompts = _load_prompts(args.prompts_jsonl, args.limit)
            rows: list[dict[str, Any]] = []
            with predictions_path.open("w") as out_f:
                for idx, prompt in enumerate(prompts):
                    if not args.skip_base:
                        base_response, base_elapsed = _run_chat(
                            base_url=base_url,
                            model=args.base_alias,
                            prompt=prompt,
                            max_tokens=args.max_tokens,
                        )
                        base_row = _row_from_response(
                            index=idx,
                            model=args.base_alias,
                            mode="base",
                            prompt=prompt,
                            response=base_response,
                            elapsed_s=base_elapsed,
                        )
                        rows.append(base_row)
                        out_f.write(json.dumps(base_row, ensure_ascii=False) + "\n")
                        out_f.flush()

                    latent_response, latent_elapsed = _run_chat(
                        base_url=base_url,
                        model=args.latent_alias,
                        prompt=prompt,
                        max_tokens=args.max_tokens,
                    )
                    latent_row = _row_from_response(
                        index=idx,
                        model=args.latent_alias,
                        mode="latent",
                        prompt=prompt,
                        response=latent_response,
                        elapsed_s=latent_elapsed,
                    )
                    rows.append(latent_row)
                    out_f.write(json.dumps(latent_row, ensure_ascii=False) + "\n")
                    out_f.flush()
                    if latent_row["reasoning_tokens"] <= 0:
                        raise RuntimeError(
                            "Latent request returned zero reasoning_tokens; "
                            f"row={idx}, usage={latent_row['usage']}"
                        )

            elapsed_total = time.monotonic() - started_at
            latent_rows = [row for row in rows if row["mode"] == "latent"]
            base_rows = [row for row in rows if row["mode"] == "base"]
            summary = {
                "num_prompts": len(prompts),
                "num_rows": len(rows),
                "num_base_rows": len(base_rows),
                "num_latent_rows": len(latent_rows),
                "elapsed_s": elapsed_total,
                "mean_latent_reasoning_tokens": sum(
                    r["reasoning_tokens"] for r in latent_rows
                )
                / max(1, len(latent_rows)),
                "mean_latent_request_elapsed_s": sum(
                    r["elapsed_s"] for r in latent_rows
                )
                / max(1, len(latent_rows)),
                "mean_base_request_elapsed_s": sum(
                    r["elapsed_s"] for r in base_rows
                )
                / max(1, len(base_rows)),
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
