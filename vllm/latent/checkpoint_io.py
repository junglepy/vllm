# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint loading helpers for latent reasoning modules.

Latent-head checkpoints may be local PyTorch files, local safetensors files, or
Hugging Face Hub references. HF references intentionally remain strings until
load time so request/config normalization does not turn them into bogus local
paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

DEFAULT_LATENT_CHECKPOINT_FILENAMES = (
    "latent_head.safetensors",
    "checkpoint.safetensors",
    "model.safetensors",
    "adapter_model.safetensors",
    "latent_head.pt",
    "checkpoint.pt",
    "pytorch_model.bin",
)

LOCAL_CHECKPOINT_SUFFIXES = {
    ".pt",
    ".pth",
    ".bin",
    ".safetensors",
}


def is_probably_hf_checkpoint_ref(checkpoint: str | Path) -> bool:
    """Return True for HF Hub refs supported by this loader.

    Supported examples:
      - ``owner/repo``
      - ``owner/repo:latent_head.safetensors``
      - ``hf://owner/repo/latent_head.safetensors``

    Existing local files are always treated as local paths.
    """

    s = str(checkpoint)
    if not s or "://" in s and not s.startswith("hf://"):
        return False
    if Path(s).expanduser().exists():
        return False
    if s.startswith("hf://"):
        return True
    if ":" in s:
        # Windows drive paths are not relevant for the Linux worker. A colon is
        # the explicit repo:filename separator for latent checkpoints.
        repo, filename = s.split(":", 1)
        return bool(repo and filename and "/" in repo)
    if s.startswith(("/", "./", "../", "~")):
        return False
    parts = s.split("/")
    # Bare HF refs are owner/repo. Relative local paths such as
    # checkpoints/head.pt should remain local paths and fail locally if missing.
    return (
        len(parts) == 2
        and all(parts)
        and Path(parts[-1]).suffix.lower() not in LOCAL_CHECKPOINT_SUFFIXES
    )


def normalize_latent_checkpoint_ref(checkpoint: str | Path) -> str:
    """Normalize local paths, keep HF Hub refs intact."""

    s = str(checkpoint)
    if is_probably_hf_checkpoint_ref(s):
        return s
    return str(Path(s).expanduser().resolve())


def _parse_hf_ref(checkpoint: str) -> tuple[str, str | None]:
    if checkpoint.startswith("hf://"):
        body = checkpoint[len("hf://") :]
        parts = body.split("/", 2)
        if len(parts) < 2:
            raise ValueError(
                "HF latent checkpoint refs must look like "
                "hf://owner/repo[/filename]"
            )
        repo_id = "/".join(parts[:2])
        filename = parts[2] if len(parts) == 3 and parts[2] else None
        return repo_id, filename

    if ":" in checkpoint:
        repo_id, filename = checkpoint.split(":", 1)
        if not repo_id or not filename:
            raise ValueError(
                "HF latent checkpoint refs must look like owner/repo:filename"
            )
        return repo_id, filename

    return checkpoint, None


def resolve_latent_checkpoint_path(checkpoint: str | Path) -> Path:
    """Resolve a local or HF latent checkpoint reference to a local file path."""

    s = str(checkpoint)
    local_path = Path(s).expanduser()
    if local_path.exists():
        return local_path.resolve()

    if not is_probably_hf_checkpoint_ref(s):
        # Preserve the old failure mode for missing local files, but show the
        # absolute path users attempted to load.
        return local_path.resolve()

    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError
    except ImportError as exc:
        raise RuntimeError(
            "Loading latent-head checkpoints from Hugging Face requires "
            "huggingface_hub to be installed."
        ) from exc

    repo_id, filename = _parse_hf_ref(s)
    if filename:
        return Path(hf_hub_download(repo_id=repo_id, filename=filename))

    last_error: Exception | None = None
    for candidate in DEFAULT_LATENT_CHECKPOINT_FILENAMES:
        try:
            return Path(hf_hub_download(repo_id=repo_id, filename=candidate))
        except EntryNotFoundError as exc:
            last_error = exc
            continue

    raise FileNotFoundError(
        f"No default latent-head checkpoint file found in HF repo {repo_id!r}. "
        "Tried: " + ", ".join(DEFAULT_LATENT_CHECKPOINT_FILENAMES)
    ) from last_error


def load_latent_checkpoint(checkpoint: str | Path) -> tuple[dict[str, Any], Path]:
    """Load a latent-head checkpoint and return ``(raw_checkpoint, path)``."""

    path = resolve_latent_checkpoint_path(checkpoint)
    suffix = path.suffix.lower()
    if suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise RuntimeError(
                "Loading .safetensors latent-head checkpoints requires "
                "safetensors to be installed."
            ) from exc
        return dict(load_file(str(path), device="cpu")), path

    return torch.load(path, map_location="cpu"), path


def load_latent_head_state_dict(checkpoint: str | Path) -> tuple[dict[str, Any], Path]:
    """Load the state_dict stored by latent-mimo training or safetensors."""

    ckpt, path = load_latent_checkpoint(checkpoint)
    state = ckpt.get("head_state_dict", ckpt)
    if not isinstance(state, dict):
        raise TypeError(
            f"Latent checkpoint {path} does not contain a state_dict-compatible "
            f"object; got {type(state).__name__}."
        )
    return state, path
