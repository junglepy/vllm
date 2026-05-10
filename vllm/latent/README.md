# Qwen3.5 latent-MTP vLLM runtime

This fork adds a vLLM V1 decode path for latent-mimo Qwen3.5 checkpoints.
Publicly, latent-MTP is exposed as a native reasoning backend: HTTP clients use
ordinary model aliases and native reasoning usage fields. The special handling
is only in the execution path, where internal reasoning steps feed a continuous
MTP embedding into the next decode position instead of `embed_tokens(input_id)`.

Runtime semantics:

1. the request starts in latent mode inside `<think>`;
2. while latent mode is active and the sampled token is not `</think>`, vLLM
   treats the sampled token as an internal token: it advances sequence/KV state
   but is not returned to the client;
3. the trained one-layer MTP head maps `(sampled_token_id, target_hidden_state)`
   to a continuous embedding for the next decode position;
4. the next target-model step uses that latent embedding through vLLM's
   `inputs_embeds` path instead of the ordinary token embedding;
5. when `</think>` is sampled, generation switches back to normal visible token
   generation.

The offline Python integration still uses `SamplingParams.extra_args`, but the
preferred key is now the backend-agnostic
`SamplingParams.extra_args["latent_reasoning"]`. The older
`SamplingParams.extra_args["latent_qwen35"]` key remains accepted for backward
compatibility.

```python
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

model = "/workspace/latent-mimo/qwen35_27b_tests/models/Qwen3.5-27B"
checkpoint = "/workspace/latent-mimo/deploy_archives/checkpoint_step5500_NEW.pt"

tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
prompt = tok.apply_chat_template(
    [{"role": "user", "content": "What is 12 + 30?"}],
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,
)
llm = LLM(
    model=model,
    dtype="bfloat16",
    trust_remote_code=True,
    max_model_len=512,
    async_scheduling=False,
)
params = SamplingParams(
    temperature=0.0,
    max_tokens=64,
    extra_args={
        "latent_reasoning": {
            "backend": "qwen35_mtp",
            "checkpoint": checkpoint,
            "think_close_token_id": 248069,
            "max_internal_tokens": 1200,
        }
    },
)
out = llm.generate([prompt], params)[0]
print(out.outputs[0].text)
print(out.latent_internal_token_count)
```

Latent-head checkpoint references support local PyTorch checkpoints, local
`safetensors`, and Hugging Face Hub references:

```python
checkpoint = "/path/to/checkpoint.pt"
checkpoint = "/path/to/latent_head.safetensors"
checkpoint = "junglepy/qwen35-latent-v2"  # tries latent_head.safetensors first
checkpoint = "junglepy/qwen35-latent-v2:latent_head.safetensors"
checkpoint = "hf://junglepy/qwen35-latent-v2/latent_head.safetensors"
```

Recommended HF repo layout:

```text
latent_head.safetensors
latent_config.json
README.md
```

Multiple latent heads can be loaded in one `LLM` instance. The worker keeps a
head cache keyed by resolved checkpoint path, while each request keeps its own
MTP KV cache:

```python
for checkpoint in [step1500, step5500]:
    params = SamplingParams(
        temperature=0.0,
        max_tokens=64,
        extra_args={
            "latent_reasoning": {
                "backend": "qwen35_mtp",
                "checkpoint": checkpoint,
                "think_close_token_id": 248069,
                "max_internal_tokens": 1200,
            }
        },
    )
    out = llm.generate([prompt], params)[0]
```

OpenAI-compatible server aliases:

```bash
vllm serve /workspace/latent-mimo/qwen35_27b_tests/models/Qwen3.5-27B \
  --served-model-name qwen35-27b-base \
  --latent-reasoning-modules qwen35-27b-latent-step1500=/workspace/latent-mimo/deploy_archives/checkpoint_step1500_NEW.pt \
  --latent-reasoning-modules '{"name":"qwen35-27b-latent-step5500","path":"/workspace/latent-mimo/deploy_archives/checkpoint_step5500_NEW.pt","backend":"qwen35_mtp","max_internal_tokens":1200}'
```

Then clients can choose the execution mode via the standard `model` field:

```json
{"model": "qwen35-27b-base", "messages": [...]}
{"model": "qwen35-27b-latent-step5500", "messages": [...]}
```

The base alias uses ordinary vLLM decoding. A latent alias injects
`SamplingParams.extra_args["latent_reasoning"]` before scheduling the request.
`--latent-qwen35-modules` remains accepted as a deprecated alias for
`--latent-reasoning-modules` with `backend="qwen35_mtp"`.
When a latent reasoning module is provided, the OpenAI-compatible server sets
`LATENT_QWEN35_CAPTURE_INPUTS_EMBEDS=1` by default unless the operator already
set it explicitly, and forces `async_scheduling=False` for correctness with the
current worker-side latent bookkeeping.

CLI:

```bash
vllm latent-qwen35 \
  --model /workspace/latent-mimo/qwen35_27b_tests/models/Qwen3.5-27B \
  --checkpoint /workspace/latent-mimo/deploy_archives/checkpoint_step5500_NEW.pt \
  --prompt "What is 12 + 30?" \
  --max-model-len 512 \
  --max-visible-tokens 64 \
  --max-internal-tokens 1200
```

For JSONL batches:

```bash
vllm latent-qwen35 \
  --model /workspace/latent-mimo/qwen35_27b_tests/models/Qwen3.5-27B \
  --checkpoint /workspace/latent-mimo/deploy_archives/checkpoint_step5500_NEW.pt \
  --input-jsonl input.jsonl \
  --prompt-field question \
  --output-jsonl predictions.jsonl \
  --limit 50
```

The offline output includes `latent_internal_token_count`. OpenAI-compatible
HTTP responses expose the same latent steps through the native reasoning-token
usage fields:

- Chat/Completions: `usage.completion_tokens_details.reasoning_tokens`
- Responses: `usage.output_tokens_details.reasoning_tokens`

For latent mode, the relevant throughput metric is usually:

```text
(visible output tokens + latent_internal_token_count) / elapsed_seconds
```

Known constraints in this branch:

- latent mode currently requires `async_scheduling=False` because the internal
  token bookkeeping updates worker state synchronously. The dedicated
  `vllm latent-qwen35` CLI and OpenAI-compatible latent aliases set this
  automatically for backends that declare `supports_async_scheduling=False`;
- MTP heads are loaded lazily as worker-side modules from latent-mimo `.pt`
  checkpoints and cached by checkpoint path;
- `max_internal_tokens` is enforced separately from vLLM `max_tokens`, because
  vLLM `max_tokens` counts visible output tokens only.
- direct Python `LLM.generate()` calls still enable latent mode through
  `SamplingParams.extra_args["latent_reasoning"]`; the OpenAI-compatible server
  avoids request-side `extra_args` by using latent model aliases.

See `NATIVE_REASONING_DESIGN.md` for the intended native integration boundary:
vLLM reasoning APIs own public protocol/accounting, while latent backends own
the next-step embedding transition.

Smoke result on B200 with `checkpoint_step5500_NEW.pt`, compiled vLLM path,
`max_model_len=512`, `async_scheduling=False`:

```text
visible=13, internal=162, total_steps=175, warm total_steps_per_s ~= 67.9
```

OpenAI-compatible server check on the same checkpoint after the native pending
state cleanup:

```text
artifact: /workspace/latent-mimo/artifacts/qwen_vllm_latent_server_check_after_native_cleanup
base alias: qwen35-base
latent alias: qwen35-latent-step5500
latent reasoning_tokens: 175
latent content: </think>\n\n42
```

OpenAI-compatible server check when the operator explicitly requested
`--async-scheduling`:

```text
artifact: /workspace/latent-mimo/artifacts/qwen_vllm_latent_server_check_async_requested_override_retry
requested api_server arg: --async-scheduling
observed server log: overriding async_scheduling for correct latent next-embedding transitions
observed engine log: Asynchronous scheduling is disabled
latent reasoning_tokens: 139
latent content: </think>\n\n42
```
