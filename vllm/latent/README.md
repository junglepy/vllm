# Qwen3.5 latent-MTP vLLM runtime

This fork adds a vLLM V1 decode path for latent-mimo Qwen3.5 checkpoints.

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

The vLLM integration uses `SamplingParams.extra_args["latent_qwen35"]`:

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
        "latent_qwen35": {
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

The output includes `latent_internal_token_count`. For latent mode, the relevant
throughput metric is usually:

```text
(visible output tokens + latent_internal_token_count) / elapsed_seconds
```

Known constraints in this branch:

- latent mode currently requires `async_scheduling=False` because the internal
  token bookkeeping updates worker state synchronously;
- the MTP head is loaded as a worker-side module from the latent-mimo `.pt`
  checkpoint;
- `max_internal_tokens` is enforced separately from vLLM `max_tokens`, because
  vLLM `max_tokens` counts visible output tokens only.

Smoke result on B200 with `checkpoint_step5500_NEW.pt`, compiled vLLM path,
`max_model_len=512`, `async_scheduling=False`:

```text
visible=13, internal=162, total_steps=175, warm total_steps_per_s ~= 67.9
```
