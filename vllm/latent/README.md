# Qwen3.5 latent-MTP runtime

This fork contains an experimental local runtime for latent-mimo Qwen3.5
checkpoints.

The runtime keeps the same latent-switch semantics used by the training/eval
scripts:

1. prefill the Qwen3.5 text backbone on the chat prompt;
2. run the trained one-layer MTP head on the prompt hidden states;
3. while the model is inside `<think>`, replace visible reasoning tokens with
   projected latent embeddings and advance the target KV cache internally;
4. when the next token predicted by the target logits is `</think>`, switch back
   to ordinary token generation and return only visible output tokens.

The entry point is:

```bash
vllm latent-qwen35 \
  --model /workspace/latent-mimo/qwen35_27b_tests/models/Qwen3.5-27B \
  --checkpoint /workspace/latent-mimo/deploy_archives/checkpoint_step1500_NEW.pt \
  --prompt "What is 2+2?" \
  --max-total-steps 1200
```

For JSONL batches:

```bash
vllm latent-qwen35 \
  --model /workspace/latent-mimo/qwen35_27b_tests/models/Qwen3.5-27B \
  --checkpoint /workspace/latent-mimo/deploy_archives/checkpoint_step1500_NEW.pt \
  --input-jsonl input.jsonl \
  --prompt-field question \
  --output-jsonl predictions.jsonl \
  --limit 50
```

The JSON output includes `total_steps_per_s`, `latent_steps_per_s`, and
`visible_tokens_per_s`. `total_steps_per_s` is the relevant throughput metric for
latent mode because latent steps advance the KV cache but are intentionally not
detokenized.

Current limitation: the first integration point is a named vLLM runtime module and
CLI. It does not yet use the production paged-attention scheduler, because vLLM's
request scheduler assumes every decode slot corresponds to a request-owned token
id, while this latent mode advances the target KV cache with continuous internal
embeddings. The module is structured so the model-side latent head can be moved
into the vLLM GPU model runner without changing checkpoint semantics.
