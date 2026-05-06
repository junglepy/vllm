# Latent reasoning as a native vLLM reasoning backend

Date: 2026-05-06
Branch: `feature/latent-mtp-integration`

## Decision

Latent-MTP should not expose a separate public protocol that competes with
vLLM reasoning mode. Publicly, latent inference is a reasoning backend:

- clients select a served model alias through the standard `model` field;
- hidden latent steps are reported as native reasoning usage;
- Chat/Completions use `usage.completion_tokens_details.reasoning_tokens`;
- Responses use `usage.output_tokens_details.reasoning_tokens`;
- visible text remains ordinary assistant content.

The extra logic belongs below the public reasoning/accounting layer: on internal
reasoning steps the next model input is a continuous embedding produced by an
MTP head, not `embed_tokens(input_id)`.

## Why existing reasoning parsers are not enough

vLLM reasoning parsers operate after token generation. They parse visible token
streams such as `<think>...</think>`, split reasoning/content, and count reasoning
tokens.

Latent-MTP changes the decode transition itself:

1. the frozen text backbone samples a token candidate;
2. while latent mode is active and the candidate is not the close-thinking token,
   the candidate is counted as an internal reasoning token and hidden from the
   client;
3. the MTP head maps `(sampled_token_id, target_hidden_state)` to a continuous
   `[batch, hidden_size]` embedding;
4. the next target-model step consumes that embedding through the `inputs_embeds`
   path.

A parser alone cannot do step 3 or replace the next-step input. Therefore the
right integration point is a native reasoning backend with an execution adapter,
not a text parser that rewrites output tokens.

## Current implementation status

Already in this branch:

- OpenAI-compatible server aliases via `--latent-reasoning-modules`;
- legacy `--latent-qwen35-modules` accepted as a compatibility alias;
- request-side `extra_args` is not required for HTTP clients;
- offline Python still uses `SamplingParams.extra_args["latent_reasoning"]`;
- backend registry exists in `vllm.latent.config` with `qwen35_mtp`;
- backend specs include the execution adapter entrypoint
  (`vllm.model_executor.models.qwen3_5_latent_mtp.Qwen3_5LatentMTP`) and own
  backend defaults for close token / max internal tokens;
- Qwen3.5 MTP head is loaded as a native vLLM module and cached by checkpoint;
- native usage fields report latent internal steps as `reasoning_tokens`;
- server smoke artifact exists under
  `/workspace/latent-mimo/artifacts/qwen_vllm_latent_server_check_generic`.

Still not done:

- `async_scheduling=True` is intentionally rejected for latent requests;
- latent mode state is still updated in worker-side Python bookkeeping;
- worker internals still use `latent_qwen35_*` names in several places;
- worker internals do not yet dispatch through a generic backend class; the
  Qwen3.5 execution adapter is registered, but the worker still calls Qwen-named
  helper methods.

## Correct async direction

Do not remove the async guard until next-step latent embeddings are produced in
an async-safe path. A naive enablement is wrong because async scheduling keeps
sampled token IDs on GPU and schedules the next step before CPU output handling.
For latent mode, the next input must be the MTP embedding, not the ordinary token
embedding.

The correct staged refactor is:

1. Keep public API on native reasoning fields.
2. Move latent-active, close-token, and internal-token-limit state from ad hoc
   worker Python bookkeeping into tensor/request state that the scheduler and
   runner can both observe.
3. Move MTP embedding generation into the runner/model transition so the next
   `inputs_embeds` buffer is prepared before scheduling the next decode step.
4. Keep the backend boundary as: backend consumes sampled token IDs and hidden
   states, returns `[batch, hidden_size]` embeddings plus close/continue masks.
5. Only then enable `async_scheduling=True` and benchmark CUDA graph behaviour.

## Backend swappability contract

A new latent backend should be able to change its internal head architecture as
long as it satisfies this external contract:

```text
inputs:
  sampled_token_ids: [num_active]
  target_hidden_states: [num_active, hidden_size]
  positions / slot mapping / backend cache metadata

outputs:
  next_input_embeddings: [num_active, hidden_size]
  continue_latent_mask: [num_active]
  internal_token_count_delta: [num_active]
```

Qwen3.5-MTP is the first backend. Other heads should register a backend spec and
provide an adapter implementing the same contract; the public OpenAI API should
not change.

## Recent hot-path cleanup

Native MTP pending state no longer needs a Python dict keyed by `(req_id, pos)`.
The runner already keeps preallocated pending arrays indexed by request index;
the dict was redundant glue and added extra Python operations in the decode path.
The current branch uses a pending counter plus the preallocated arrays.
