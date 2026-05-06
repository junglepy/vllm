# Latent-MTP интеграция в vLLM: разбор замедления и план оптимизации

**Дата:** 2026-05-06 (обновлено под актуальное состояние fork с локальными несокоммиченными правками)
**Контекст:** fork `junglepy/vllm`, ветка `feature/latent-mtp-integration`
**HEAD коммит:** `644f803 Add Qwen latent model aliases`
**Локальные правки** (не запушены): +530 строк в `gpu_model_runner.py`, +58 в `qwen3_5_latent_mtp.py` — подключение native MTP-головы и батчирование MTP forward.

## Что уже сделано хорошо

После последних правок hot path **существенно изменился** относительно того, что было запушено в HEAD:

- **`Qwen3_5LatentMTP` подключён как нативный vLLM-module** через `_maybe_init_latent_qwen35_native_head`. Регистрируется в `compilation_config.static_forward_context`, готов к torch.compile.
- **MTP-голова батчируется**: `_run_latent_qwen35_native_head` делает **один forward** для всех latent-активных запросов в текущем decode-step (вместо N forward'ов batch=1).
- **PagedAttention для MTP cache**: новая реализация строит `CommonAttentionMetadata` с `slot_mapping`, `block_table`, `query_start_loc` — то есть MTP-голова использует тот же attention backend, что и основная модель, через `vllm.v1.attention`. Никаких HF `DynamicCache` в hot path.
- **Standalone HF-голова осталась как fallback** (`_compute_latent_qwen35_next_embed`, `_advance_latent_qwen35_mtp_cache`), но не основной путь когда native head доступен.

То есть пункты «batch=1 forward», «standalone HF head», «отдельный DynamicCache» из предыдущей ревизии этого документа — **закрыты**.

## Симптомы (после батчирования + native head)

- Веса MTP head: **<5%** от Qwen3.5-27B (1 transformer layer + `to_embed`).
- Важное исправление методики: ранние замеры `base≈676 tok/s` vs `latent≈253 steps/s` включали cold-start эффекты latent path: lazy load checkpoint, первый прогрев native MTP head и/или first-use compile. Эти цифры нельзя использовать как steady-state throughput.
- Корректный synthetic warmup-замер на B200, Qwen3.5-27B, `checkpoint_step5500_NEW.pt`, `max_tokens=300`, `max_internal_tokens=300`, `LATENT_QWEN35_CAPTURE_INPUTS_EMBEDS=1`:
  - batch=5: base `378.9 tok/s`, latent `313.3 total_steps/s`, overhead `~21%`.
  - batch=20: base `1352-1358 tok/s`, latent `1074 total_steps/s`, overhead `~26%`.
- Текущий overhead **не 2.5×**, а примерно **20-30%** на synthetic steady-state. Это всё ещё выше идеальных `+5-15%`, но уже близко к ожидаемому порядку для дополнительной MTP-головы и Python-side glue.

## Где остаются bottleneck'и

### 1. `async_scheduling=False` обязателен — но CUDA graphs не отключены полностью

```python
# vllm/v1/worker/gpu_model_runner.py: _bookkeeping_sync
if not self.use_async_scheduling and valid_sampled_token_ids:
    # latent processing — переключение режима, lookup MTP cache, build pending entries
```

Latent-логика **синхронно после sampler** меняет `req_state.latent_qwen35_active`, читает `cache.get_seq_length()`, наполняет `latent_qwen35_pending_by_req_pos`. Эти модификации происходят на стороне worker'а **до** того как scheduler получит `ModelRunnerOutput`, поэтому async-scheduling нельзя включить — он закладывался на чисто-data-flow output.

Исправление: утверждение «без async_scheduling vLLM не использует CUDA graphs» неверно в текущем fork. Логи engine init показывают `Capturing CUDA graphs (mixed prefill-decode, PIECEWISE)` и `Capturing CUDA graphs (decode, FULL)` даже при `async_scheduling=False`.

Остаётся верным более слабое утверждение: `async_scheduling=False` добавляет scheduler/worker sync overhead и может ограничивать максимальный throughput, но это **не главный 2-3× bottleneck** в текущем состоянии.

**Чем починить (это и есть основная победа):**
- Перенести switch latent→normal на сторону scheduler/output_processor: вместо `req_state.latent_qwen35_active = False` в worker — добавить эту информацию в `ModelRunnerOutput.internal_token_ids` (она уже там как маркер) и обработать в `Scheduler.update_from_output`.
- `latent_qwen35_pending_by_req_pos` — это map позиций latent_embed для следующего forward. Сейчас он наполняется в worker. Нужно переместить логику генерации latent_embed **внутрь** model.forward (см. пункт 4 ниже), и тогда latent_embed автоматически будет в KV-cache → не нужен dict pending.
- После этого можно включить `async_scheduling=True` и захватить CUDA graph для всего decode-цикла.

**Ожидаемый эффект:** пока не подтверждён. После исправления методики замера realistic upside скорее десятки процентов, а не гарантированные `×1.5-2.5`.

### 2. Python bookkeeping после sample

В `_bookkeeping_sync` после sampler идёт:

```python
for req_idx, sampled_ids in enumerate(valid_sampled_token_ids):
    if len(sampled_ids) != 1: continue
    req_state = self.requests[req_id]
    if not req_state.latent_qwen35_active: continue
    token_id = int(sampled_ids[0])                             # GPU→CPU sync
    if token_id == int(req_state.latent_qwen35_think_close_token_id):
        req_state.latent_qwen35_active = False; continue
    num_internal_tokens = len(req_state.latent_qwen35_internal_positions)  # python set len
    if (req_state.latent_qwen35_max_internal_tokens >= 0
            and num_internal_tokens >= req_state.latent_qwen35_max_internal_tokens):
        req_state.latent_qwen35_active = False; continue
    ...
```

Исправление: формулировка «каждое `int(sampled_ids[0])` — отдельный GPU sync» в текущем коде, скорее всего, неверна или сильно завышена. `valid_sampled_token_ids` уже приходит как CPU-side structure после общего sampler output conversion, поэтому `int(...)` не обязан делать N отдельных `cudaStreamSynchronize`.

Тем не менее, Python loop остаётся overhead: per-request lookup, update `req_state`, set/dict операции и подготовка pending entries. Его имеет смысл убирать, но это не подтверждённый `>50% wall-clock` bottleneck.

**Чем починить:**
- Все условия (`is_close`, `over_internal_limit`) выразить как **тензорные маски** на GPU:
  ```python
  next_id_gpu = sampler_output.sampled_token_ids[:, 0]  # tensor
  active_mask = req_state_active_gpu & (~mask_close) & (~mask_over_limit)
  ```
- Один синк в конце forward, либо вообще без синка (нужна только маска для следующего forward).

**Ожидаемый эффект:** требует отдельного phase-profile после исправленного warmup. Предварительно это оптимизация второго порядка, не главный источник ранней 2.5× ошибки.

### 3. Условные ветки в input-embeds path ломают CUDA graph capture

Сейчас в `execute_model`:

```python
elif self.latent_qwen35_use_inputs_embeds and is_first_rank:
    if num_input_tokens > num_scheduled_tokens:
        self.inputs_embeds.gpu[num_scheduled_tokens:num_input_tokens].zero_()
    inputs_embeds = self.inputs_embeds.gpu[:num_input_tokens]
    input_ids = None
    model_kwargs = self._init_model_kwargs()
```

И отдельно — в `_prepare_latent_qwen35_native_inputs_embeds`:

```python
if len(entries) == total_num_scheduled_tokens and all(
    item[1] == idx for idx, item in enumerate(entries)
):
    # все слоты — latent: прямая запись latent_embeds
    self.inputs_embeds.gpu[:total_num_scheduled_tokens].copy_(embeds)
else:
    # частично latent: сначала embed_input_ids, потом перезапись слотов
    token_embeds = self.model.embed_input_ids(input_ids=...)
    self.inputs_embeds.gpu[...].copy_(token_embeds)
    self.inputs_embeds.gpu[scheduled_slots] = embeds
```

Две развилки:
1. `latent_qwen35_use_inputs_embeds` (True/False) — попадаем ли в input_ids или inputs_embeds path.
2. all-slots-latent vs partial-latent — внутри inputs_embeds path два разных пути формирования.

Первичная проверка `LATENT_QWEN35_CAPTURE_INPUTS_EMBEDS=1` показала, что унификация capture path полезна, но не объясняет 2× разницу. В текущем состоянии CUDA graphs всё равно захватываются; проблема скорее в стабильности/форме input path и Python glue, а не в полном отсутствии graph capture.

**Чем починить:**
- Унифицировать путь: **всегда через inputs_embeds**. Когда latent неактивен — просто `embed_input_ids(input_ids)`, никаких условий на input vs embeds.
- Внутри prepare: убрать развилку «all-slots vs partial»: всегда сначала `embed_input_ids` для всего batch, потом mask-overwrite активных latent slots. Лишний `embed_input_ids` стоит ~0.1% времени, выигрыш от стабильной формы графа — на порядки больше.
- Можно ввести «фиксированный» путь captured-graph с `is_latent_step: tensor[bool]` маской размера batch — тогда CUDA graph один для всех шагов.

**Ожидаемый эффект:** умеренный. Оставляем как дешёвую и правильную стабилизацию hot path, но не как главный bottleneck.

### 4. CPU list/tensor construction на каждом latent step (6+ host→device copies)

В `_prepare_latent_qwen35_native_inputs_embeds` и `_run_latent_qwen35_native_head`:

```python
entries: list[tuple[...]] = []
for slot in range(total_num_scheduled_tokens):           # python loop по batch
    req_idx = int(req_indices_np[slot])                  # CPU int convert
    req_id = self.input_batch.req_ids[req_idx]
    pos = int(positions_np[slot])
    pending = self.latent_qwen35_pending_by_req_pos.get(...)  # dict lookup
    ...
    entries.append((req_idx, slot, pos, token_id, prev_hidden, checkpoint, len(entries)))

entries.sort(key=lambda item: (item[0], item[2]))        # python sort

# 4× host→device copy от list comprehensions:
compact_input_ids = torch.tensor([item[3] for item in entries], dtype=torch.int32, device=...)
compact_positions = torch.tensor([item[2] for item in entries], dtype=torch.int64, device=...)
compact_hidden = torch.stack([item[4] for item in entries]).to(device=..., dtype=...)
compact_slot_mapping = group_slot_mapping[
    torch.tensor([item[1] for item in entries], dtype=torch.long, device=...)
]

# В _build_latent_qwen35_native_metadata ещё:
torch.tensor(req_indices, dtype=torch.long, device=...)
torch.tensor(seq_lens, dtype=torch.int32, device=...)
torch.tensor([qlen > 1 for qlen in query_lens], dtype=torch.bool, device=...)
```

Итого **6-7 host→device копий + 4 list-comprehensions + 1 sort + 1 dict-lookup-в-цикле** на каждый decode-step где есть хотя бы один latent slot. Это наиболее правдоподобный оставшийся источник overhead в районе наблюдаемых `20-30%`.

**Чем починить:**
- Pre-allocate **GPU buffers** для всего за `__init__`:
  ```python
  self.latent_pending_token_id_gpu = torch.empty(max_reqs, dtype=torch.int32, device=...)
  self.latent_pending_hidden_gpu = torch.empty(max_reqs, hidden_size, dtype=dtype, device=...)
  self.latent_pending_active_gpu = torch.empty(max_reqs, dtype=torch.bool, device=...)
  self.latent_pending_position_gpu = torch.empty(max_reqs, dtype=torch.int64, device=...)
  ```
- В worker'е **писать** в эти buffer'ы по `req_idx` сразу как тензор, без list/dict промежутка.
- Build `compact_*` через **boolean indexing** на GPU: `compact_input_ids = self.latent_pending_token_id_gpu[active_mask]`.
- Slot, query_lens, seq_lens получать через сегментацию маски: `torch.cumsum`, `torch.diff` — все на GPU.

После этого `_prepare_latent_qwen35_native_inputs_embeds` становится существенно ближе к tensor-only hot path. Полностью убрать Python будет сложно из-за request-state модели vLLM, но можно сократить количество list/dict/host→device операций.

**Ожидаемый эффект: ×1.15-1.3**.

## Сводка — реальный потенциал по этапам

| Этап | Действие | Эффект | Сложность |
|---|---|---|---|
| **A** | Исправить benchmark methodology: warm base+latent, прогрев target batch, repeats, `torch.cuda.synchronize()` | уже сделано | Низкая |
| **B** | Стабилизировать input-embeds path и держать `LATENT_QWEN35_CAPTURE_INPUTS_EMBEDS=1` по умолчанию в latent bench/server | уже частично сделано | Низкая |
| **C** | Pre-allocate GPU/CPU buffers для latent_pending_*, убрать list/dict/torch.tensor(list) из hot path | вероятно +5-20% | Средняя |
| **D** | Упростить bookkeeping и перенести часть state-switch в scheduler/output_processor | неизвестно, измерять | Высокая |
| **E** | Экспериментально включить async_scheduling после D | неизвестно | Высокая |

Текущая подтверждённая цель после исправления методики: с `batch=20` уже есть `~1074 total_steps/s` против `~1355 base tok/s`. Следующая реалистичная цель — снизить overhead с `20-30%` до `10-15%`, а не «догнать с 253 до 1000+».

## Замечания по корректности

- `latent_qwen35_pending_by_req_pos` — ключ `(req_id, position)`. После переноса в model.forward этот dict уйдёт целиком: latent_embed будет вычисляться внутри одного forward'а и сразу подставляться в KV.
- `_advance_latent_qwen35_mtp_cache` (warm-start prefill для MTP cache) сейчас вызывается во время bookkeeping. После миграции на native MTP с PagedAttention (которая уже частично сделана) prefill MTP cache должен происходить в том же batched forward, что и обычный text_backbone prefill.
- `Qwen3_5LatentMTP._remap_latent_checkpoint_weights` — корректный mapping `latent_head.core.* → model.layers.0.*`, `latent_head.to_embed.* → to_embed.*`. После полного переключения на native head standalone-загрузка через `Qwen35StandaloneLatentMTPHead` останется только для CLI fallback.
- В текущей реализации native head поддерживает **один checkpoint на engine instance** (`_ensure_latent_qwen35_native_head` бросает RuntimeError при попытке переключить). Это OK для production, но bench-скрипт что прогоняет step1500 + step5500 в одном процессе должен пере-инициализировать LLM между checkpoints.

## Hot path (актуальный, после локальных правок)

Decode step:

```
1. _prepare_inputs (обычный путь vLLM)
2. _prepare_latent_qwen35_native_inputs_embeds:
     - python loop по slot'ам, сбор entries (см. пункт 4)
     - sort + 4× torch.tensor(list) → compact_* tensors
     - _run_latent_qwen35_native_head:
         _build_latent_qwen35_native_metadata (≈3 host→device copy)
         head.forward(input_ids, positions, hidden_states, kv_cache via PagedAttention)
         compute_latent_embeds (= self.to_embed(hidden))
     - запись в self.inputs_embeds.gpu[scheduled_slots]
3. text_backbone forward через model.forward(inputs_embeds=...)
4. sampler → valid_sampled_token_ids (GPU tensor)
5. _bookkeeping_sync — python loop:
     - int(sampled_id) per req → CPU sync (см. пункт 2)
     - update req_state.latent_qwen35_active
     - заполнение self.latent_qwen35_pending_by_req_pos[(req_id, pos)] = (token_id, hidden, ckpt)
6. push в scheduler
```

Этапы 2 (python/list/dict/host→device overhead) и 5 (bookkeeping) — основные мишени оптимизации. Этап 3 (сам text_backbone forward) и `head.forward` внутри 2 — уже компилируемы через torch.compile + PagedAttention, проблем там нет. Ранее заявленный `per-request int(...) GPU sync` надо считать неподтверждённым до отдельного профиля.

## Рекомендуемая последовательность

1. Держать исправленный benchmark как базовый: full warmup, repeats, best/median steady-state, отдельные JSON artifacts.
2. Не оптимизировать «2.5× slowdown» — он оказался артефактом cold-start/методики.
3. Следующая инженерная цель: убрать `torch.tensor(list)`, `entries.sort`, dict lookup и лишние host→device copies из latent prepare path.
4. После этого заново замерить `batch=5`, `batch=20`, затем уже реальные math prompts с `max_tokens=6000`.
5. Только после подтверждения остаточного overhead возвращаться к scheduler/async_scheduling.

## Ссылки на код

- `vllm/v1/worker/gpu_model_runner.py:720-740` — init latent state (включая native head + pending dict).
- `vllm/v1/worker/gpu_model_runner.py:1809-2050` — `_prepare_latent_qwen35_native_inputs_embeds`, `_update_latent_qwen35_native_cache_from_hidden_states`, `_run_latent_qwen35_native_head`.
- `vllm/v1/worker/gpu_model_runner.py:2030-2090` — `_maybe_init_latent_qwen35_native_head`, `_ensure_latent_qwen35_native_head`.
- `vllm/v1/worker/gpu_model_runner.py:3650-3770` — `_bookkeeping_sync` с per-request python loop (sync path).
- `vllm/v1/worker/gpu_model_runner.py:3470-3490` — execute_model с `latent_qwen35_use_inputs_embeds` веткой.
- `vllm/model_executor/models/qwen3_5_latent_mtp.py` — Qwen3_5LatentMTP (native vLLM head, готов к torch.compile).
- `vllm/v1/core/sched/scheduler.py:1099-1110, 1306-1369` — scheduler treats internal_token_ids correctly; место где можно перенести state-switch для async_scheduling.
- `vllm/v1/request.py:215-237` — `append_internal_token_ids`.
