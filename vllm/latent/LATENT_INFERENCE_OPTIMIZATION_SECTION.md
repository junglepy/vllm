# Оптимизация vLLM для инференса в режиме латентных рассуждений

## Контекст

В режиме латентных рассуждений модель генерирует не длинную цепочку видимых reasoning-токенов, а компактную последовательность скрытых векторов внутри блока рассуждений. После завершения скрытого блока модель возвращается к обычной генерации видимых токенов ответа.

Такой режим уменьшает число шагов рассуждения: вместо длинной последовательности токенов вида `r1, r2, ..., rN` модель проходит более короткую последовательность латентных состояний `l1, l2, ..., lK`. На практике это дает существенный выигрыш по end-to-end времени, потому что число шагов становится в несколько раз меньше.

При этом один латентный шаг устроен сложнее, чем обычный шаг генерации токена. Обычная генерация делает один forward основного backbone и затем получает распределение по словарю через `lm_head`. Латентный шаг дополнительно вызывает небольшой MTP/latent-head, который по последнему hidden state backbone предсказывает следующий continuous embedding. Этот embedding затем подается как вход следующего шага backbone.

Упрощенно:

```text
Обычный token step:
backbone -> lm_head -> token

Latent internal step:
backbone -> latent/MTP head -> latent embedding -> следующий backbone step
```

Из-за этого latent-режим выигрывает по общему времени за счет меньшего числа шагов, но один шаг может быть медленнее обычного CoT decode step.

## Что было сделано

Основная цель оптимизации состояла в том, чтобы перевести latent-инференс из прототипного режима в нативный путь vLLM, максимально близкий к обычному decode path.

Ключевые изменения:

1. Нативная интеграция latent-head в vLLM runtime.

   Вместо отдельного внешнего HF-style вызова latent-head был подключен как часть inference runtime. Это позволило использовать тот же механизм scheduling, attention metadata и GPU execution path, что и основной backbone.

2. Batched forward для latent-head.

   Ранний вариант фактически обрабатывал latent-предсказания менее эффективно: отдельные запросы могли приводить к множеству мелких вызовов. Был сделан batched path, где latent-head вызывается один раз на группу активных latent-запросов.

3. Использование paged KV-cache механики vLLM для latent-head.

   Latent-head использует собственные Q/K/V веса, поэтому физически его KV-cache нельзя объединить с KV-cache backbone. Но можно использовать ту же схему позиций, slot mapping, block table и общий attention backend. Это важно: кэш у backbone и latent-head разный по значениям, но управляется через совместимую инфраструктуру.

4. Исправление double-write KV проблемы.

   В latent-режиме часть входов приходит не как token id, а как continuous embedding. Было важно гарантировать, что такие шаги не записываются в KV-cache некорректно дважды и что cache state соответствует фактическому входу модели. Это фундаментальная корректностная часть: ускорения, нарушающие это свойство, отбрасывались.

5. Fast path для contiguous latent inputs.

   Если весь текущий batch состоит из latent-шагов или latent-шаги идут плотным префиксом batch-а, можно избежать части лишних операций: не пересобирать embeddings для всех токенов, не делать лишние scatter/copy и использовать более прямой путь подготовки входов.

6. CUDA graph path для latent-head.

   Для частых форм decode batch-а был добавлен CUDA graph fast path. Это уменьшает overhead от Python, dispatch и повторного построения части runtime metadata. На contiguous latent batches это дает заметное снижение стоимости MTP-head forward.

7. CUDA graph fast path включен по умолчанию для native latent-head.

   Ранее этот путь зависел от явного env-флага. В таком виде легко случайно запустить benchmark в native mode, но без CUDA graph для MTP-head. Теперь fast path включается по умолчанию при `LATENT_QWEN35_ENABLE_NATIVE=1`; для диагностики его можно отключить через `LATENT_QWEN35_CUDAGRAPH_HEAD=0`.

8. Pre-capture CUDA graphs.

   Чтобы не платить цену первого graph capture внутри измеряемого benchmark-а, графы для типовых размеров batch-а заранее прогреваются при инициализации engine. Это переносит cold-start стоимость из измеряемого decode loop.

9. Профилирование по фазам.

   Были добавлены измерения отдельных фаз: основной backbone forward, подготовка latent inputs, forward latent-head, projection в embedding, bookkeeping. Это позволило отделить реальные bottleneck-и от эффектов холодного старта и от общей стоимости загрузки модели.

## Что оказалось сложным

Главная сложность в том, что latent-head сам по себе маленький относительно Qwen3.5-27B, но overhead латентного шага определяется не только FLOPs.

Важные источники overhead:

1. Дополнительный forward на критическом пути.

   Даже если latent-head составляет условно несколько процентов от весов модели, он добавляет отдельный запуск вычислений перед следующим backbone step. Это дополнительный GPU dispatch, attention metadata, cache update и projection в embedding.

2. Mixed batch layout.

   На практике batch не всегда состоит только из latent-шагов. Часть запросов может уже выйти из latent reasoning и генерировать обычные видимые токены, часть может еще быть внутри latent блока, часть может находиться на prefill/extend. Самый быстрый путь работает, когда latent-запросы идут плотным contiguous блоком. Если latent slots разбросаны по batch-у, приходится использовать более общий и медленный путь.

3. Нельзя произвольно переупорядочивать запросы.

   Кажется естественным переставить pending latent-запросы в начало batch-а, чтобы чаще попадать в contiguous fast path. Но на практике это затрагивает scheduler state, sampling state, KV-cache layout, logits processors и соответствие request id к batch index. Эксперименты с таким reordering показали изменение длины latent reasoning и ухудшение self-exit/maxed метрик, поэтому такой подход был отклонен.

4. Нельзя объединить KV-cache backbone и latent-head.

   Хотя latent-head архитектурно похож на один decoder layer backbone, у него другие Q/K/V веса. Следовательно, ключи и значения в cache имеют другой смысл. Можно разделять инфраструктуру cache management, но нельзя физически переиспользовать значения cache между backbone и latent-head.

5. Качество чувствительно к небольшим на первый взгляд runtime changes.

   Некоторые оптимизации ускоряли или меняли execution path, но приводили к изменению средней длины latent блока, снижению self-exit rate или росту maxed rate. Такие изменения нельзя считать валидными, даже если они выглядят технически близкими к исходному вычислению.

6. Профилирование само замедляет запуск.

   Подробные CUDA event measurements и синхронизации полезны для breakdown-а, но они сильно искажают абсолютную скорость. Поэтому финальные speed numbers нужно брать из прогонов без детального профилирования, а profile runs использовать только для локализации bottleneck-а.

## Текущие результаты

На GSM8K 50 примеров с Qwen3.5-27B и checkpoint Stage11C step500 текущий стабильный latent path дает следующий порядок величин.

Сравнение end-to-end:

| Режим | Среднее число шагов | Время на пример |
|---|---:|---:|
| Base CoT | ~1531 | ~1.73 s |
| Latent | ~322 | ~0.44 s |

То есть по полному времени ответа latent-режим быстрее примерно в 3.9 раза на этой выборке. Это происходит не потому, что каждый шаг быстрее, а потому что шагов существенно меньше.

Сравнение per-step:

| Режим | sec / total step | Относительно base |
|---|---:|---:|
| Base CoT | ~0.00113 | 1.00x |
| Latent текущий | ~0.00137 | ~1.21x |

Текущий latent step все еще примерно на 21% медленнее обычного base decode step. До оптимизаций overhead был выше: около 1.33x по `sec / total step`. После перевода части пути на native batching, CUDA graph и contiguous fast path overhead снизился примерно до 1.21x.

Если считать через `sec / visible token`, latent выглядит хуже, потому что visible token-ов в latent-режиме меньше: большая часть рассуждения заменена hidden vectors. Поэтому для анализа чистой вычислительной стоимости latent-механизма полезнее смотреть не только `sec / visible token`, но и `sec / total step`, где latent internal steps учитываются явно.

Важное следствие для acceptance-метрики: если `sec / visible token` считать только по обычным токенам после latent-блока, то эта метрика включает время скрытых latent steps в числитель, но не включает их в знаменатель. На текущем валидном GSM8K-50 среднее отношение `total steps / visible tokens` для latent равно примерно `322 / 240 = 1.34`. Поэтому даже идеальный вариант, где latent имеет ровно такую же скорость одного total step, как base CoT, дал бы `sec / visible token` примерно в `1.34x` от base, что уже выше порога `1.07x`.

Для прохождения такого strict visible-token gate на текущей длине траектории latent должен быть не просто сопоставим с base per-step, а примерно на 20% быстрее base по `sec / total step`. Это другая инженерная цель: она требует не только убрать overhead latent-head, но и ускорить весь backbone decode step относительно обычного base CoT.

## Что осталось bottleneck-ом

После оптимизаций основной bottleneck находится в mixed-slots path: когда latent-запросы не образуют плотный contiguous блок в batch-е. В таких случаях latent-head чаще идет через более общий eager path. По профилю именно эта зона дает заметную часть overhead-а.

Типичная картина:

- contiguous/all-latent path через CUDA graph стоит существенно дешевле;
- mixed fallback path имеет дополнительную стоимость на подготовку metadata, gather/scatter, token embedding copy и eager MTP-head forward;
- сам `to_embed` projection не является главным bottleneck-ом;
- обычная генерация видимых токенов после выхода из latent-блока не должна платить стоимость latent-head, потому что latent-head используется только внутри hidden reasoning segment.

## Отклоненные варианты

Были проверены несколько направлений, которые выглядели перспективно, но не были оставлены как стабильные:

1. Batch reordering pending latent-запросов.

   Идея: переставлять запросы так, чтобы latent slots становились префиксом batch-а и попадали в fast path. На практике это изменило динамику генерации: выросла длина latent reasoning, снизился self-exit rate и вырос maxed rate. Такой результат нарушает критерии качества.

2. Compact CUDA graph path для произвольных latent slots.

   Идея: собрать только latent slots в компактный batch и прогнать latent-head через CUDA graph. Вариант показал, что технически можно убрать часть eager overhead-а, но стабильный выигрыш по full benchmark не подтвердился, а динамика reasoning менялась. Поэтому вариант не был принят как финальный.

3. Masked dummy slots.

   Идея: сохранить размер batch-а, но замаскировать non-latent позиции. Это оказалось слишком дорогим и не дало нужного ускорения.

4. Mixed `input_ids + inputs_embeds` path.

   Идея: оставить обычные token slots на `input_ids`, а continuous latent slots подмешивать через mask внутри model forward, чтобы вернуть embedding layer обычных токенов внутрь backbone CUDA graph. Повторный smoke на GSM8K-5 показал нарушение динамики latent reasoning: средняя длина hidden-блока упала примерно до `41` latent steps при допустимом диапазоне `[80, 200]`, а `sec / total step` стал существенно хуже. Поэтому этот путь также не принят.

Общее правило: оптимизация отклоняется, если она меняет поведение latent reasoning за пределы допустимых метрик, даже если локально уменьшает стоимость одного участка.

## Ожидаемый теоретический предел

Latent-head по размеру значительно меньше основной модели: грубо это один дополнительный decoder-like block плюс projection в embedding space. Поэтому по чистым FLOPs идеальный overhead latent step не должен быть большим. При достаточно хорошей интеграции можно ожидать overhead порядка 5-10% относительно обычного decode step.

Практический предел зависит от того, удастся ли убрать фиксированные runtime costs:

1. Свести mixed-slots path к CUDA graph или fused execution без изменения batch semantics.
2. Убрать лишние host-side metadata allocations и синхронизации.
3. Минимизировать copies между token ids, hidden states и inputs_embeds.
4. Возможно, объединить часть подготовки latent embedding с существующим decode graph, не нарушая отдельный KV-cache latent-head.
5. Сохранить корректное cache update правило для continuous embeddings.

Целевая инженерная планка: приблизить latent per-step throughput к base CoT с overhead не более 7%. Это соответствует режиму, где дополнительная стоимость latent-head почти полностью скрыта за эффективным batching/CUDA graph/fused execution.

Даже при текущем overhead около 21% по per-step latent уже быстрее end-to-end на задачах, где сжимает reasoning trajectory в 4-8 раз. Если довести per-step overhead до 5-7%, end-to-end ускорение будет определяться почти только коэффициентом сжатия рассуждений. Для текущего диапазона сжатия это означает реалистичный выигрыш порядка 4-6x, а на задачах с особенно длинным CoT потенциально выше.

## Итог

Оптимизация vLLM под latent reasoning состоит не просто в добавлении маленькой головы поверх модели. Ключевая задача - встроить continuous latent steps в decode runtime так, чтобы они использовали ту же эффективную инфраструктуру, что и обычные токены: batched execution, paged cache, CUDA graphs и минимальные copies.

Текущий результат уже дает сильный end-to-end выигрыш за счет сокращения длины reasoning, но per-step стоимость latent шага все еще выше base decode. Основной оставшийся резерв - mixed batch execution path для scattered latent slots. Его нужно ускорять без изменения порядка запросов, cache semantics и критериев качества.
