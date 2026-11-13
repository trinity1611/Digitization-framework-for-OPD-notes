# Memory and inference-time report

## Per model

| model | mock | runs | complete | truncated | failed | precision | load s | generation s: mean (min, max) | output tokens: mean (min, max) | tokens/s: mean (min, max) | peak memory GB, load | peak memory GB, inference: mean (min, max) | released after unload |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| medgemma_1_5_4b | no | 2 | 2 | 0 | 0 | bitsandbytes 4-bit NF4 (double quant) | 12.7, 14.4 | 428.62 (min 425.12, max 432.12) | 3761.00 (min 3761.00, max 3761.00) | 8.78 (min 8.70, max 8.85) | 3.01 | 4.05 (min 4.05, max 4.05) | True |

Peak memory basis: torch.cuda.max_memory_allocated (discrete GPU memory).

## Per run

| experiment | model | status | image tokens | input tokens | output tokens | generation s | peak memory GB (inference) | errors |
|---|---|---|---|---|---|---|---|---|
| EXP_0001 | medgemma_1_5_4b | complete | 256 | 4729 | 3761 | 432.12 | 4.05 |  |
| EXP_0002 | medgemma_1_5_4b | complete | 256 | 4729 | 3761 | 425.12 | 4.05 |  |
