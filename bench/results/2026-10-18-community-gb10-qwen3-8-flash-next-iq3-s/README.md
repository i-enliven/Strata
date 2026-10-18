# Community benchmark on NVIDIA GB10

Measured on 2026-10-18 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GB10, VRAM not read
- CPU: aarch64 (20 cores, 20 threads); installed RAM: 121.6 GiB; storage: not measured
- OS: Linux 7.0.0-1019-nvidia; GPU driver: 595.99.02; CUDA: 13.2
- Strata: project 0.1.45, commit 4bb21c1b; engine 0.1.45
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq3_s
- Context: 131072 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 54.4 (47.8-72.3) | 0.43 (0.39-0.47) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1472.9 (1373.2-1525.0) | not measured | 2.76 (2.67-2.96) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1674.1 (1642.2-1678.4) | not measured | 9.63 (9.61-9.82) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 1700.5 (1698.9-1701.1) | not measured | 18.91 (18.90-18.92) |
| agent session, follow-up turns | 14593 (14585-14593) at the last turn | see below | up to 64 | 3 | see below | 59.5 (41.0-70.8) | 2.44 (1.62-3.12) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.8 % (68.8-68.8); earlier-turn tokens read again per session: 4 (3-4) (earlier turns that the cache did not hold; the 13719 (13718-13719) new tokens of the files are read once); first turn (nothing cached) first token: 1.16 s (1.16-1.27).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 494.2 | 1.266 | 47.9 |
| 1 | 3756 | 633 | 3123 | 1022.6 | 3.089 | 60.4 |
| 2 | 5289 | 3819 | 1470 | 777.2 | 1.927 | 61.4 |
| 3 | 8722 | 5353 | 3369 | 1093.1 | 3.122 | 54.0 |
| 4 | 11642 | 8780 | 2862 | 1020.4 | 2.846 | 64.0 |
| 5 | 14593 | 11694 | 2899 | 1086.7 | 2.707 | 51.9 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 180 s.
