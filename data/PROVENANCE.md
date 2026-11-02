# Run provenance

All three runs: Tesla T4 (15360 MiB), vLLM 0.32.0, Qwen/Qwen2.5-1.5B-Instruct, profile_vllm.py with
rates 1 2 4 8 16 24 32, duration 60 s, 3 cold start trials, max_model_len 2048, gpu_util 0.85, seed 42.

| folder | provider | date | driver | kernel | notes |
|---|---|---|---|---|---|
| results_run1 | Kaggle (T4 x2 session, GPU 0 used) | 2026-10-21 | 580.178.04 | 6.18.48+ | Smoke test ran first in the same session, so cold start trial 0 had the model and torch.compile cache already on disk (not a true first boot). |
| results_run2 | Kaggle | 2026-10-25 | 580.178.04 | 6.18.48+ | No smoke test; cold start trial 0 is a true first boot (download + compile). Zip stored the absolute path kaggle/working/results_run2; moved here. |
| results_run3 | Google Colab | 2026-10-25 | 580.82.07 | 6.6.122+ | No smoke test. Run with --out results_run2 inside Colab (env.json args.out says results_run2); renamed to results_run3 here. Zip stored content/results_run2. |

## Long-prompt profile (results_longctx_run1, 2026-11-01)
Google Colab, Tesla T4, driver 580.82.07, kernel 6.6.159+, vLLM 0.32.0, same
model, max_model_len 2048 and gpu_util 0.85 as above. profile_vllm.py with
`--rates 0.5 1 2 4 8 16 --duration 60 --coldstart-trials 1 --in-mean 1024
--max-in 1536 --out-mean 128 --max-out 512` (validation/RUNBOOK.md, section A).
No smoke test; cold start trial 0 is a first boot (154 s spawn to ready).
KV cache reported 312,816 tokens (runs 1-3: 324,704). At 16 req/s, 385 of 933
requests hit the profiler's 300 s client timeout (`TimeoutError`); they are kept
as failures. Zip stored the relative path results_longctx_run1/.

## Two real replicas (runs/val_t4x2_v1, 2026-11-01)
Kaggle "GPU T4 x2", driver 580.178.04, vLLM 0.32.0, torch 2.13.0, transformers
5.16.1 (`env.txt`). `validation/two_replica_t4.py --warmup`, then `--campaign
--trials 3`: step12 x {queue, running, kv-cal, static-1, static-2} x 3 seeds,
15 of 15 trials done. The first attempt in the session ran before vLLM was
installed (Internet was off): it left `warmup_events.jsonl`'s first GPU 0
`create` line without a `ready`, the traceback at the top of `campaign.log`,
and `step12/running/trial0/server_gpu0_1793584665.log` (ModuleNotFoundError).
That trial was rerun from scratch afterwards (no done.json existed); the files
are kept as-is. Warm-up first boot: GPU 0 176 s, GPU 1 72 s. 9 of 28,830
requests failed with HTTP 500 from the script's router (aiohttp
`ServerDisconnectedError` / `Connection reset by peer` on the upstream vLLM
connection, see `campaign.log`); counted as failures. One `ValueError: I/O
operation on closed file` in queue/trial0: a scale-down drain outlived the
trial, so its final `killed` event was not logged. Zip stored the relative path
val_t4x2/; unpacked into runs/val_t4x2_v1/.

## Emulator TTFT metric excludes queueing (found 2026-10-26)
llm-d-inference-sim v1.0.0 records `vllm:time_to_first_token_seconds` without
waiting-queue time: after a step/ramp trial every one of 4,643 observations is
<= 2.5 s while client-side TTFT p99 was 100-230 s. vLLM v1 measures TTFT from
frontend arrival (`iteration_timestamp - req_stats.arrival_time`,
vllm/v1/metrics/stats.py), so it includes queueing. Effect: on kind the
`ttft` policy's p90 signal saturates near 2.35 s and scales up more slowly than
it would on real vLLM. The simulator follows vLLM semantics. SLO attainment
agrees (0.196 vs 0.198 step), pod-seconds differ by 16-21% for this policy only.

## Azure LLM inference trace 2024 (downloaded 2026-10-30)
Source: https://github.com/Azure/AzurePublicDataset/releases/download/dataset-llm-2024/AzureLLMInferenceTrace_conv_1week.csv
(CC-BY; cite Stojkovic et al., DynamoLLM, HPCA 2025). 1,135,195,393 bytes,
SHA-256 a0cc9b969a9bbf0fd811802cbf4323edd3a209ace791e3799ad4f9207f213941,
27,303,999 requests, 2024-05-12 00:00 to 2024-05-18 23:59 UTC. Not in git
(see .gitignore); `conv_per_minute.json` (per-minute counts, all 7 days) is.
`python sim/trace.py extract 2024-05-13` writes the Monday arrival times
(4,598,555 requests) used by `sim/trace_study.py`. Only arrival times are used;
prompt and output lengths come from the T4 profile's lognormal mix, because the
trace's context lengths (mean 1,631, p90 3,823 tokens) exceed the calibrated
range and max_model_len 2048. `trace30/seed*.arrivals.jsonl` are the thinned
30-minute windows (15:50-16:20 UTC) replayed on kind and in `sim/trace30.py`.
