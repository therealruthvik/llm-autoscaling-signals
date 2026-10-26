# Kaggle T4 profiling runbook

Goal: collect real single-replica vLLM measurements to parameterize the autoscaling study.

## Setup (once)
1. kaggle.com -> New Notebook -> Settings -> Accelerator: **GPU T4 x1** (or P100 if T4 unavailable; note which in the paper).
2. Settings -> Internet: **On** (needed to pip install and download the model; phone verification required).
3. Upload `profile_vllm.py` as a notebook file, or paste it into a cell with `%%writefile profile_vllm.py`.

## Run
Work in `/kaggle/working` (only files there appear in the Output tab). The `!` shell may start in `/root`, so `cd` first.
```
%cd /kaggle/working
!pip install -q vllm aiohttp numpy
!pip uninstall -y torchaudio
```
The pip resolver errors about bigframes, cudf, google-* etc. are harmless. Uninstalling torchaudio is required: Kaggle ships torchaudio built for CUDA 12.8, vllm pulls a CUDA 13.0 torch, and transformers imports torchaudio on startup, which crashes the vLLM server ("PyTorch and TorchAudio were compiled with different CUDA versions").

Smoke test first (about 3 min), then the real run. `CUDA_VISIBLE_DEVICES=0` pins one GPU if the session has T4 x2.
```
!CUDA_VISIBLE_DEVICES=0 python profile_vllm.py --model Qwen/Qwen2.5-1.5B-Instruct --out smoke \
    --rates 1 --duration 20 --coldstart-trials 1
!CUDA_VISIBLE_DEVICES=0 python profile_vllm.py --model Qwen/Qwen2.5-1.5B-Instruct --out results_runN \
    --rates 1 2 4 8 16 24 32 --duration 60 --coldstart-trials 3
```
Do the smoke test only in the first session. In runs 2 and 3 skip it, so cold start trial 0 is a true first boot (model download plus torch.compile, no caches). Run 1 trial 0 was already warm from the smoke test.

Tested 2026-10-21: vllm 0.32.0, torch 2.13.0+cu130, driver 580.178.04, Tesla T4.
Expected wall time: roughly 30-40 min (3 cold starts + 7 x 60s sweeps + drains; overloaded levels take several minutes to drain). Cold start timings depend on download/cache state.

If pip install of vllm breaks the Kaggle torch install, start a fresh notebook session and install vllm first before anything else. If the server dies on startup, read `results/server_coldstart_0.log`. The T4 is an older GPU (compute capability 7.5), and recent vLLM releases have had gaps in Turing support for some attention backends. If the latest vllm will not start, pin an older release that does (`pip install vllm==<version>`), use that same version for all three runs, and send me the error text so the paper can state the version and why.

Cold start trial 0 includes downloading the model from Hugging Face. Trials 1 and 2 load from local disk. These are different quantities; keep them separate.

`summary.csv` is rewritten after each rate level, so if the session times out mid-sweep, the finished levels are still usable.

## Get results out
```
!zip -r results.zip results
```
Download `results.zip` from the notebook Output tab. Send me the contents of: `env.json`, `coldstart.json`, `summary.csv` (small). Raw jsonl stays in your repo.

## Checks before trusting the data
- `summary.csv`: at low rate, `fail` = 0. At high rate, TTFT p99 should rise sharply = saturation knee. Find the rate where it bends; that is the single-replica capacity.
- If no knee by 16 rps, add `--rates 24 32 48`.
- `metrics_rate*.jsonl`: `waiting` should grow past the knee, `kv_cache` should climb.
- `coldstart.json` should include `kv_cache_tokens` (the KV cache capacity). If it is missing, send me the line in the server log that reports KV cache size.
- Metric names missing in the trace = vLLM version renamed them; tell me the version in `env.json`.

## Run three times
Run the whole thing 3 separate times (different sessions, different days if possible). Save each as `results_run1/`, `results_run2/`, ... Variance across sessions goes in the paper.

## Not measured here (do on your Mac kind cluster)
- Container image pull time (vllm image is multi-GB): time `docker pull` / pod `ContainerCreating` -> `Running`.
- Prometheus scrape interval and KEDA polling/stabilization lag.