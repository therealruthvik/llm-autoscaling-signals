# Real-GPU runs still needed (Kaggle)

Two sets of measurements cannot be made on the laptop. Both use free Kaggle GPU
sessions and the same vLLM setup as `KAGGLE_RUNBOOK.md` (vLLM 0.32.0,
Qwen2.5-1.5B-Instruct, fp16, `max_model_len` 2048, GPU memory utilization 0.85).
Follow that runbook's install steps (`pip install vllm aiohttp numpy`,
`pip uninstall -y torchaudio`). Record the versions in `env.json` as before.

Upload the repository to the session first (Kaggle dataset or a zip in the
notebook's Input). The scripts below need `cluster/loadgen.py`, `profile_vllm.py`
and `validation/`.

## A. Long-prompt profile (one T4, about 40 min)

Purpose: test the KV-bound regime. With the default 256-token prompts, KV usage
peaks at 35% and the 0.8 threshold never fires. With prompts around 1,024
tokens, KV memory should fill before the 256-request cap.

Settings: **GPU T4 x1**, Internet on.
```
%cd /kaggle/working
!CUDA_VISIBLE_DEVICES=0 python profile_vllm.py --model Qwen/Qwen2.5-1.5B-Instruct \
    --out results_longctx_run1 --rates 0.5 1 2 4 8 16 --duration 60 \
    --coldstart-trials 1 --in-mean 1024 --max-in 1536 --out-mean 128 --max-out 512
!zip -r results_longctx_run1.zip results_longctx_run1
```
`in + out` never exceeds 1,536 + 512 = 2,048 = `max_model_len`.
If there is no knee by 16 req/s, add `--rates 24 32`. Repeat in a second session
as `results_longctx_run2` if quota allows. Put the unzipped folders in `data/` and
add a row to `data/PROVENANCE.md`.

Checks: `metrics_rate*.jsonl` should show `kv_cache` near 1.0 and preemptions
(`preemptions` > 0) at the highest rates. If KV usage still stays below 0.8, that
is a result too: report it.

## B. Two real replicas (T4 x2, about 2.5 h)

Purpose: check the paper's mechanisms on real vLLM rather than the emulator:
replica startup time when the autoscaler fires, backlog pinned to the first
replica, and the ranking of queue vs running vs kv-cal vs static.

Settings: **GPU T4 x2**, Internet on.
```
%cd /kaggle/working/<repo>
# 1. first boot on each GPU (model download + compile), so later starts are warm
!python validation/two_replica_t4.py --warmup --out /kaggle/working/val_t4x2
# 2. 5 policies x 3 trials on the step12 burst (2 -> 12 -> 2 req/s), max 2 replicas
!python validation/two_replica_t4.py --campaign --trials 3 --out /kaggle/working/val_t4x2
!cd /kaggle/working && zip -r val_t4x2.zip val_t4x2
```
The campaign skips finished trials, so rerun the same command if the session
stops. Each trial takes about 7–10 min including server starts.

Bring `val_t4x2/` back into `runs/val_t4x2_v1/` and summarize it with the same
script as the kind trials:
```
python analysis/summarize_trials.py runs/val_t4x2_v1 --csv analysis/results/val_t4x2.csv
```
The matching kind trials (`runs/val_kind_step12_v1`) and simulator runs
(`sim/validation_step12.py`) are already in the repository for comparison.

The harness can be self-tested without a GPU:
`FAKE_VLLM=1 python validation/two_replica_t4.py --policy queue --out /tmp/valtest`.
