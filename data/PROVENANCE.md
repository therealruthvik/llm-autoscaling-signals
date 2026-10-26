# Run provenance

All three runs: Tesla T4 (15360 MiB), vLLM 0.32.0, Qwen/Qwen2.5-1.5B-Instruct, profile_vllm.py with
rates 1 2 4 8 16 24 32, duration 60 s, 3 cold start trials, max_model_len 2048, gpu_util 0.85, seed 42.

| folder | provider | date | driver | kernel | notes |
|---|---|---|---|---|---|
| results_run1 | Kaggle (T4 x2 session, GPU 0 used) | 2026-10-21 | 580.178.04 | 6.18.48+ | Smoke test ran first in the same session, so cold start trial 0 had the model and torch.compile cache already on disk (not a true first boot). |
| results_run2 | Kaggle | 2026-10-25 | 580.178.04 | 6.18.48+ | No smoke test; cold start trial 0 is a true first boot (download + compile). Zip stored the absolute path kaggle/working/results_run2; moved here. |
| results_run3 | Google Colab | 2026-10-25 | 580.82.07 | 6.6.122+ | No smoke test. Run with --out results_run2 inside Colab (env.json args.out says results_run2); renamed to results_run3 here. Zip stored content/results_run2. |
