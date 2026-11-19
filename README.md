# Which signal should LLM inference scale on?

Code, raw measurements and run logs for a controlled comparison of reactive
Kubernetes autoscaling signals for vLLM, calibrated to measured single-replica
behavior on an NVIDIA T4.

## Layout

| Path | Contents |
|---|---|
| `profile_vllm.py`, `KAGGLE_RUNBOOK.md` | Single-GPU vLLM profiler and how to run it on Kaggle/Colab |
| `data/` | Measured T4 profiles (three sessions, two providers), see `data/PROVENANCE.md` |
| `cluster/` | kind cluster, Prometheus, KEDA, calibrated vLLM emulator, load generator, autoscaling policies |
| `experiments/` | Emulator validation and the policy-comparison campaign runner |
| `runs/` | Raw logs from emulator validation and campaign trials |
| `sim/` | Discrete-event simulator, calibration, sensitivity sweeps (all results simulated) |
| `analysis/` | Tables and figures |
| `paper/` | Paper draft (LaTeX) |

## Reproduce

1. Profile vLLM on a T4: follow `KAGGLE_RUNBOOK.md`.
2. Cluster: `cluster/01_create_cluster.sh`, `cluster/02_install_monitoring.sh`, `cluster/03_deploy_sim.sh`.
3. Emulator validation: `experiments/emulator_validation.sh runs/emuval_v2`.
4. Campaign: `python experiments/run_campaign.py --out runs/main`.
5. Simulator: `python sim/calibrate.py`, `python sim/sweep.py --out sim/results/sweep_v1.csv`.
6. Analysis: `python analysis/summarize_trials.py runs/main`, `python analysis/plots.py --campaign runs/main --sweep sim/results/sweep_v1.csv`.

Python dependencies: `aiohttp numpy scipy matplotlib`.

## License

Code is MIT (`LICENSE`). Data, run logs and the paper are CC BY 4.0
(`LICENSE-DATA`). Files under `data/traces/azure2024/` are derived from the
Azure LLM Inference Trace 2024 (CC BY 4.0; cite Stojkovic et al., DynamoLLM,
HPCA 2025).
