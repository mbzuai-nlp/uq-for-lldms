# CIAI cluster: how to launch experiments (paste into a new session)

Everything below is verified working as of 2026-07-26 on branch `feat/diffusiongemma-uq`.

---

## Access

```bash
ssh ciai.mbzuai.ac.ae          # user artem.shelmanov (shared account), login node ciai-login-1
```

Requires VPN. Internal addresses are `10.127.x` and the VPN drops intermittently. Jobs already
submitted survive the drop, so never conclude a job failed just because SSH died. When polling
job state from a script, probe liveness first (`echo LIVE; squeue ...`) and treat "no SSH" as
unknown rather than "finished".

**Shared account.** `squeue -u artem.shelmanov` shows other people's jobs too. Never `scancel`
anything you did not submit.

---

## Partitions, accounts, QOS

Two usable GPU partitions, both A100-40GB, 4 per node. The gres name is
`a100-sxm4-40gb`, not plain `a100`.

**Partition `cscc-gpu-p`** needs all three flags, and caps you at **2 concurrent jobs**
(a third sits `PENDING` with reason `QOSMaxJobsPerUser`):

```bash
#SBATCH --partition=cscc-gpu-p
#SBATCH --account=cscc-users
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:a100-sxm4-40gb:2
```

**Partition `long`** uses the default account and QOS, so no extra flags:

```bash
#SBATCH -p long
#SBATCH --gres=gpu:a100-sxm4-40gb:2
```

**To run more than 2 jobs at once, split across both.** Put two on `cscc-gpu-p` and the rest on
`long`. Keeping three on `cscc-gpu-p` serialises the last one and roughly doubles wall-clock.

Check availability with `sinfo -p cscc-gpu-p,long -o '%.12P %.9T %.5D'`. Many nodes sit in
`down*` or `drained`; look for `mixed` or `idle`.

---

## Storage

- **`/nfs-stor/artem.shelmanov/`** is the working area: shared NFS, 175T, no quota. Put envs,
  caches, outputs and logs here.
- **Home is quota'd.** An env build will die partway through with a confusing error if you put
  it there.
- `/scratch` and `/tmp` are node-local and vanish between jobs.

Note: on the `long` partition TMPDIR sits on NFS, so logs may contain
`OSError: [Errno 16] Device or resource busy: '.nfsXXXX'` from multiprocessing temp-dir cleanup.
It is cosmetic. Confirm the job is still advancing rather than killing it.

---

## Working sbatch template

```bash
#!/bin/bash
#SBATCH -J my-job
#SBATCH --partition=cscc-gpu-p
#SBATCH --account=cscc-users
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:a100-sxm4-40gb:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH -t 10:00:00
#SBATCH -o /nfs-stor/artem.shelmanov/diffgemma/my-job.%j.out
set -x
export HF_HOME=/nfs-stor/artem.shelmanov/diffgemma/hf_cache \
       HUGGINGFACE_HUB_CACHE=/nfs-stor/artem.shelmanov/diffgemma/hf_cache/hub \
       HF_DATASETS_CACHE=/nfs-stor/artem.shelmanov/diffgemma/hf_cache/datasets \
       HF_MODULES_CACHE=/nfs-stor/artem.shelmanov/diffgemma/hf_modules
export TORCH_HOME=/nfs-stor/artem.shelmanov/diffgemma/torch_cache \
       NLTK_DATA=/nfs-stor/artem.shelmanov/diffgemma/nltk_data
export PYTHONNOUSERSITE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
source /home/artem.shelmanov/miniconda3/etc/profile.d/conda.sh
conda activate /nfs-stor/artem.shelmanov/diffgemma/envs/diffgemma
echo "node=$(hostname)"; nvidia-smi --query-gpu=index,memory.total --format=csv,noheader
cd /home/artem.shelmanov/vlad/uncertainty_dllm_dg
<your command here>
echo "==== EXIT=$? ===="
```

Always end with an `EXIT=$?` marker. It is the only reliable way to tell a clean finish from a
kill when scanning logs later.

**Compute nodes have internet.** Do not set `HF_HUB_OFFLINE` or `HF_DATASETS_OFFLINE`; models
and datasets download or resolve from cache normally. Setting them breaks dataset loading.

---

## Code sync

Local and cluster share one git repo. Never run from uncommitted local edits.

```bash
# locally
git commit -am "..." && git push origin <branch>

# on the cluster
ssh ciai.mbzuai.ac.ae "cd /home/artem.shelmanov/vlad/uncertainty_dllm_dg \
  && git pull -q origin <branch> && git log --oneline -1"
```

Worktree `/home/artem.shelmanov/vlad/uncertainty_dllm_dg` is on `feat/diffusiongemma-uq`.
Vlad's main checkout is a **different directory** on another branch; leave it alone. Always
confirm the branch before launching, it may have been left on something else.

---

## Environment (DiffusionGemma)

Prefix env at `/nfs-stor/artem.shelmanov/diffgemma/envs/diffgemma`, python 3.11. Direct
interpreter for analysis, no activation needed:

```bash
/nfs-stor/artem.shelmanov/diffgemma/envs/diffgemma/bin/python
```

Pins that matter, all handled by `scripts/setup_diffusiongemma_env.sh`:

- `torch 2.6.0+cu124` with matching `torchvision 0.21.0` (required: the checkpoint is
  Gemma4-multimodal and the processor imports torchvision)
- `transformers 5.14.0` installed **last, with deps**, otherwise extras downgrade it
- **`datasets==3.6.0`**, not 5.x. Version 5 dropped `trust_remote_code` and script-dataset
  support that the LM-Polygraph datasets need
- `setuptools<81` for `pkg_resources`, which comet and torchmetrics import
- a patch to `unbabel-comet`'s XLM-R encoder for transformers 5.x, without which WMT19/COMET
  scoring aborts on the first sample

---

## Launching a DiffusionGemma sweep

```bash
WT=/home/artem.shelmanov/vlad/uncertainty_dllm_dg
CFG=$WT/examples/configs
env HYDRA_CONFIG=$CFG/polygraph_eval_<ds>_simple_instruct.yaml python polygraph_eval \
  batch_size=1 subsample_eval_dataset=500 ignore_exceptions=true \
  model=diffusiongemma \
  model.path_to_load_script=$WT/llada/load_diffusiongemma.py \
  estimators=estimators_diffusiongemma \
  stat_calculators=calculators_diffusiongemma \
  cache_path=/nfs-stor/artem.shelmanov/diffgemma/out/dg_<ds>
```

For the sampling-based estimators (SemanticEntropy, KLE, CoCoA-MSP) swap in
`estimators=estimators_diffusiongemma_sampling` and
`stat_calculators=calculators_diffusiongemma_sampling`. That draws K=10 decodes per input, so
budget roughly 10x the runtime.

Two config gotchas that cost real time:

1. Config files must live **inside `examples/configs/`**, or hydra cannot find the config group.
2. `model.path_to_load_script` must be passed **absolute on the CLI**. The relative value in the
   model config resolves against the wrong directory.

Rough timings, 2x A100, 500 instances: trajectory-only sweep 30 to 60 min per dataset; sampling
sweep with K=10 about 3 hours.

---

## Rules of thumb

**Always smoke-test before a long sweep.** `subsample_eval_dataset=10` costs about five minutes
and has caught real config errors that would otherwise have wasted 15 GPU-hours.

**Never leave a free GPU idle.** When a run finishes, queue the next point immediately, and keep
more experiments queued than running.

**Save anything worth keeping to Hugging Face.** Cluster `out/` directories are per-host and get
reclaimed. Existing archive: `karantonis/diffusiongemma-uq-pickles` (private dataset repo).
There is no HF token on the cluster, so pull results to a local machine and upload from there.

---

## Reading results

Managers are written with `torch.save`, so `pickle.load` raises a persistent-id error:

```python
import torch
man = torch.load(path, weights_only=False, map_location="cpu")
man["metrics"][("sequence", "<Estimator>", "<GenMetric>", "prr_0.5_normalized")]
man["estimations"][("sequence", "<Estimator>")]
man["gen_metrics"][("sequence", "AlignScore")]
```

`man["metrics"]` is the harness-computed PRR and is authoritative. Do not substitute a
hand-rolled PRR; it will not match.

Output directories contain a bracketed dataset name with quotes and spaces, for example
`['LM-Polygraph/triviaqa', 'simple_instruct']`. **Shell globs fail on these.** Use
`find <root> -name ue_manager_seed1` instead.

The harness persists only `greedy_texts`, `greedy_tokens`, `input_texts` and `target_texts` in
`man["stats"]`. Anything else, such as `sample_texts`, has to be recovered from the run log.

Useful scripts in `scripts/diffusiongemma/`: `analyze_dg.py` (per-pickle PRR),
`consolidate_dg.py` (cross-dataset table), `bootstrap_dg.py` (confidence intervals),
`compare_sampling_vs_trace.py`, `sample_diversity.py`, `bench_dg_overhead.py`.

**n=100 is underpowered.** No gap between estimators reached significance at 100 instances, and
the sampling-based estimators shifted a lot between 100 and 500. Use 500 for anything
load-bearing, and report bootstrap intervals rather than bare point estimates.
