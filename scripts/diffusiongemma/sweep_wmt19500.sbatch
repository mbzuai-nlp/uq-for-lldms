#!/bin/bash
#SBATCH -J dg-wmt19500
#SBATCH --partition=cscc-gpu-p
#SBATCH --account=cscc-users
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:a100-sxm4-40gb:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH -t 10:00:00
#SBATCH -o /nfs-stor/artem.shelmanov/diffgemma/sweep_wmt19500.%j.out
set -x
export HF_HOME=/nfs-stor/artem.shelmanov/diffgemma/hf_cache HUGGINGFACE_HUB_CACHE=/nfs-stor/artem.shelmanov/diffgemma/hf_cache/hub HF_DATASETS_CACHE=/nfs-stor/artem.shelmanov/diffgemma/hf_cache/datasets HF_MODULES_CACHE=/nfs-stor/artem.shelmanov/diffgemma/hf_modules
export TORCH_HOME=/nfs-stor/artem.shelmanov/diffgemma/torch_cache NLTK_DATA=/nfs-stor/artem.shelmanov/diffgemma/nltk_data
export PYTHONNOUSERSITE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
source /home/artem.shelmanov/miniconda3/etc/profile.d/conda.sh; conda activate /nfs-stor/artem.shelmanov/diffgemma/envs/diffgemma
echo "node=$(hostname)"; nvidia-smi --query-gpu=index,memory.total --format=csv,noheader
cd /home/artem.shelmanov/vlad/uncertainty_dllm_dg
env HYDRA_CONFIG=/home/artem.shelmanov/vlad/uncertainty_dllm_dg/examples/configs/polygraph_eval_wmt19_deen_simple_instruct.yaml python polygraph_eval   batch_size=1 subsample_eval_dataset=500 ignore_exceptions=true   model=diffusiongemma model.path_to_load_script=/home/artem.shelmanov/vlad/uncertainty_dllm_dg/llada/load_diffusiongemma.py   estimators=estimators_diffusiongemma stat_calculators=calculators_diffusiongemma   cache_path=/nfs-stor/artem.shelmanov/diffgemma/out/dg_wmt19
echo "==== SWEEP wmt19500 EXIT=$? ===="
