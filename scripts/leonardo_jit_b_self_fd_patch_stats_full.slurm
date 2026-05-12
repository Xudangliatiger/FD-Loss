#!/bin/bash
#SBATCH --job-name=jit_patch_full
#SBATCH --account=EUHPC_D32_028
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --time=12:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --output=/leonardo_scratch/large/userexternal/dxu00000/2027-ICLR-PixelGen/FD-Loss/runs/slurm/%x-%j.out
#SBATCH --error=/leonardo_scratch/large/userexternal/dxu00000/2027-ICLR-PixelGen/FD-Loss/runs/slurm/%x-%j.err

set -euo pipefail

module load python/3.11.7
module load cuda/12.6

source /leonardo_scratch/fast/EUHPC_D32_028/dongli/envs/fdloss-cu126-py311/bin/activate

cd /leonardo_scratch/fast/EUHPC_D32_028/dongli/FD-Loss

SCRATCH_ROOT=/leonardo_scratch/large/userexternal/dxu00000/2027-ICLR-PixelGen/FD-Loss
RUN_ROOT="${SCRATCH_ROOT}/runs"
CACHE_ROOT="${SCRATCH_ROOT}/cache"

export HF_HOME="${CACHE_ROOT}/huggingface"
export TORCH_HOME="${CACHE_ROOT}/torch"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export TMPDIR="/tmp/fdloss_${USER}_${SLURM_JOB_ID:-manual}"
export WANDB_MODE=disabled
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="/leonardo_scratch/fast/EUHPC_D32_028/dongli/envs/ggn-cu126-py311/lib/python3.11/site-packages:${PYTHONPATH:-}"

mkdir -p "${RUN_ROOT}/slurm" "${HF_HOME}" "${TORCH_HOME}" "${TMPDIR}" data/fid_stats

GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
NNODES="${SLURM_NNODES:-2}"
MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
MASTER_PORT="${MASTER_PORT:-29604}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DATA_ROOT="${DATA_ROOT:-/leonardo_scratch/fast/EUHPC_D32_028/dongli/huggingface/imagenet-1k}"
STATS_NAME="self_jit_b_block11_t005_in256_patch_stats.npz"

echo "[$(date --iso-8601=seconds)] starting full JiT-B patch self-FD stats"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-manual}"
echo "GIT_COMMIT=$(git rev-parse HEAD)"
echo "GIT_STATUS_START"
git status --short
echo "GIT_STATUS_END"
echo "NNODES=${NNODES} GPUS_PER_NODE=${GPUS_PER_NODE} WORLD_SIZE=$(( NNODES * GPUS_PER_NODE ))"
echo "BATCH_SIZE=${BATCH_SIZE} NUM_WORKERS=${NUM_WORKERS}"
echo "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
echo "DATA_ROOT=${DATA_ROOT}"

srun --ntasks="${NNODES}" --ntasks-per-node=1 bash -lc "\
    torchrun \
      --nnodes='${NNODES}' \
      --nproc_per_node='${GPUS_PER_NODE}' \
      --rdzv_id='${SLURM_JOB_ID:-manual}' \
      --rdzv_backend=c10d \
      --rdzv_endpoint='${MASTER_ADDR}:${MASTER_PORT}' \
      compute_self_pmf_stats.py \
      --data_path '${DATA_ROOT}' \
      --load_from checkpoints/base/JiT-B.pth \
      --output_dir data/fid_stats \
      --output_name '${STATS_NAME}' \
      --batch_size '${BATCH_SIZE}' \
      --num_workers '${NUM_WORKERS}' \
      --model JiT_B --rope_2d --learned_pe --legacy_time_convention \
      --fd_self_jit_block 11 --fd_self_t 0.05 --fd_self_pool patch"

python - <<'PY'
import numpy as np
path = "data/fid_stats/self_jit_b_block11_t005_in256_patch_stats.npz"
arr = np.load(path)
print(f"stats_path={path}")
print(f"mu_shape={arr['mu'].shape}")
print(f"sigma_shape={arr['sigma'].shape}")
print(f"sigma_trace={float(np.trace(arr['sigma']))}")
PY

echo "[$(date --iso-8601=seconds)] finished full JiT-B patch self-FD stats"
