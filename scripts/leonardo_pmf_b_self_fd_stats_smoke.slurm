#!/bin/bash
#SBATCH --job-name=pmf_self_stats
#SBATCH --account=EUHPC_D32_028
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
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
export TMPDIR="${SCRATCH_ROOT}/tmp"
export WANDB_MODE=disabled
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="/leonardo_scratch/fast/EUHPC_D32_028/dongli/envs/ggn-cu126-py311/lib/python3.11/site-packages:${PYTHONPATH:-}"

mkdir -p "${RUN_ROOT}/slurm" "${HF_HOME}" "${TORCH_HOME}" "${TMPDIR}" data/fid_stats

GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29561}"
NUM_IMAGES="${NUM_IMAGES:-4096}"
BATCH_SIZE="${BATCH_SIZE:-64}"
DATA_ROOT="${DATA_ROOT:-/leonardo_scratch/fast/EUHPC_D32_028/dongli/huggingface/imagenet-1k}"
STATS_NAME="self_pmf_b_shared7_t005_in256_stats.npz"

echo "[$(date --iso-8601=seconds)] starting pMF-B self-FD stats smoke"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-manual}"
echo "PWD=$(pwd)"
echo "GIT_COMMIT=$(git rev-parse HEAD)"
echo "GIT_STATUS_START"
git status --short
echo "GIT_STATUS_END"
echo "GPUS_PER_NODE=${GPUS_PER_NODE} NUM_IMAGES=${NUM_IMAGES} BATCH_SIZE=${BATCH_SIZE}"
echo "DATA_ROOT=${DATA_ROOT}"

python - <<'PY'
import pyarrow
print(f"pyarrow={pyarrow.__version__}")
PY

torchrun --nproc_per_node="${GPUS_PER_NODE}" --master_port="${MASTER_PORT}" \
    compute_self_pmf_stats.py \
    --data_path "${DATA_ROOT}" \
    --load_from checkpoints/base/pMF-B_256.pth \
    --output_dir data/fid_stats \
    --output_name "${STATS_NAME}" \
    --num_images "${NUM_IMAGES}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers 8 \
    --model pMF_B --rope_2d --learned_pe --disable_v_head \
    --fd_self_shared_block 7 --fd_self_t 0.05 \
    --cfg 8.5 --interval_min 0.1 --interval_max 0.7

python - <<'PY'
import numpy as np
path = "data/fid_stats/self_pmf_b_shared7_t005_in256_stats.npz"
arr = np.load(path)
print(f"stats_path={path}")
print(f"mu_shape={arr['mu'].shape}")
print(f"sigma_shape={arr['sigma'].shape}")
PY

echo "[$(date --iso-8601=seconds)] finished pMF-B self-FD stats smoke"
