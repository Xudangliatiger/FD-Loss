#!/bin/bash
#SBATCH --job-name=fdloss_pmf_dbg
#SBATCH --account=EUHPC_D32_028
#SBATCH --partition=boost_usr_prod
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONDONTWRITEBYTECODE=1

mkdir -p "${RUN_ROOT}/slurm" "${RUN_ROOT}/pmf_b_debug_images" \
    "${HF_HOME}" "${TORCH_HOME}" "${TMPDIR}"

NUM_IMAGES="${NUM_IMAGES:-16}"
EVAL_BSZ="${EVAL_BSZ:-8}"
EXP_NAME="pMF_B-debug-images-${NUM_IMAGES}-${SLURM_JOB_ID:-manual}"

echo "[$(date --iso-8601=seconds)] generating ${NUM_IMAGES} pMF-B debug images"

torchrun --nproc_per_node=1 --master_port="${MASTER_PORT:-29513}" \
    eval_all_fds.py \
    --gen_only \
    --model pMF_B \
    --rope_2d --learned_pe --disable_v_head \
    --cfg 8.5 --cfg_list 8.5 \
    --interval_min 0.1 --interval_max 0.7 \
    --num_sampling_steps 1 \
    --eval_ema_labels online \
    --disable_wandb \
    --eval_bsz "${EVAL_BSZ}" \
    --num_images "${NUM_IMAGES}" \
    --num_images_for_eval_and_search "${NUM_IMAGES}" \
    --load_from checkpoints/post-trained/pMF-B_FD-Inception.pth \
    --output_dir "${RUN_ROOT}/pmf_b_debug_images" \
    --project eval_released \
    --exp_name "${EXP_NAME}"

echo "[$(date --iso-8601=seconds)] generated image files:"
find "${RUN_ROOT}/pmf_b_debug_images/eval_released/${EXP_NAME}/gen_images" -type f -name '*.png' -print | sort
