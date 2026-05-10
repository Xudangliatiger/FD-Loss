#!/bin/bash
#SBATCH --job-name=fdloss_jit_dbg
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

mkdir -p "${RUN_ROOT}/slurm" "${RUN_ROOT}/debug" \
    "${HF_HOME}" "${TORCH_HOME}" "${TMPDIR}"

EXP_NAME="JiT_B-debug-one-${SLURM_JOB_ID:-manual}"

echo "[$(date --iso-8601=seconds)] generating one debug image"

torchrun --nproc_per_node=1 --master_port="${MASTER_PORT:-29503}" \
    eval_all_fds.py \
    --gen_only \
    --model JiT_B \
    --rope_2d --learned_pe --legacy_time_convention --ema_type edm \
    --cfg 3.0 --cfg_list 3.0 \
    --interval_min 0.1 --interval_max 1.0 \
    --num_sampling_steps 1 \
    --eval_ema_labels online \
    --disable_wandb \
    --eval_bsz 1 \
    --num_images 1 \
    --num_images_for_eval_and_search 1 \
    --load_from checkpoints/post-trained/JiT-B_FD-Inception.pth \
    --output_dir "${RUN_ROOT}/debug" \
    --project eval_released \
    --exp_name "${EXP_NAME}"

echo "[$(date --iso-8601=seconds)] debug image directory:"
find "${RUN_ROOT}/debug/eval_released/${EXP_NAME}/gen_images" -type f -name '*.png' -print
