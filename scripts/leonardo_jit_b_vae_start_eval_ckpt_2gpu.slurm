#!/bin/bash
#SBATCH --job-name=jitB_vae_eval
#SBATCH --account=EUHPC_D32_028
#SBATCH --partition=boost_usr_prod
#SBATCH --time=04:00:00
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

mkdir -p "${RUN_ROOT}/slurm" "${RUN_ROOT}/jit_vae_start_mid_eval" \
    "${HF_HOME}" "${TORCH_HOME}" "${TMPDIR}"

GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29547}"
NUM_IMAGES="${NUM_IMAGES:-50000}"
EVAL_BSZ="${EVAL_BSZ:-64}"
EXP_NAME="${EXP_NAME:-JiT_B-vae-start-tc0p30-kw1p00-gbsz1024-full}"
CKPT_STEP="${CKPT_STEP:-0050000}"
CKPT_PATH="${CKPT_PATH:-${RUN_ROOT}/jit_vae_start/${EXP_NAME}/checkpoints/step_${CKPT_STEP}.pth}"
CKPT_TAG="$(basename "${CKPT_PATH}" .pth)"

if [[ ! -f "${CKPT_PATH}" ]]; then
    echo "[ERR] checkpoint not found: ${CKPT_PATH}" >&2
    exit 2
fi

echo "[$(date --iso-8601=seconds)] evaluating JiT-B VAE-start checkpoint"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-manual}"
echo "PWD=$(pwd)"
echo "GIT_COMMIT=$(git rev-parse HEAD)"
echo "CKPT_PATH=${CKPT_PATH}"
echo "NUM_IMAGES=${NUM_IMAGES} EVAL_BSZ=${EVAL_BSZ}"

torchrun --nproc_per_node="${GPUS_PER_NODE}" --master_port="${MASTER_PORT}" \
    eval_all_fds.py \
    --model JiT_B \
    --rope_2d --learned_pe --legacy_time_convention --ema_type edm \
    --cfg 3.0 --cfg_list 3.0 \
    --interval_min 0.1 --interval_max 1.0 \
    --num_sampling_steps 1 \
    --eval_ema_labels online \
    --disable_wandb --no_prc \
    --models inception \
    --eval_bsz "${EVAL_BSZ}" \
    --num_images "${NUM_IMAGES}" \
    --num_images_for_eval_and_search "${NUM_IMAGES}" \
    --load_from "${CKPT_PATH}" \
    --output_dir "${RUN_ROOT}/jit_vae_start_mid_eval" \
    --project eval_vae_start_mid \
    --exp_name "${EXP_NAME}-${CKPT_TAG}-fid-${NUM_IMAGES}-${SLURM_JOB_ID:-manual}"

echo "[$(date --iso-8601=seconds)] finished JiT-B VAE-start checkpoint eval"
