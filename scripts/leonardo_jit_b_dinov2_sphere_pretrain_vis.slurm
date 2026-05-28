#!/bin/bash
#SBATCH --job-name=jitB_dinovis
#SBATCH --account=EUHPC_D32_028
#SBATCH --partition=boost_usr_prod
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=192G
#SBATCH --exclude=lrdn2272,lrdn3214,lrdn2118
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

mkdir -p "${RUN_ROOT}/slurm" "${RUN_ROOT}/jit_dinov2_sphere_pretrain_vis" \
    "${HF_HOME}" "${TORCH_HOME}" "${TMPDIR}"

DATA_ROOT="${DATA_ROOT:-/leonardo_scratch/fast/EUHPC_D32_028/dongli/huggingface/imagenet-1k}"
DATA_ARGS=()
if [[ -d "${DATA_ROOT}/train" || -d "${DATA_ROOT}/data" ]]; then
    DATA_ARGS=(--data_path "${DATA_ROOT}")
elif [[ -d "${DATA_ROOT}/n01440764" ]]; then
    DATA_ARGS=(--data_path "__unused_imagefolder_root__" --image_list data/train.txt --image_root "${DATA_ROOT}")
else
    echo "[ERR] Could not find ImageNet train images under DATA_ROOT=${DATA_ROOT}" >&2
    exit 2
fi

GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
GLOBAL_BSZ="${GLOBAL_BSZ:-32}"
BATCH_SIZE=$(( GLOBAL_BSZ / GPUS_PER_NODE ))
MASTER_PORT="${MASTER_PORT:-30011}"
VAE_STEPS="${VAE_STEPS:-1000}"
VIS_EVERY="${VIS_EVERY:-250}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
KL_WEIGHT="${KL_WEIGHT:-5.0}"
START_SUPPORT_MODE="${START_SUPPORT_MODE:-pixel_sphere}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DINO_MODEL="${DINO_MODEL:-vit_base_patch14_dinov2.lvd142m}"
DINO_PATCH_SIZE="${DINO_PATCH_SIZE:-14}"
DINO_LATENT_TOKENS="${DINO_LATENT_TOKENS:-256}"
DINO_TOKEN_DIM="${DINO_TOKEN_DIM:-64}"

KL_TAG="${KL_WEIGHT/./p}"
EXP_NAME="${EXP_NAME:-JiT_B-dinov2-latent-start-${START_SUPPORT_MODE}-kw${KL_TAG}-steps${VAE_STEPS}-${SLURM_JOB_ID:-manual}}"

echo "[$(date --iso-8601=seconds)] starting JiT-B DINOv2 latent-token start diagnostic"
echo "GIT_COMMIT=$(git rev-parse HEAD)"
echo "DATA_ROOT=${DATA_ROOT}"
echo "GLOBAL_BSZ=${GLOBAL_BSZ} BATCH_SIZE=${BATCH_SIZE} GPUS=${GPUS_PER_NODE}"
echo "DINO_MODEL=${DINO_MODEL} DINO_PATCH_SIZE=${DINO_PATCH_SIZE} DINO_LATENT_TOKENS=${DINO_LATENT_TOKENS} DINO_TOKEN_DIM=${DINO_TOKEN_DIM}"
echo "VAE_STEPS=${VAE_STEPS} KL_WEIGHT=${KL_WEIGHT} START_SUPPORT_MODE=${START_SUPPORT_MODE}"

torchrun --nproc_per_node="${GPUS_PER_NODE}" --master_port="${MASTER_PORT}" \
    main_jit_vae_start_pretrain_vis.py \
    --project jit_dinov2_sphere_pretrain_vis \
    --exp_name "${EXP_NAME}" \
    --output_dir "${RUN_ROOT}" \
    --batch_size "${BATCH_SIZE}" \
    "${DATA_ARGS[@]}" \
    --load_from checkpoints/base/JiT-B.pth \
    --model JiT_B --rope_2d --learned_pe --legacy_time_convention \
    --cfg 3.0 --interval_min 0.1 --interval_max 1.0 \
    --start_support_mode "${START_SUPPORT_MODE}" \
    --ema_type edm \
    --num_sampling_steps 1 \
    --vae_start_encoder_type dinov2_latent \
    --dinov2_start_model "${DINO_MODEL}" \
    --dinov2_start_patch_size "${DINO_PATCH_SIZE}" \
    --dinov2_start_latent_tokens "${DINO_LATENT_TOKENS}" \
    --dinov2_start_token_dim "${DINO_TOKEN_DIM}" \
    --vae_only_steps "${VAE_STEPS}" \
    --vae_vis_every "${VIS_EVERY}" \
    --vae_save_every "${SAVE_EVERY}" \
    --num_vis_images 16 \
    --vae_start_kl_weight "${KL_WEIGHT}" \
    --vae_start_cycle_weight 1.0 \
    --vae_start_lr 2e-4 \
    --vae_start_sample_mode posterior \
    --epochs 1 --steps_per_epoch "${VAE_STEPS}" --warmup_epochs 0 \
    --num_workers "${NUM_WORKERS}" \
    --print_freq 50 \
    --disable_wandb

echo "[$(date --iso-8601=seconds)] finished JiT-B DINOv2 latent-token start diagnostic"
