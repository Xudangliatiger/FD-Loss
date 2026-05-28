#!/bin/bash
#SBATCH --job-name=jitB_svaep5k
#SBATCH --account=EUHPC_D32_028
#SBATCH --partition=boost_usr_prod
#SBATCH --time=02:00:00
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
export PYTHONPATH="/leonardo_scratch/fast/EUHPC_D32_028/dongli/envs/ggn-cu126-py311/lib/python3.11/site-packages:${PYTHONPATH:-}"

mkdir -p "${RUN_ROOT}/slurm" "${RUN_ROOT}/jit_sphere_vae_start_frozen_post5k_vis" \
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
GLOBAL_BSZ="${GLOBAL_BSZ:-64}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
BATCH_SIZE=$(( GLOBAL_BSZ / GPUS_PER_NODE / GRAD_ACCUM_STEPS ))
MASTER_PORT="${MASTER_PORT:-29991}"
POST_STEPS="${POST_STEPS:-5000}"
TC="${TC:-0.30}"
KL_WEIGHT="${KL_WEIGHT:-5.0}"
START_SUPPORT_MODE="${START_SUPPORT_MODE:-pixel_sphere}"
VIS_EVERY="${VIS_EVERY:-1000}"
NUM_WORKERS="${NUM_WORKERS:-4}"
VAE_CKPT="${VAE_CKPT:-${RUN_ROOT}/jit_sphere_vae_pretrain_vis/JiT_B-vae-only-pixel_sphere-kw5-steps1000-42873095/checkpoints/vae_encoder_latest.pth}"
EXP_NAME="${EXP_NAME:-JiT_B-sphere-vae-frozenenc-kw5-pre1k-tc0p30-post5k-${SLURM_JOB_ID:-manual}}"

if [[ ! -f "${VAE_CKPT}" ]]; then
    echo "[ERR] missing VAE encoder checkpoint: ${VAE_CKPT}" >&2
    exit 3
fi

echo "[$(date --iso-8601=seconds)] starting JiT-B frozen-encoder VAE-start posttraining diagnostic"
echo "GIT_COMMIT=$(git rev-parse HEAD)"
echo "DATA_ROOT=${DATA_ROOT}"
echo "VAE_CKPT=${VAE_CKPT}"
echo "GLOBAL_BSZ=${GLOBAL_BSZ} BATCH_SIZE=${BATCH_SIZE} GPUS=${GPUS_PER_NODE} GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS}"
echo "POST_STEPS=${POST_STEPS} TC=${TC} KL_WEIGHT=${KL_WEIGHT} START_SUPPORT_MODE=${START_SUPPORT_MODE} VIS_EVERY=${VIS_EVERY}"

torchrun --nproc_per_node="${GPUS_PER_NODE}" --master_port="${MASTER_PORT}" \
    main_jit_vae_start.py \
    --project jit_sphere_vae_start_frozen_post5k_vis \
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
    --print_freq 20 --milestone_interval 0 --save_freq 10 \
    --epochs 1 --steps_per_epoch "${POST_STEPS}" --warmup_epochs 1 \
    --lr 1e-5 --lr_sched cosine --min_lr 0.0 \
    --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
    --vae_start_tc "${TC}" \
    --vae_start_pre_steps 0 \
    --vae_start_encoder_ckpt "${VAE_CKPT}" \
    --freeze_vae_start_encoder \
    --vae_start_kl_weight "${KL_WEIGHT}" \
    --vae_start_cycle_weight 1.0 \
    --vae_start_hidden 64 \
    --vae_start_lr 2e-4 \
    --vae_start_sample_mode posterior \
    --vae_start_vis_every "${VIS_EVERY}" \
    --vae_start_vis_images 16 \
    --train_t_min 0.02 --train_t_max 1.0 \
    --num_workers "${NUM_WORKERS}" \
    --disable_wandb

echo "[$(date --iso-8601=seconds)] finished JiT-B frozen-encoder VAE-start posttraining diagnostic"
