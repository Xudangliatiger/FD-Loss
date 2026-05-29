#!/bin/bash
#SBATCH --job-name=jitB_dinojoint
#SBATCH --account=EUHPC_D32_028
#SBATCH --partition=boost_usr_prod
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
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

mkdir -p "${RUN_ROOT}/slurm" "${RUN_ROOT}/jit_dinov3_sphere_joint" \
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

GPUS_PER_NODE=1
GLOBAL_BSZ="${GLOBAL_BSZ:-4}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MASTER_PORT="${MASTER_PORT:-30023}"
TC="${TC:-0.30}"
PRE_STEPS="${PRE_STEPS:-0}"
POST_STEPS="${POST_STEPS:-5000}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-500}"
EPOCHS=$(( (POST_STEPS + STEPS_PER_EPOCH - 1) / STEPS_PER_EPOCH ))
NUM_WORKERS="${NUM_WORKERS:-4}"
START_SUPPORT_MODE="${START_SUPPORT_MODE:-pixel_sphere}"
DINO_MODEL="${DINO_MODEL:-vit_base_patch16_dinov3}"
DINO_PRETRAINED_PATH="${DINO_PRETRAINED_PATH:-/leonardo_scratch/fast/EUHPC_D32_028/dongli/huggingface/dinov3-vitb16-pretrain-lvd1689m/model.safetensors}"
DINO_PATCH_SIZE="${DINO_PATCH_SIZE:-16}"
DINO_LATENT_TOKENS="${DINO_LATENT_TOKENS:-256}"
DINO_TOKEN_DIM="${DINO_TOKEN_DIM:-768}"
DINO_NOISE_ANGLE="${DINO_NOISE_ANGLE:-85.0}"
DINO_DECODER_DEPTH="${DINO_DECODER_DEPTH:-4}"
DINO_DECODER_HEADS="${DINO_DECODER_HEADS:-12}"
CYCLE_L1_WEIGHT="${CYCLE_L1_WEIGHT:-0.0}"
CYCLE_L2_WEIGHT="${CYCLE_L2_WEIGHT:-1.0}"
CYCLE_LPIPS_WEIGHT="${CYCLE_LPIPS_WEIGHT:-0.0}"
LPIPS_CKPT_DIR="${LPIPS_CKPT_DIR:-/leonardo_scratch/fast/EUHPC_D32_028/dongli/huggingface/lpips-vgg}"
VGG16_CKPT="${VGG16_CKPT:-/leonardo_scratch/fast/EUHPC_D32_028/dongli/huggingface/vgg16-imagenet1k/vgg16-397923af.pth}"
LOAD_FROM="${LOAD_FROM:-checkpoints/base/JiT-B.pth}"
VAE_CKPT="${VAE_CKPT:-}"
VAE_CKPT_ARGS=()
if [[ -n "${VAE_CKPT}" ]]; then
    VAE_CKPT_ARGS=(--vae_start_encoder_ckpt "${VAE_CKPT}")
fi
mkdir -p "${LPIPS_CKPT_DIR}" "${TORCH_HOME}/hub/checkpoints"
if [[ -f "${VGG16_CKPT}" && ! -e "${TORCH_HOME}/hub/checkpoints/vgg16-397923af.pth" ]]; then
    ln -s "${VGG16_CKPT}" "${TORCH_HOME}/hub/checkpoints/vgg16-397923af.pth"
fi

EXP_NAME="${EXP_NAME:-JiT_B-dinov3-patch-sphere-vitbridge-joint-tc${TC/./p}-pre${PRE_STEPS}-post${POST_STEPS}-${SLURM_JOB_ID:-manual}}"

echo "[$(date --iso-8601=seconds)] starting JiT-B DINOv3 sphere joint post-training"
echo "GIT_COMMIT=$(git rev-parse HEAD)"
echo "DATA_ROOT=${DATA_ROOT}"
echo "GLOBAL_BSZ=${GLOBAL_BSZ} BATCH_SIZE=${BATCH_SIZE} GPUS=${GPUS_PER_NODE}"
echo "TC=${TC} PRE_STEPS=${PRE_STEPS} POST_STEPS=${POST_STEPS} EPOCHS=${EPOCHS} STEPS_PER_EPOCH=${STEPS_PER_EPOCH}"
echo "DINO_MODEL=${DINO_MODEL} DINO_PRETRAINED_PATH=${DINO_PRETRAINED_PATH}"
echo "DINO_NOISE_ANGLE=${DINO_NOISE_ANGLE} DINO_DECODER_DEPTH=${DINO_DECODER_DEPTH} DINO_DECODER_HEADS=${DINO_DECODER_HEADS}"
echo "CYCLE_L1_WEIGHT=${CYCLE_L1_WEIGHT} CYCLE_L2_WEIGHT=${CYCLE_L2_WEIGHT} CYCLE_LPIPS_WEIGHT=${CYCLE_LPIPS_WEIGHT}"
echo "LPIPS_CKPT_DIR=${LPIPS_CKPT_DIR}"
echo "LOAD_FROM=${LOAD_FROM}"
echo "VAE_CKPT=${VAE_CKPT:-<none>} START_SUPPORT_MODE=${START_SUPPORT_MODE}"

torchrun --nproc_per_node="${GPUS_PER_NODE}" --master_port="${MASTER_PORT}" \
    main_jit_vae_start.py \
    --project jit_dinov3_sphere_joint \
    --exp_name "${EXP_NAME}" \
    --output_dir "${RUN_ROOT}" \
    --batch_size "${BATCH_SIZE}" \
    "${DATA_ARGS[@]}" \
    --load_from "${LOAD_FROM}" \
    --model JiT_B --rope_2d --learned_pe --legacy_time_convention \
    --cfg 3.0 --interval_min 0.1 --interval_max 1.0 \
    --start_support_mode "${START_SUPPORT_MODE}" \
    --ema_type edm \
    --num_sampling_steps 1 \
    --print_freq 100 --milestone_interval 0 --save_freq 10 \
    --epochs "${EPOCHS}" --steps_per_epoch "${STEPS_PER_EPOCH}" --warmup_epochs 1 \
    --lr 1e-5 --lr_sched cosine --min_lr 0.0 \
    --vae_start_tc "${TC}" \
    --vae_start_pre_steps "${PRE_STEPS}" \
    --vae_start_encoder_type dino_patch_sphere \
    --dinov2_start_model "${DINO_MODEL}" \
    --dinov2_start_pretrained_path "${DINO_PRETRAINED_PATH}" \
    --dinov2_start_patch_size "${DINO_PATCH_SIZE}" \
    --dinov2_start_latent_tokens "${DINO_LATENT_TOKENS}" \
    --dinov2_start_token_dim "${DINO_TOKEN_DIM}" \
    --dinov2_start_bridge_type vit_decoder \
    --dinov2_start_decoder_depth "${DINO_DECODER_DEPTH}" \
    --dinov2_start_decoder_heads "${DINO_DECODER_HEADS}" \
    --dinov2_start_noise_angle "${DINO_NOISE_ANGLE}" \
    "${VAE_CKPT_ARGS[@]}" \
    --vae_start_kl_weight 0.0 \
    --vae_start_cycle_weight 1.0 \
    --vae_start_cycle_l1_weight "${CYCLE_L1_WEIGHT}" \
    --vae_start_cycle_l2_weight "${CYCLE_L2_WEIGHT}" \
    --vae_start_cycle_lpips_weight "${CYCLE_LPIPS_WEIGHT}" \
    --vae_start_lpips_ckpt_dir "${LPIPS_CKPT_DIR}" \
    --vae_start_lr 2e-4 \
    --vae_start_sample_mode posterior \
    --vae_start_vis_every 1000 \
    --vae_start_vis_images 8 \
    --train_t_min 0.02 --train_t_max 1.0 \
    --num_workers "${NUM_WORKERS}" \
    --disable_wandb

echo "[$(date --iso-8601=seconds)] finished JiT-B DINOv3 sphere joint post-training"
