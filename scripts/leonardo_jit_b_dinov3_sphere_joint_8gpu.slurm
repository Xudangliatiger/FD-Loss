#!/bin/bash
#SBATCH --job-name=jitB_dino8
#SBATCH --account=EUHPC_D32_028
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --time=2-00:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
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
if [[ -d "${DATA_ROOT}/train" || -d "${DATA_ROOT}/data" ]]; then
    DATA_ARG_STRING="--data_path ${DATA_ROOT}"
elif [[ -d "${DATA_ROOT}/n01440764" ]]; then
    DATA_ARG_STRING="--data_path __unused_imagefolder_root__ --image_list data/train.txt --image_root ${DATA_ROOT}"
else
    echo "[ERR] Could not find ImageNet train images under DATA_ROOT=${DATA_ROOT}" >&2
    exit 2
fi

GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
NNODES="${SLURM_NNODES:-2}"
WORLD_SIZE=$(( NNODES * GPUS_PER_NODE ))
GLOBAL_BSZ="${GLOBAL_BSZ:-1024}"
# DINO bridge micro-batch 128/GPU is too close to the A100-64G limit. Keep the
# effective FD-Loss global batch while using the validated 64/GPU micro-batch.
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
MICRO_GLOBAL_BSZ=$(( GLOBAL_BSZ / GRAD_ACCUM_STEPS ))
BATCH_SIZE=$(( MICRO_GLOBAL_BSZ / WORLD_SIZE ))
MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
MASTER_PORT="${MASTER_PORT:-30031}"

TC="${TC:-0.30}"
PRE_STEPS="${PRE_STEPS:-0}"
POST_STEPS="${POST_STEPS:-125000}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-1250}"
EPOCHS=$(( (POST_STEPS + STEPS_PER_EPOCH - 1) / STEPS_PER_EPOCH ))
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
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
CYCLE_L1_WEIGHT="${CYCLE_L1_WEIGHT:-1.0}"
CYCLE_L2_WEIGHT="${CYCLE_L2_WEIGHT:-1.0}"
CYCLE_LPIPS_WEIGHT="${CYCLE_LPIPS_WEIGHT:-1.0}"
CLEAN_BRIDGE_KL_WEIGHT="${CLEAN_BRIDGE_KL_WEIGHT:-0.0}"
CLEAN_BRIDGE_KL_GRANULARITY="${CLEAN_BRIDGE_KL_GRANULARITY:-sample}"
VAE_START_VIS_EVERY="${VAE_START_VIS_EVERY:-0}"
INF_EVERY="${INF_EVERY:-1000}"
INF_IMAGES="${INF_IMAGES:-16}"
LPIPS_CKPT_DIR="${LPIPS_CKPT_DIR:-/leonardo_scratch/fast/EUHPC_D32_028/dongli/huggingface/lpips-vgg}"
VGG16_CKPT="${VGG16_CKPT:-/leonardo_scratch/fast/EUHPC_D32_028/dongli/huggingface/vgg16-imagenet1k/vgg16-397923af.pth}"
LOAD_FROM="${LOAD_FROM:-checkpoints/base/JiT-B.pth}"
VAE_CKPT="${VAE_CKPT:-}"
VAE_CKPT_ARGS=""
if [[ -n "${VAE_CKPT}" ]]; then
    VAE_CKPT_ARGS="--vae_start_encoder_ckpt ${VAE_CKPT}"
fi

if (( GLOBAL_BSZ % GRAD_ACCUM_STEPS != 0 )); then
    echo "[ERR] GLOBAL_BSZ=${GLOBAL_BSZ} must be divisible by GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS}" >&2
    exit 2
fi
if (( MICRO_GLOBAL_BSZ % WORLD_SIZE != 0 )); then
    echo "[ERR] MICRO_GLOBAL_BSZ=${MICRO_GLOBAL_BSZ} must be divisible by WORLD_SIZE=${WORLD_SIZE}" >&2
    exit 2
fi

mkdir -p "${LPIPS_CKPT_DIR}" "${TORCH_HOME}/hub/checkpoints"
if [[ -f "${VGG16_CKPT}" && ! -e "${TORCH_HOME}/hub/checkpoints/vgg16-397923af.pth" ]]; then
    ln -s "${VGG16_CKPT}" "${TORCH_HOME}/hub/checkpoints/vgg16-397923af.pth"
fi

EXP_NAME="${EXP_NAME:-JiT_B-dinov3-patch-sphere-vitbridge-joint-l1l2lpips1-gbsz1024-post125k-${SLURM_JOB_ID:-manual}}"

echo "[$(date --iso-8601=seconds)] starting JiT-B DINOv3 sphere joint 8GPU post-training"
echo "GIT_COMMIT=$(git rev-parse HEAD)"
echo "GIT_STATUS_START"
git status --short || true
echo "GIT_STATUS_END"
echo "DATA_ROOT=${DATA_ROOT}"
echo "DATA_ARG_STRING=${DATA_ARG_STRING}"
echo "NNODES=${NNODES} GPUS_PER_NODE=${GPUS_PER_NODE} WORLD_SIZE=${WORLD_SIZE}"
echo "GLOBAL_BSZ=${GLOBAL_BSZ} GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS}"
echo "MICRO_GLOBAL_BSZ=${MICRO_GLOBAL_BSZ} BATCH_SIZE=${BATCH_SIZE}"
echo "TC=${TC} PRE_STEPS=${PRE_STEPS} POST_STEPS=${POST_STEPS} EPOCHS=${EPOCHS} STEPS_PER_EPOCH=${STEPS_PER_EPOCH} WARMUP_EPOCHS=${WARMUP_EPOCHS}"
echo "DINO_MODEL=${DINO_MODEL} DINO_PRETRAINED_PATH=${DINO_PRETRAINED_PATH}"
echo "DINO_NOISE_ANGLE=${DINO_NOISE_ANGLE} DINO_DECODER_DEPTH=${DINO_DECODER_DEPTH} DINO_DECODER_HEADS=${DINO_DECODER_HEADS}"
echo "CYCLE_L1_WEIGHT=${CYCLE_L1_WEIGHT} CYCLE_L2_WEIGHT=${CYCLE_L2_WEIGHT} CYCLE_LPIPS_WEIGHT=${CYCLE_LPIPS_WEIGHT}"
echo "CLEAN_BRIDGE_KL_WEIGHT=${CLEAN_BRIDGE_KL_WEIGHT} CLEAN_BRIDGE_KL_GRANULARITY=${CLEAN_BRIDGE_KL_GRANULARITY}"
echo "VAE_START_VIS_EVERY=${VAE_START_VIS_EVERY} INF_EVERY=${INF_EVERY} INF_IMAGES=${INF_IMAGES}"
echo "LOAD_FROM=${LOAD_FROM}"
echo "VAE_CKPT=${VAE_CKPT:-<none>} START_SUPPORT_MODE=${START_SUPPORT_MODE}"
echo "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"

srun --ntasks="${NNODES}" --ntasks-per-node=1 bash -lc "\
    mkdir -p '${TMPDIR}' && \
    torchrun \
      --nnodes='${NNODES}' \
      --nproc_per_node='${GPUS_PER_NODE}' \
      --rdzv_id='${SLURM_JOB_ID:-manual}' \
      --rdzv_backend=c10d \
      --rdzv_endpoint='${MASTER_ADDR}:${MASTER_PORT}' \
      main_jit_vae_start.py \
      --project jit_dinov3_sphere_joint \
      --exp_name '${EXP_NAME}' \
      --output_dir '${RUN_ROOT}' \
      --batch_size '${BATCH_SIZE}' \
      ${DATA_ARG_STRING} \
      --load_from '${LOAD_FROM}' \
      --model JiT_B --rope_2d --learned_pe --legacy_time_convention \
      --cfg 3.0 --interval_min 0.1 --interval_max 1.0 \
      --start_support_mode '${START_SUPPORT_MODE}' \
      --ema_type edm \
      --num_sampling_steps 1 \
      --print_freq 20 --milestone_interval 10 --save_freq 5 \
      --epochs '${EPOCHS}' --steps_per_epoch '${STEPS_PER_EPOCH}' --warmup_epochs '${WARMUP_EPOCHS}' \
      --lr 1e-5 --lr_sched cosine --min_lr 0.0 \
      --gradient_accumulation_steps '${GRAD_ACCUM_STEPS}' \
      --vae_start_tc '${TC}' \
      --vae_start_pre_steps '${PRE_STEPS}' \
      --vae_start_encoder_type dino_patch_sphere \
      --dinov2_start_model '${DINO_MODEL}' \
      --dinov2_start_pretrained_path '${DINO_PRETRAINED_PATH}' \
      --dinov2_start_patch_size '${DINO_PATCH_SIZE}' \
      --dinov2_start_latent_tokens '${DINO_LATENT_TOKENS}' \
      --dinov2_start_token_dim '${DINO_TOKEN_DIM}' \
      --dinov2_start_bridge_type vit_decoder \
      --dinov2_start_decoder_depth '${DINO_DECODER_DEPTH}' \
      --dinov2_start_decoder_heads '${DINO_DECODER_HEADS}' \
      --dinov2_start_noise_angle '${DINO_NOISE_ANGLE}' \
      ${VAE_CKPT_ARGS} \
      --vae_start_kl_weight 0.0 \
      --vae_start_cycle_weight 1.0 \
      --vae_start_cycle_l1_weight '${CYCLE_L1_WEIGHT}' \
      --vae_start_cycle_l2_weight '${CYCLE_L2_WEIGHT}' \
      --vae_start_cycle_lpips_weight '${CYCLE_LPIPS_WEIGHT}' \
      --vae_start_clean_bridge_kl_weight '${CLEAN_BRIDGE_KL_WEIGHT}' \
      --vae_start_clean_bridge_kl_granularity '${CLEAN_BRIDGE_KL_GRANULARITY}' \
      --vae_start_lpips_ckpt_dir '${LPIPS_CKPT_DIR}' \
      --vae_start_lr 2e-4 \
      --vae_start_sample_mode posterior \
      --vae_start_vis_every '${VAE_START_VIS_EVERY}' \
      --vae_start_vis_images 8 \
      --vae_start_inf_every '${INF_EVERY}' \
      --vae_start_inf_images '${INF_IMAGES}' \
      --train_t_min 0.02 --train_t_max 1.0 \
      --num_workers '${NUM_WORKERS}' \
      --disable_wandb"

echo "[$(date --iso-8601=seconds)] finished JiT-B DINOv3 sphere joint 8GPU post-training"
