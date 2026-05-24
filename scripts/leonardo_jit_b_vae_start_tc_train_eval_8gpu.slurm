#!/bin/bash
#SBATCH --job-name=jitB_vae8
#SBATCH --account=EUHPC_D32_028
#SBATCH --partition=boost_usr_prod
#SBATCH --time=1-00:00:00
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

mkdir -p "${RUN_ROOT}/slurm" "${RUN_ROOT}/jit_vae_start" "${RUN_ROOT}/jit_vae_start_eval" \
    "${HF_HOME}" "${TORCH_HOME}" "${TMPDIR}"

DATA_ROOT="${DATA_ROOT:-/leonardo_scratch/fast/EUHPC_D32_028/dongli/huggingface/imagenet-1k}"
if [[ -d "${DATA_ROOT}/train" || -d "${DATA_ROOT}/data" ]]; then
    DATA_ARGS_STRING="--data_path ${DATA_ROOT}"
elif [[ -d "${DATA_ROOT}/n01440764" ]]; then
    DATA_ARGS_STRING="--data_path __unused_imagefolder_root__ --image_list data/train.txt --image_root ${DATA_ROOT}"
else
    echo "[ERR] Could not find ImageNet train images under DATA_ROOT=${DATA_ROOT}" >&2
    echo "      Expected ImageFolder train/, HF parquet data/train-*.parquet, or class folders." >&2
    exit 2
fi

GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
NNODES="${SLURM_NNODES:-2}"
WORLD_SIZE=$(( NNODES * GPUS_PER_NODE ))
GLOBAL_BSZ="${GLOBAL_BSZ:-1024}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
MICRO_GLOBAL_BSZ=$(( GLOBAL_BSZ / GRAD_ACCUM_STEPS ))
BATCH_SIZE=$(( MICRO_GLOBAL_BSZ / WORLD_SIZE ))
MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
MASTER_PORT="${MASTER_PORT:-29539}"
TC="${TC:-0.30}"
KL_WEIGHT="${KL_WEIGHT:-1.0}"
PRE_STEPS="${PRE_STEPS:-2000}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-1250}"
EPOCHS="${EPOCHS:-100}"
LR="${LR:-1e-5}"
NUM_IMAGES="${NUM_IMAGES:-50000}"
EVAL_BSZ="${EVAL_BSZ:-256}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EXP_NAME="${EXP_NAME:-JiT_B-vae-start-tc0p30-kw1p00-gbsz1024-full}"

if (( GLOBAL_BSZ % GRAD_ACCUM_STEPS != 0 )); then
    echo "GLOBAL_BSZ=${GLOBAL_BSZ} must be divisible by GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS}" >&2
    exit 1
fi
if (( MICRO_GLOBAL_BSZ % WORLD_SIZE != 0 )); then
    echo "MICRO_GLOBAL_BSZ=${MICRO_GLOBAL_BSZ} must be divisible by WORLD_SIZE=${WORLD_SIZE}" >&2
    exit 1
fi

echo "[$(date --iso-8601=seconds)] starting 8-GPU JiT-B VAE-start post-training"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-manual}"
echo "PWD=$(pwd)"
echo "GIT_COMMIT=$(git rev-parse HEAD)"
echo "GIT_STATUS_START"
git status --short || true
echo "GIT_STATUS_END"
echo "DATA_ROOT=${DATA_ROOT}"
echo "NNODES=${NNODES} GPUS_PER_NODE=${GPUS_PER_NODE} WORLD_SIZE=${WORLD_SIZE}"
echo "GLOBAL_BSZ=${GLOBAL_BSZ} GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS}"
echo "MICRO_GLOBAL_BSZ=${MICRO_GLOBAL_BSZ} BATCH_SIZE=${BATCH_SIZE}"
echo "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
echo "TC=${TC} KL_WEIGHT=${KL_WEIGHT} PRE_STEPS=${PRE_STEPS}"
echo "EPOCHS=${EPOCHS} STEPS_PER_EPOCH=${STEPS_PER_EPOCH} LR=${LR}"

srun --ntasks="${NNODES}" --ntasks-per-node=1 bash -lc "\
    mkdir -p '${TMPDIR}' && \
    torchrun \
    --nnodes='${NNODES}' \
    --nproc_per_node='${GPUS_PER_NODE}' \
    --rdzv_id='${SLURM_JOB_ID:-manual}' \
    --rdzv_backend=c10d \
    --rdzv_endpoint='${MASTER_ADDR}:${MASTER_PORT}' \
    main_jit_vae_start.py \
    --project jit_vae_start \
    --exp_name '${EXP_NAME}' \
    --output_dir '${RUN_ROOT}' \
    --batch_size '${BATCH_SIZE}' \
    ${DATA_ARGS_STRING} \
    --load_from checkpoints/base/JiT-B.pth \
    --model JiT_B --rope_2d --learned_pe --legacy_time_convention \
    --cfg 3.0 --interval_min 0.1 --interval_max 1.0 \
    --ema_type edm \
    --num_sampling_steps 1 \
    --print_freq 20 --milestone_interval 10 --save_freq 5 \
    --epochs '${EPOCHS}' --steps_per_epoch '${STEPS_PER_EPOCH}' --warmup_epochs 5 \
    --lr '${LR}' --lr_sched cosine --min_lr 0.0 \
    --gradient_accumulation_steps '${GRAD_ACCUM_STEPS}' \
    --vae_start_tc '${TC}' \
    --vae_start_pre_steps '${PRE_STEPS}' \
    --vae_start_kl_weight '${KL_WEIGHT}' \
    --vae_start_cycle_weight 1.0 \
    --vae_start_hidden 64 \
    --vae_start_lr 2e-4 \
    --vae_start_sample_mode posterior \
    --train_t_min 0.02 --train_t_max 1.0 \
    --num_workers '${NUM_WORKERS}' \
    --auto_resume \
    --disable_wandb"

CKPT_DIR="${RUN_ROOT}/jit_vae_start/${EXP_NAME}/checkpoints"
CKPT_PATH="$(find "${CKPT_DIR}" -maxdepth 1 -name 'step_*.pth' | sort | tail -1)"
if [[ -z "${CKPT_PATH}" ]]; then
    echo "[ERR] no checkpoint found in ${CKPT_DIR}" >&2
    exit 3
fi

echo "[$(date --iso-8601=seconds)] training finished; evaluating ${CKPT_PATH} with ${NUM_IMAGES} images"

srun --ntasks="${NNODES}" --ntasks-per-node=1 bash -lc "\
    mkdir -p '${TMPDIR}' && \
    torchrun \
    --nnodes='${NNODES}' \
    --nproc_per_node='${GPUS_PER_NODE}' \
    --rdzv_id='${SLURM_JOB_ID:-manual}-eval' \
    --rdzv_backend=c10d \
    --rdzv_endpoint='${MASTER_ADDR}:$(( MASTER_PORT + 1 ))' \
    eval_all_fds.py \
    --model JiT_B \
    --rope_2d --learned_pe --legacy_time_convention --ema_type edm \
    --cfg 3.0 --cfg_list 3.0 \
    --interval_min 0.1 --interval_max 1.0 \
    --num_sampling_steps 1 \
    --eval_ema_labels online \
    --disable_wandb --no_prc \
    --models inception \
    --eval_bsz '${EVAL_BSZ}' \
    --num_images '${NUM_IMAGES}' \
    --num_images_for_eval_and_search '${NUM_IMAGES}' \
    --load_from '${CKPT_PATH}' \
    --output_dir '${RUN_ROOT}/jit_vae_start_eval' \
    --project eval_vae_start \
    --exp_name '${EXP_NAME}-fid-${NUM_IMAGES}-${SLURM_JOB_ID:-manual}'"

echo "[$(date --iso-8601=seconds)] finished 8-GPU JiT-B VAE-start train+50k eval"
