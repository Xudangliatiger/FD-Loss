#!/bin/bash
#SBATCH --job-name=jitB_vaestart
#SBATCH --account=EUHPC_D32_028
#SBATCH --partition=boost_usr_prod
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
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
export TMPDIR="${SCRATCH_ROOT}/tmp"
export WANDB_MODE=disabled
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONDONTWRITEBYTECODE=1

mkdir -p "${RUN_ROOT}/slurm" "${RUN_ROOT}/jit_vae_start" "${RUN_ROOT}/jit_vae_start_eval" \
    "${HF_HOME}" "${TORCH_HOME}" "${TMPDIR}"

DATA_ROOT="${DATA_ROOT:-/leonardo_scratch/fast/EUHPC_D32_028/dongli/huggingface/imagenet-1k}"

DATA_ARGS=()
if [[ -d "${DATA_ROOT}/train" || -d "${DATA_ROOT}/data" ]]; then
    DATA_ARGS=(--data_path "${DATA_ROOT}")
elif [[ -d "${DATA_ROOT}/n01440764" ]]; then
    DATA_ARGS=(--data_path "__unused_imagefolder_root__" --image_list data/train.txt --image_root "${DATA_ROOT}")
else
    echo "[ERR] Could not find ImageNet train images under DATA_ROOT=${DATA_ROOT}" >&2
    echo "      Expected ImageFolder train/, HF parquet data/train-*.parquet, or class folders." >&2
    exit 2
fi

GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
GLOBAL_BSZ="${GLOBAL_BSZ:-64}"
BATCH_SIZE=$(( GLOBAL_BSZ / GPUS_PER_NODE ))
MASTER_PORT="${MASTER_PORT:-29529}"
TC="${TC:-0.30}"
KL_WEIGHT="${KL_WEIGHT:-0.25}"
PRE_STEPS="${PRE_STEPS:-2000}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-500}"
EPOCHS="${EPOCHS:-20}"
NUM_IMAGES="${NUM_IMAGES:-50000}"
EVAL_BSZ="${EVAL_BSZ:-64}"
EXP_NAME="${EXP_NAME:-JiT_B-vae-start-tc0p30-kw0p25-${SLURM_JOB_ID:-manual}}"

echo "[$(date --iso-8601=seconds)] starting JiT-B VAE-start post-training"
echo "DATA_ROOT=${DATA_ROOT}"
echo "GPUS_PER_NODE=${GPUS_PER_NODE} GLOBAL_BSZ=${GLOBAL_BSZ} BATCH_SIZE=${BATCH_SIZE}"
echo "TC=${TC} KL_WEIGHT=${KL_WEIGHT} PRE_STEPS=${PRE_STEPS} EPOCHS=${EPOCHS} STEPS_PER_EPOCH=${STEPS_PER_EPOCH}"

torchrun --nproc_per_node="${GPUS_PER_NODE}" --master_port="${MASTER_PORT}" \
    main_jit_vae_start.py \
    --project jit_vae_start \
    --exp_name "${EXP_NAME}" \
    --output_dir "${RUN_ROOT}" \
    --batch_size "${BATCH_SIZE}" \
    "${DATA_ARGS[@]}" \
    --load_from checkpoints/base/JiT-B.pth \
    --model JiT_B --rope_2d --learned_pe --legacy_time_convention \
    --cfg 3.0 --interval_min 0.1 --interval_max 1.0 \
    --ema_type edm \
    --num_sampling_steps 1 \
    --print_freq 20 --milestone_interval 10 --save_freq 5 \
    --epochs "${EPOCHS}" --steps_per_epoch "${STEPS_PER_EPOCH}" --warmup_epochs 1 \
    --lr 1e-5 --lr_sched cosine --min_lr 0.0 \
    --vae_start_tc "${TC}" \
    --vae_start_pre_steps "${PRE_STEPS}" \
    --vae_start_kl_weight "${KL_WEIGHT}" \
    --vae_start_cycle_weight 1.0 \
    --vae_start_hidden 64 \
    --vae_start_lr 2e-4 \
    --vae_start_sample_mode posterior \
    --train_t_min 0.02 --train_t_max 1.0 \
    --num_workers 12 \
    --disable_wandb

CKPT_DIR="${RUN_ROOT}/jit_vae_start/${EXP_NAME}/checkpoints"
CKPT_PATH="$(find "${CKPT_DIR}" -maxdepth 1 -name 'step_*.pth' | sort | tail -1)"
if [[ -z "${CKPT_PATH}" ]]; then
    echo "[ERR] no checkpoint found in ${CKPT_DIR}" >&2
    exit 3
fi

echo "[$(date --iso-8601=seconds)] training finished; evaluating ${CKPT_PATH} with ${NUM_IMAGES} images"

torchrun --nproc_per_node="${GPUS_PER_NODE}" --master_port="$(( MASTER_PORT + 1 ))" \
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
    --output_dir "${RUN_ROOT}/jit_vae_start_eval" \
    --project eval_vae_start \
    --exp_name "${EXP_NAME}-fid-${NUM_IMAGES}-${SLURM_JOB_ID:-manual}"

echo "[$(date --iso-8601=seconds)] finished JiT-B VAE-start train+50k eval"
