#!/bin/bash
#SBATCH --job-name=jitB_fd5k
#SBATCH --account=EUHPC_D32_028
#SBATCH --partition=boost_usr_prod
#SBATCH --time=06:00:00
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
export TMPDIR="/tmp/fdloss_${USER}_${SLURM_JOB_ID:-manual}"
export WANDB_MODE=disabled
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="/leonardo_scratch/fast/EUHPC_D32_028/dongli/envs/ggn-cu126-py311/lib/python3.11/site-packages:${PYTHONPATH:-}"

mkdir -p "${RUN_ROOT}/slurm" "${RUN_ROOT}/jit_b_fd_inception_gbsz64_5k" \
    "${RUN_ROOT}/jit_b_fd_inception_gbsz64_5k_eval" "${HF_HOME}" "${TORCH_HOME}" "${TMPDIR}"

GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
GLOBAL_BSZ="${GLOBAL_BSZ:-64}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
if (( GLOBAL_BSZ % (GPUS_PER_NODE * GRAD_ACCUM_STEPS) != 0 )); then
    echo "[ERR] GLOBAL_BSZ=${GLOBAL_BSZ} must be divisible by GPUS_PER_NODE*GRAD_ACCUM_STEPS=$(( GPUS_PER_NODE * GRAD_ACCUM_STEPS ))" >&2
    exit 2
fi
BATCH_SIZE=$(( GLOBAL_BSZ / GPUS_PER_NODE / GRAD_ACCUM_STEPS ))
MASTER_PORT="${MASTER_PORT:-29537}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-500}"
EPOCHS="${EPOCHS:-10}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-1}"
NUM_IMAGES="${NUM_IMAGES:-50000}"
EVAL_BSZ="${EVAL_BSZ:-64}"
QUEUE_SIZE="${QUEUE_SIZE:-50000}"
FD_QUEUE_FILL_BSZ="${FD_QUEUE_FILL_BSZ:-64}"
EXP_NAME="${EXP_NAME:-JiT_B-fd-inception-gbsz${GLOBAL_BSZ}-5k-${SLURM_JOB_ID:-manual}}"

echo "[$(date --iso-8601=seconds)] starting JiT-B FD-Inception gbsz64 5k probe"
echo "GPUS_PER_NODE=${GPUS_PER_NODE} GLOBAL_BSZ=${GLOBAL_BSZ} BATCH_SIZE=${BATCH_SIZE} GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS}"
echo "EPOCHS=${EPOCHS} STEPS_PER_EPOCH=${STEPS_PER_EPOCH} WARMUP_EPOCHS=${WARMUP_EPOCHS}"
echo "QUEUE_SIZE=${QUEUE_SIZE} FD_QUEUE_FILL_BSZ=${FD_QUEUE_FILL_BSZ}"

torchrun --nproc_per_node="${GPUS_PER_NODE}" --master_port="${MASTER_PORT}" \
    main_fd.py \
    --project jit_b_fd_inception_gbsz64_5k \
    --exp_name "${EXP_NAME}" \
    --output_dir "${RUN_ROOT}" \
    --batch_size "${BATCH_SIZE}" \
    --load_from checkpoints/base/JiT-B.pth \
    --model JiT_B --rope_2d --learned_pe --legacy_time_convention \
    --cfg 3.0 --interval_min 0.1 --interval_max 1.0 \
    --ema_type edm \
    --num_sampling_steps 1 \
    --print_freq 20 --milestone_interval 0 --save_freq 10 --disable_vis \
    --epochs "${EPOCHS}" --steps_per_epoch "${STEPS_PER_EPOCH}" --warmup_epochs "${WARMUP_EPOCHS}" \
    --lr 1e-5 --lr_sched cosine --min_lr 0.0 \
    --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
    --queue_size "${QUEUE_SIZE}" --fd_queue_fill_bsz "${FD_QUEUE_FILL_BSZ}" \
    --fd_repr_models inception --fd_eigvalsh --fd_ema_beta 0.999 \
    --num_images_for_eval_and_search "${NUM_IMAGES}" --eval_bsz "${EVAL_BSZ}" \
    --compile --disable_wandb

CKPT_DIR="${RUN_ROOT}/jit_b_fd_inception_gbsz64_5k/${EXP_NAME}/checkpoints"
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
    --output_dir "${RUN_ROOT}/jit_b_fd_inception_gbsz64_5k_eval" \
    --project eval_fd_inception_gbsz64_5k \
    --exp_name "${EXP_NAME}-fid-${NUM_IMAGES}-${SLURM_JOB_ID:-manual}"

echo "[$(date --iso-8601=seconds)] finished JiT-B FD-Inception gbsz64 5k probe"
