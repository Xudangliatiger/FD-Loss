#!/bin/bash
#SBATCH --job-name=pmf_patch_norm
#SBATCH --account=EUHPC_D32_028
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --time=12:00:00
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

mkdir -p "${RUN_ROOT}/slurm" "${RUN_ROOT}/self_fd_patch_train" \
    "${HF_HOME}" "${TORCH_HOME}" "${TMPDIR}"

GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
NNODES="${SLURM_NNODES:-2}"
WORLD_SIZE=$(( NNODES * GPUS_PER_NODE ))
MASTER_PORT="${MASTER_PORT:-29597}"
GLOBAL_BSZ="${GLOBAL_BSZ:-128}"
BATCH_SIZE=$(( GLOBAL_BSZ / WORLD_SIZE ))
EPOCHS="${EPOCHS:-5}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-1250}"
QUEUE_SIZE="${QUEUE_SIZE:-262144}"
FD_QUEUE_FILL_BSZ="${FD_QUEUE_FILL_BSZ:-64}"
LR="${LR:-1e-6}"
STATS_PATH="${STATS_PATH:-data/fid_stats/self_pmf_b_shared7_t005_in256_patch_stats.npz}"
EXP_NAME="${EXP_NAME:-pMF_B-patch-self-fd-normal-${SLURM_JOB_ID:-manual}}"
MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"

echo "[$(date --iso-8601=seconds)] starting pMF-B patch self-FD normal training"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-manual}"
echo "PWD=$(pwd)"
echo "GIT_COMMIT=$(git rev-parse HEAD)"
echo "GIT_STATUS_START"
git status --short
echo "GIT_STATUS_END"
echo "NNODES=${NNODES} GPUS_PER_NODE=${GPUS_PER_NODE} WORLD_SIZE=${WORLD_SIZE}"
echo "GLOBAL_BSZ=${GLOBAL_BSZ} BATCH_SIZE=${BATCH_SIZE}"
echo "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
echo "EPOCHS=${EPOCHS} STEPS_PER_EPOCH=${STEPS_PER_EPOCH}"
echo "QUEUE_SIZE=${QUEUE_SIZE} FD_QUEUE_FILL_BSZ=${FD_QUEUE_FILL_BSZ}"
echo "LR=${LR}"
echo "STATS_PATH=${STATS_PATH}"
echo "TMPDIR=${TMPDIR}"

test -f "${STATS_PATH}"

srun --ntasks="${NNODES}" --ntasks-per-node=1 bash -lc "\
    mkdir -p '${TMPDIR}' && \
    torchrun \
    --nnodes='${NNODES}' \
    --nproc_per_node='${GPUS_PER_NODE}' \
    --rdzv_id='${SLURM_JOB_ID:-manual}' \
    --rdzv_backend=c10d \
    --rdzv_endpoint='${MASTER_ADDR}:${MASTER_PORT}' \
    main_fd.py \
    --project self_fd_patch_train \
    --exp_name '${EXP_NAME}' \
    --output_dir '${RUN_ROOT}/self_fd_patch_train' \
    --batch_size '${BATCH_SIZE}' \
    --load_from checkpoints/base/pMF-B_256.pth \
    --model pMF_B --rope_2d --learned_pe --disable_v_head \
    --cfg 8.5 --interval_min 0.1 --interval_max 0.7 \
    --num_sampling_steps 1 \
    --print_freq 20 --vis_freq 0 --eval_freq 0 --val_freq 0 \
    --save_freq 1 --milestone_interval 0 --keep_n_ckpts 5 \
    --epochs '${EPOCHS}' --steps_per_epoch '${STEPS_PER_EPOCH}' --warmup_epochs 0 \
    --lr '${LR}' --lr_sched cosine --min_lr 0.0 \
    --fd_repr_models self_pmf_b_patch \
    --fd_repr_stats_paths '${STATS_PATH}' \
    --fd_self_shared_block 7 --fd_self_t 0.05 \
    --fd_eigvalsh --fd_online_accum \
    --queue_size '${QUEUE_SIZE}' --fd_queue_fill_bsz '${FD_QUEUE_FILL_BSZ}' \
    --auto_resume --disable_wandb --disable_vis"

echo "[$(date --iso-8601=seconds)] finished pMF-B patch self-FD normal training"
