#!/bin/bash
#SBATCH --job-name=jit_patch_inc
#SBATCH --account=EUHPC_D32_028
#SBATCH --partition=boost_usr_prod
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
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
export TMPDIR="/tmp/fdloss_${USER}_${SLURM_JOB_ID:-manual}"
export WANDB_MODE=disabled
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONDONTWRITEBYTECODE=1

mkdir -p "${RUN_ROOT}/slurm" "${RUN_ROOT}/self_fd_jit_patch_inception" \
    "${HF_HOME}" "${TORCH_HOME}" "${TMPDIR}"

GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
NNODES="${SLURM_NNODES:-1}"
WORLD_SIZE=$(( NNODES * GPUS_PER_NODE ))
MASTER_PORT="${MASTER_PORT:-29645}"
GLOBAL_BSZ="${GLOBAL_BSZ:-1024}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
MICRO_GLOBAL_BSZ=$(( GLOBAL_BSZ / GRAD_ACCUM_STEPS ))
BATCH_SIZE=$(( MICRO_GLOBAL_BSZ / WORLD_SIZE ))
EPOCHS="${EPOCHS:-1}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-1000}"
QUEUE_SIZE="${QUEUE_SIZE:-262144}"
FD_QUEUE_FILL_BSZ="${FD_QUEUE_FILL_BSZ:-64}"
LR="${LR:-1e-5}"
VIS_EVERY_STEPS="${VIS_EVERY_STEPS:-500}"
PATCH_WEIGHT="${PATCH_WEIGHT:-1.0}"
INCEPTION_WEIGHT="${INCEPTION_WEIGHT:-1.0}"
PATCH_STATS="${PATCH_STATS:-data/fid_stats/self_jit_b_block11_t005_in256_patch_stats.npz}"
INCEPTION_STATS="${INCEPTION_STATS:-data/fid_stats/guided_diffusion_stats.npz}"
EXP_NAME="${EXP_NAME:-JiT_B-patch-self-fd-inception-w1-${SLURM_JOB_ID:-manual}}"
MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"

if (( GLOBAL_BSZ % GRAD_ACCUM_STEPS != 0 )); then
    echo "GLOBAL_BSZ=${GLOBAL_BSZ} must be divisible by GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS}" >&2
    exit 1
fi
if (( MICRO_GLOBAL_BSZ % WORLD_SIZE != 0 )); then
    echo "MICRO_GLOBAL_BSZ=${MICRO_GLOBAL_BSZ} must be divisible by WORLD_SIZE=${WORLD_SIZE}" >&2
    exit 1
fi

echo "[$(date --iso-8601=seconds)] starting JiT-B patch self-FD + Inception FD smoke"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-manual}"
echo "PWD=$(pwd)"
echo "GIT_COMMIT=$(git rev-parse HEAD)"
echo "GIT_STATUS_START"
git status --short || true
echo "GIT_STATUS_END"
echo "NNODES=${NNODES} GPUS_PER_NODE=${GPUS_PER_NODE} WORLD_SIZE=${WORLD_SIZE}"
echo "GLOBAL_BSZ=${GLOBAL_BSZ} GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS}"
echo "MICRO_GLOBAL_BSZ=${MICRO_GLOBAL_BSZ} BATCH_SIZE=${BATCH_SIZE}"
echo "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
echo "EPOCHS=${EPOCHS} STEPS_PER_EPOCH=${STEPS_PER_EPOCH}"
echo "QUEUE_SIZE=${QUEUE_SIZE} FD_QUEUE_FILL_BSZ=${FD_QUEUE_FILL_BSZ}"
echo "LR=${LR}"
echo "VIS_EVERY_STEPS=${VIS_EVERY_STEPS}"
echo "PATCH_WEIGHT=${PATCH_WEIGHT} INCEPTION_WEIGHT=${INCEPTION_WEIGHT}"
echo "PATCH_STATS=${PATCH_STATS}"
echo "INCEPTION_STATS=${INCEPTION_STATS}"

test -f "${PATCH_STATS}"
test -f "${INCEPTION_STATS}"
test -f "${TORCH_HOME}/hub/checkpoints/weights-inception-2015-12-05-6726825d.pth"

srun --ntasks="${NNODES}" --ntasks-per-node=1 bash -lc "\
    mkdir -p '${TMPDIR}' && \
    torchrun \
    --nnodes='${NNODES}' \
    --nproc_per_node='${GPUS_PER_NODE}' \
    --rdzv_id='${SLURM_JOB_ID:-manual}' \
    --rdzv_backend=c10d \
    --rdzv_endpoint='${MASTER_ADDR}:${MASTER_PORT}' \
    main_fd.py \
    --project self_fd_jit_patch_inception \
    --exp_name '${EXP_NAME}' \
    --output_dir '${RUN_ROOT}/self_fd_jit_patch_inception' \
    --batch_size '${BATCH_SIZE}' \
    --load_from checkpoints/base/JiT-B.pth \
    --model JiT_B --rope_2d --learned_pe --legacy_time_convention \
    --cfg 3.0 --interval_min 0.1 --interval_max 1.0 \
    --num_sampling_steps 1 \
    --print_freq 20 --vis_freq 0 --vis_every_steps '${VIS_EVERY_STEPS}' --eval_freq 0 --val_freq 0 \
    --save_freq 1 --milestone_interval 0 --keep_n_ckpts 3 \
    --epochs '${EPOCHS}' --steps_per_epoch '${STEPS_PER_EPOCH}' --warmup_epochs 0 \
    --lr '${LR}' --lr_sched cosine --min_lr 0.0 \
    --gradient_accumulation_steps '${GRAD_ACCUM_STEPS}' \
    --fd_repr_models self_jit_b_patch inception \
    --fd_repr_stats_paths '${PATCH_STATS}' '${INCEPTION_STATS}' \
    --fd_repr_pool_types cls cls \
    --fd_repr_weights '${PATCH_WEIGHT}' '${INCEPTION_WEIGHT}' \
    --fd_self_jit_block 11 --fd_self_t 0.05 \
    --fd_target_sizes 256 256 \
    --fd_eigvalsh --fd_online_accum \
    --queue_size '${QUEUE_SIZE}' --fd_queue_fill_bsz '${FD_QUEUE_FILL_BSZ}' \
    --auto_resume --disable_wandb"

echo "[$(date --iso-8601=seconds)] finished JiT-B patch self-FD + Inception FD smoke"
