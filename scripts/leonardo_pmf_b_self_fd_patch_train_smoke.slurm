#!/bin/bash
#SBATCH --job-name=pmf_patch_train
#SBATCH --account=EUHPC_D32_028
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:30:00
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
export TMPDIR="/tmp/fdloss_${USER}_${SLURM_JOB_ID:-manual}"
export WANDB_MODE=disabled
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTHONDONTWRITEBYTECODE=1

mkdir -p "${RUN_ROOT}/slurm" "${RUN_ROOT}/self_fd_patch_train_smoke" \
    "${HF_HOME}" "${TORCH_HOME}" "${TMPDIR}"

GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29595}"
GLOBAL_BSZ="${GLOBAL_BSZ:-16}"
BATCH_SIZE=$(( GLOBAL_BSZ / GPUS_PER_NODE ))
STATS_PATH="data/fid_stats/self_pmf_b_shared7_t005_in256_patch_stats.npz"

echo "[$(date --iso-8601=seconds)] starting pMF-B patch self-FD train smoke"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-manual}"
echo "PWD=$(pwd)"
echo "GIT_COMMIT=$(git rev-parse HEAD)"
echo "GIT_STATUS_START"
git status --short
echo "GIT_STATUS_END"
echo "GPUS_PER_NODE=${GPUS_PER_NODE} GLOBAL_BSZ=${GLOBAL_BSZ} BATCH_SIZE=${BATCH_SIZE}"
echo "STATS_PATH=${STATS_PATH}"
echo "TMPDIR=${TMPDIR}"

test -f "${STATS_PATH}"

torchrun --nproc_per_node="${GPUS_PER_NODE}" --master_port="${MASTER_PORT}" \
    main_fd.py \
    --project self_fd_patch_train_smoke \
    --exp_name "pMF_B-patch-self-fd-smoke-${SLURM_JOB_ID:-manual}" \
    --output_dir "${RUN_ROOT}/self_fd_patch_train_smoke" \
    --batch_size "${BATCH_SIZE}" \
    --load_from checkpoints/base/pMF-B_256.pth \
    --model pMF_B --rope_2d --learned_pe --disable_v_head \
    --cfg 8.5 --interval_min 0.1 --interval_max 0.7 \
    --num_sampling_steps 1 \
    --print_freq 1 --vis_freq 0 --eval_freq 0 --val_freq 0 \
    --save_freq 100 --milestone_interval 0 \
    --epochs 1 --steps_per_epoch 10 --warmup_epochs 0 \
    --lr 1e-6 --lr_sched cosine --min_lr 0.0 \
    --fd_repr_models self_pmf_b_patch \
    --fd_repr_stats_paths "${STATS_PATH}" \
    --fd_self_shared_block 7 --fd_self_t 0.05 \
    --fd_eigvalsh --fd_online_accum \
    --queue_size 65536 --fd_queue_fill_bsz 64 \
    --disable_wandb --disable_vis

echo "[$(date --iso-8601=seconds)] finished pMF-B patch self-FD train smoke"
