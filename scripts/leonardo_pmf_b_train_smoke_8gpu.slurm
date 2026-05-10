#!/bin/bash
#SBATCH --job-name=fdloss_train_8g
#SBATCH --account=EUHPC_D32_028
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:30:00
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
export TMPDIR="${SCRATCH_ROOT}/tmp"
export WANDB_MODE=disabled
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONDONTWRITEBYTECODE=1

mkdir -p "${RUN_ROOT}/slurm" "${RUN_ROOT}/train_smoke" \
    "${HF_HOME}" "${TORCH_HOME}" "${TMPDIR}"

GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
NNODES="${SLURM_NNODES:-2}"
WORLD_SIZE=$(( NNODES * GPUS_PER_NODE ))
GLOBAL_BSZ="${GLOBAL_BSZ:-64}"
BATCH_SIZE=$(( GLOBAL_BSZ / WORLD_SIZE ))
MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
MASTER_PORT="${MASTER_PORT:-29543}"

echo "[$(date --iso-8601=seconds)] starting pMF-B FD-Loss 8GPU train smoke"
echo "NNODES=${NNODES} GPUS_PER_NODE=${GPUS_PER_NODE} WORLD_SIZE=${WORLD_SIZE}"
echo "GLOBAL_BSZ=${GLOBAL_BSZ} BATCH_SIZE=${BATCH_SIZE}"
echo "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"

srun --ntasks="${NNODES}" --ntasks-per-node=1 bash -lc "\
    torchrun \
      --nnodes='${NNODES}' \
      --nproc_per_node='${GPUS_PER_NODE}' \
      --rdzv_id='${SLURM_JOB_ID:-manual}' \
      --rdzv_backend=c10d \
      --rdzv_endpoint='${MASTER_ADDR}:${MASTER_PORT}' \
      main_fd.py \
      --project train_smoke \
      --exp_name 'pMF_B-fd-inception-smoke-8gpu-${SLURM_JOB_ID:-manual}' \
      --output_dir '${RUN_ROOT}/train_smoke' \
      --batch_size '${BATCH_SIZE}' \
      --load_from checkpoints/base/pMF-B_256.pth \
      --model pMF_B --rope_2d --learned_pe --disable_v_head \
      --cfg 8.5 --interval_min 0.1 --interval_max 0.7 \
      --num_sampling_steps 1 \
      --print_freq 1 --vis_freq 0 --eval_freq 0 --val_freq 0 \
      --save_freq 100 --milestone_interval 0 \
      --epochs 1 --steps_per_epoch 10 --warmup_epochs 0 \
      --lr 1e-6 --lr_sched cosine --min_lr 0.0 \
      --fd_repr_models inception --fd_eigvalsh \
      --queue_size 1024 --fd_queue_fill_bsz 128 \
      --disable_wandb --disable_vis"

echo "[$(date --iso-8601=seconds)] finished pMF-B FD-Loss 8GPU train smoke"
