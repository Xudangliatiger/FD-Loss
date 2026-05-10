#!/bin/bash
#SBATCH --job-name=fdloss_jitB_t3
#SBATCH --account=EUHPC_D32_028
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --time=2-00:00:00
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

mkdir -p "${RUN_ROOT}/slurm" "${RUN_ROOT}/table_3_JiT" \
    "${HF_HOME}" "${TORCH_HOME}" "${TMPDIR}"

GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
NNODES="${SLURM_NNODES:-2}"
WORLD_SIZE=$(( NNODES * GPUS_PER_NODE ))
GLOBAL_BSZ="${GLOBAL_BSZ:-1024}"
BATCH_SIZE=$(( GLOBAL_BSZ / WORLD_SIZE ))
MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
MASTER_PORT="${MASTER_PORT:-29545}"

echo "[$(date --iso-8601=seconds)] starting JiT-B Table 3 FD-Inception training"
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
      --project table_3_JiT \
      --exp_name 'JiT_B-fd-inception-8gpu-${SLURM_JOB_ID:-manual}' \
      --output_dir '${RUN_ROOT}/table_3_JiT' \
      --batch_size '${BATCH_SIZE}' \
      --data_path data \
      --load_from checkpoints/base/JiT-B.pth \
      --model JiT_B --rope_2d --learned_pe --legacy_time_convention \
      --cfg 3.0 --interval_min 0.1 --interval_max 1.0 \
      --ema_type edm \
      --num_sampling_steps 1 \
      --eval_bsz 256 --num_images_for_eval_and_search 50000 \
      --vis_freq 5 --online_eval --eval_freq 10000 \
      --print_freq 20 --milestone_interval 10 --save_freq 5 \
      --epochs 100 --steps_per_epoch 1250 --warmup_epochs 5 \
      --lr 1e-5 --lr_sched cosine --min_lr 0.0 \
      --fd_repr_models inception --fd_eigvalsh --fd_ema_beta 0.999 \
      --compile --auto_resume --disable_wandb"

echo "[$(date --iso-8601=seconds)] finished JiT-B Table 3 FD-Inception training"
