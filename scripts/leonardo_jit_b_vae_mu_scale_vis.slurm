#!/bin/bash
#SBATCH --job-name=jitB_vaemu
#SBATCH --account=EUHPC_D32_028
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --output=/leonardo_scratch/large/userexternal/dxu00000/2027-ICLR-PixelGen/FD-Loss/runs/slurm/%x-%j.out
#SBATCH --error=/leonardo_scratch/large/userexternal/dxu00000/2027-ICLR-PixelGen/FD-Loss/runs/slurm/%x-%j.err

set -euo pipefail

module load python/3.11.7
module load cuda/12.6

source /leonardo_scratch/fast/EUHPC_D32_028/dongli/envs/fdloss-cu126-py311/bin/activate

cd /leonardo_scratch/fast/EUHPC_D32_028/dongli/FD-Loss

SCRATCH_ROOT=/leonardo_scratch/large/userexternal/dxu00000/2027-ICLR-PixelGen/FD-Loss
CACHE_ROOT="${SCRATCH_ROOT}/cache"
RUN_ROOT="${SCRATCH_ROOT}/runs"

export HF_HOME="${CACHE_ROOT}/huggingface"
export TORCH_HOME="${CACHE_ROOT}/torch"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export TMPDIR="/tmp/fdloss_${USER}_${SLURM_JOB_ID:-manual}"
export WANDB_MODE=disabled
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="/leonardo_scratch/fast/EUHPC_D32_028/dongli/envs/ggn-cu126-py311/lib/python3.11/site-packages:${PYTHONPATH:-}"

mkdir -p "${RUN_ROOT}/slurm" "${HF_HOME}" "${TORCH_HOME}" "${TMPDIR}"

DATA_ROOT="${DATA_ROOT:-/leonardo_scratch/fast/EUHPC_D32_028/dongli/huggingface/imagenet-1k}"
CKPT="${CKPT:-${RUN_ROOT}/jit_vae_start_pretrain_vis/JiT_B-vae-only-kw0p25-steps5000-42583626/checkpoints/vae_encoder_step_0005000.pth}"
OUT_DIR="${OUT_DIR:-${RUN_ROOT}/jit_vae_mu_scale_sweep/JiT_B-vae-only-kw0p25-step5000}"
SCALES="${SCALES:-0.00 0.10 0.25 0.50 0.75 1.00}"
NUM_IMAGES="${NUM_IMAGES:-16}"
NUM_WORKERS="${NUM_WORKERS:-2}"

echo "[$(date --iso-8601=seconds)] VAE mean-scale visualization"
echo "CKPT=${CKPT}"
echo "OUT_DIR=${OUT_DIR}"
echo "SCALES=${SCALES}"

for scale in ${SCALES}; do
    tag="${scale/./p}"
    out="${OUT_DIR}/scale_${tag}"
    echo "[$(date --iso-8601=seconds)] scale=${scale} -> ${out}"
    python scripts/visualize_jit_b_vae_start_checkpoint.py \
        --ckpt "${CKPT}" \
        --out_dir "${out}" \
        --data_path "${DATA_ROOT}" \
        --model JiT_B --rope_2d --learned_pe --legacy_time_convention \
        --ema_type edm \
        --vae_start_hidden 64 \
        --vae_start_sample_mode mean_shift \
        --vae_start_mean_scale "${scale}" \
        --num_images "${NUM_IMAGES}" \
        --num_workers "${NUM_WORKERS}" \
        --disable_wandb
done

echo "[$(date --iso-8601=seconds)] finished VAE mean-scale visualization"
