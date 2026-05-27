#!/bin/bash
#SBATCH --job-name=jitB_vshuf
#SBATCH --account=EUHPC_D32_028
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
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
DATA_ROOT="${DATA_ROOT:-/leonardo_scratch/fast/EUHPC_D32_028/dongli/huggingface/imagenet-1k}"

export HF_HOME="${CACHE_ROOT}/huggingface"
export TORCH_HOME="${CACHE_ROOT}/torch"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export TMPDIR="/tmp/fdloss_${USER}_${SLURM_JOB_ID:-manual}"
export WANDB_MODE=disabled
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="/leonardo_scratch/fast/EUHPC_D32_028/dongli/envs/ggn-cu126-py311/lib/python3.11/site-packages:${PYTHONPATH:-}"
export FDLOSS_FORCE_SINGLE_PROCESS=1

mkdir -p "${RUN_ROOT}/slurm" "${RUN_ROOT}/jit_vae_start_shuffle_diag" \
    "${HF_HOME}" "${TORCH_HOME}" "${TMPDIR}"

run_diag() {
    local name="$1"
    local ckpt="$2"
    local out_dir="${RUN_ROOT}/jit_vae_start_shuffle_diag/${name}-${SLURM_JOB_ID:-manual}"

    if [[ ! -f "${ckpt}" ]]; then
        echo "[WARN] missing checkpoint for ${name}: ${ckpt}" >&2
        return 0
    fi
    echo "[$(date --iso-8601=seconds)] shuffle diag ${name}: ${ckpt}"
    python scripts/visualize_jit_b_vae_start_checkpoint.py \
        --ckpt "${ckpt}" \
        --data_path "${DATA_ROOT}" \
        --model JiT_B --rope_2d --learned_pe --legacy_time_convention \
        --ema_type edm \
        --cfg 3.0 --interval_min 0.1 --interval_max 1.0 \
        --vae_start_hidden 64 \
        --vae_start_sample_mode posterior \
        --num_images 16 \
        --num_workers 2 \
        --out_dir "${out_dir}" \
        --disable_wandb
}

SOURCE_JOB_ID="${SOURCE_JOB_ID:-42738799}"

run_diag \
    "kl1-post1k" \
    "${RUN_ROOT}/jit_vae_start_kl_sweep_post1k/JiT_B-vae-start-tc0p30-kw1-pre1000-post1k-${SOURCE_JOB_ID}/checkpoints/step_0000999.pth"

run_diag \
    "kl9-post1k" \
    "${RUN_ROOT}/jit_vae_start_kl_sweep_post1k/JiT_B-vae-start-tc0p30-kw9-pre1000-post1k-${SOURCE_JOB_ID}/checkpoints/step_0000999.pth"

run_diag \
    "kl1-post10k" \
    "${RUN_ROOT}/jit_vae_start/JiT_B-vae-start-tc0p30-kw1p00-gbsz64-rep1-0525/checkpoints/step_0009999.pth"

echo "[$(date --iso-8601=seconds)] finished shuffle diagnostics"
