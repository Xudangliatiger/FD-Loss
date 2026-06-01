#!/bin/bash
#SBATCH --job-name=jitB_dinorec
#SBATCH --account=EUHPC_D32_028
#SBATCH --partition=boost_usr_prod
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --exclude=lrdn2272,lrdn3214,lrdn2118
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

mkdir -p "${RUN_ROOT}/slurm" "${TMPDIR}" "${HF_HOME}" "${TORCH_HOME}"

DATA_ROOT="${DATA_ROOT:-/leonardo_scratch/fast/EUHPC_D32_028/dongli/huggingface/imagenet-1k}"
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/jit_dinov3_sphere_joint/JiT_B-dinov3-patchsphere-vitbridge-v0p1p7-project128-sigreg1-l1l2lpips1-gbsz1024-post125k-0531}"
CKPT="${CKPT:-${RUN_DIR}/checkpoints/latest.pth}"
OUT_DIR="${OUT_DIR:-${RUN_DIR}/visualization/recursive_inference/latest-clean-rounds3-${SLURM_JOB_ID:-manual}}"
NUM_IMAGES="${NUM_IMAGES:-16}"
NUM_ROUNDS="${NUM_ROUNDS:-3}"
REENCODE_MODE="${REENCODE_MODE:-clean}"
DINO_PRETRAINED_PATH="${DINO_PRETRAINED_PATH:-/leonardo_scratch/fast/EUHPC_D32_028/dongli/huggingface/dinov3-vitb16-pretrain-lvd1689m/model.safetensors}"

if [[ ! -f "${CKPT}" ]]; then
    echo "[ERR] checkpoint not found: ${CKPT}" >&2
    exit 2
fi

echo "[$(date --iso-8601=seconds)] starting DINO recursive inference"
echo "GIT_COMMIT=$(git rev-parse HEAD)"
echo "CKPT=${CKPT}"
echo "OUT_DIR=${OUT_DIR}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "NUM_IMAGES=${NUM_IMAGES} NUM_ROUNDS=${NUM_ROUNDS} REENCODE_MODE=${REENCODE_MODE}"

python -u scripts/visualize_dino_recursive_inference.py \
  --ckpt "${CKPT}" \
  --out_dir "${OUT_DIR}" \
  --data_path "${DATA_ROOT}" \
  --model JiT_B --rope_2d --learned_pe --legacy_time_convention \
  --ema_type edm \
  --cfg 3.0 --interval_min 0.1 --interval_max 1.0 \
  --start_support_mode pixel_sphere \
  --num_sampling_steps 1 \
  --sampling_method euler \
  --num_images "${NUM_IMAGES}" \
  --num_rounds "${NUM_ROUNDS}" \
  --reencode_mode "${REENCODE_MODE}" \
  --dinov2_start_pretrained_path "${DINO_PRETRAINED_PATH}"
