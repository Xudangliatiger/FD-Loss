#!/bin/bash
#SBATCH --job-name=jitB_vaeklm
#SBATCH --account=EUHPC_D32_028
#SBATCH --partition=boost_usr_prod
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
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

mkdir -p "${RUN_ROOT}/slurm" "${RUN_ROOT}/jit_vae_start_kl_sweep_post1k_eval_steps" \
    "${HF_HOME}" "${TORCH_HOME}" "${TMPDIR}"

GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-29831}"
NUM_IMAGES="${NUM_IMAGES:-50000}"
EVAL_BSZ="${EVAL_BSZ:-64}"
STEPS_LIST="${STEPS_LIST:-2 4}"
KL_LIST="${KL_LIST:-1 3 5 7 9}"
SOURCE_JOB_ID="${SOURCE_JOB_ID:-42738799}"
OUT_DIR="${RUN_ROOT}/jit_vae_start_kl_sweep_post1k_eval_steps"
PROJECT="eval_vae_start_kl_sweep_post1k_steps"

run_eval() {
    local ckpt_path="$1"
    local exp_name="$2"
    local steps="$3"
    local final_csv="${OUT_DIR}/${PROJECT}/${exp_name}-steps${steps}-fid-${NUM_IMAGES}-${SLURM_JOB_ID:-manual}/final_eval_summary.csv"

    if [[ -f "${final_csv}" ]]; then
        echo "[$(date --iso-8601=seconds)] skip existing ${final_csv}"
        return 0
    fi
    if [[ ! -f "${ckpt_path}" ]]; then
        echo "[ERR] checkpoint not found: ${ckpt_path}" >&2
        exit 2
    fi

    echo "[$(date --iso-8601=seconds)] eval exp=${exp_name} steps=${steps} ckpt=${ckpt_path}"
    torchrun --nproc_per_node="${GPUS_PER_NODE}" --master_port="${MASTER_PORT}" \
        eval_all_fds.py \
        --model JiT_B \
        --rope_2d --learned_pe --legacy_time_convention --ema_type edm \
        --cfg 3.0 --cfg_list 3.0 \
        --interval_min 0.1 --interval_max 1.0 \
        --num_sampling_steps "${steps}" \
        --eval_ema_labels online \
        --disable_wandb --no_prc \
        --models inception \
        --eval_bsz "${EVAL_BSZ}" \
        --num_images "${NUM_IMAGES}" \
        --num_images_for_eval_and_search "${NUM_IMAGES}" \
        --load_from "${ckpt_path}" \
        --output_dir "${OUT_DIR}" \
        --project "${PROJECT}" \
        --exp_name "${exp_name}-steps${steps}-fid-${NUM_IMAGES}-${SLURM_JOB_ID:-manual}"

    MASTER_PORT=$(( MASTER_PORT + 1 ))
}

echo "[$(date --iso-8601=seconds)] starting JiT-B VAE-start KL multistep eval"
echo "SOURCE_JOB_ID=${SOURCE_JOB_ID} KL_LIST=${KL_LIST} STEPS_LIST=${STEPS_LIST}"
echo "NUM_IMAGES=${NUM_IMAGES} GPUS_PER_NODE=${GPUS_PER_NODE} EVAL_BSZ=${EVAL_BSZ}"

for STEPS in ${STEPS_LIST}; do
    run_eval "checkpoints/base/JiT-B.pth" "JiT_B-base" "${STEPS}"
done

for KL_WEIGHT in ${KL_LIST}; do
    KL_TAG="${KL_WEIGHT/./p}"
    TRAIN_EXP="JiT_B-vae-start-tc0p30-kw${KL_TAG}-pre1000-post1k-${SOURCE_JOB_ID}"
    CKPT_DIR="${RUN_ROOT}/jit_vae_start_kl_sweep_post1k/${TRAIN_EXP}/checkpoints"
    CKPT_PATH="$(find "${CKPT_DIR}" -maxdepth 1 -name 'step_*.pth' | sort | tail -1)"
    if [[ -z "${CKPT_PATH}" ]]; then
        echo "[ERR] no checkpoint found in ${CKPT_DIR}" >&2
        exit 3
    fi
    for STEPS in ${STEPS_LIST}; do
        run_eval "${CKPT_PATH}" "${TRAIN_EXP}" "${STEPS}"
    done
done

echo "[$(date --iso-8601=seconds)] finished JiT-B VAE-start KL multistep eval"
