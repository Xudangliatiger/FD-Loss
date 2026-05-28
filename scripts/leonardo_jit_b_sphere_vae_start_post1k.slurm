#!/bin/bash
#SBATCH --job-name=jitB_svae1k
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

mkdir -p "${RUN_ROOT}/slurm" "${RUN_ROOT}/jit_sphere_vae_start_post1k" \
    "${RUN_ROOT}/jit_sphere_vae_start_post1k_eval_steps" \
    "${HF_HOME}" "${TORCH_HOME}" "${TMPDIR}"

DATA_ROOT="${DATA_ROOT:-/leonardo_scratch/fast/EUHPC_D32_028/dongli/huggingface/imagenet-1k}"
DATA_ARGS=()
if [[ -d "${DATA_ROOT}/train" || -d "${DATA_ROOT}/data" ]]; then
    DATA_ARGS=(--data_path "${DATA_ROOT}")
elif [[ -d "${DATA_ROOT}/n01440764" ]]; then
    DATA_ARGS=(--data_path "__unused_imagefolder_root__" --image_list data/train.txt --image_root "${DATA_ROOT}")
else
    echo "[ERR] Could not find ImageNet train images under DATA_ROOT=${DATA_ROOT}" >&2
    exit 2
fi

GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
GLOBAL_BSZ="${GLOBAL_BSZ:-64}"
BATCH_SIZE=$(( GLOBAL_BSZ / GPUS_PER_NODE ))
MASTER_PORT="${MASTER_PORT:-29931}"
TC="${TC:-0.30}"
PRE_STEPS="${PRE_STEPS:-1000}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-500}"
EPOCHS="${EPOCHS:-2}"
NUM_IMAGES="${NUM_IMAGES:-5000}"
EVAL_BSZ="${EVAL_BSZ:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
KL_LIST="${KL_LIST:-1 5 9}"
STEPS_LIST="${STEPS_LIST:-1 2 4}"
START_SUPPORT_MODE="${START_SUPPORT_MODE:-pixel_sphere}"

TRAIN_PROJECT="jit_sphere_vae_start_post1k"
EVAL_PROJECT="eval_sphere_vae_start_post1k_steps"
EVAL_ROOT="${RUN_ROOT}/jit_sphere_vae_start_post1k_eval_steps"

run_eval() {
    local ckpt_path="$1"
    local exp_name="$2"
    local steps="$3"
    local final_csv="${EVAL_ROOT}/${EVAL_PROJECT}/${exp_name}-steps${steps}-fid-${NUM_IMAGES}-${SLURM_JOB_ID:-manual}/final_eval_summary.csv"

    if [[ -f "${final_csv}" ]]; then
        echo "[$(date --iso-8601=seconds)] skip existing ${final_csv}"
        return 0
    fi

    echo "[$(date --iso-8601=seconds)] eval exp=${exp_name} steps=${steps} support=${START_SUPPORT_MODE}"
    torchrun --nproc_per_node="${GPUS_PER_NODE}" --master_port="${MASTER_PORT}" \
        eval_all_fds.py \
        --model JiT_B \
        --rope_2d --learned_pe --legacy_time_convention --ema_type edm \
        --cfg 3.0 --cfg_list 3.0 \
        --interval_min 0.1 --interval_max 1.0 \
        --num_sampling_steps "${steps}" \
        --start_support_mode "${START_SUPPORT_MODE}" \
        --eval_ema_labels online \
        --disable_wandb --no_prc \
        --models inception \
        --eval_bsz "${EVAL_BSZ}" \
        --num_images "${NUM_IMAGES}" \
        --num_images_for_eval_and_search "${NUM_IMAGES}" \
        --load_from "${ckpt_path}" \
        --output_dir "${EVAL_ROOT}" \
        --project "${EVAL_PROJECT}" \
        --exp_name "${exp_name}-steps${steps}-fid-${NUM_IMAGES}-${SLURM_JOB_ID:-manual}"

    MASTER_PORT=$(( MASTER_PORT + 1 ))
}

echo "[$(date --iso-8601=seconds)] starting JiT-B sphere VAE-start post1k probe"
echo "DATA_ROOT=${DATA_ROOT}"
echo "GPUS_PER_NODE=${GPUS_PER_NODE} GLOBAL_BSZ=${GLOBAL_BSZ} BATCH_SIZE=${BATCH_SIZE}"
echo "TC=${TC} PRE_STEPS=${PRE_STEPS} POST_STEPS=$(( EPOCHS * STEPS_PER_EPOCH )) KL_LIST=${KL_LIST}"
echo "START_SUPPORT_MODE=${START_SUPPORT_MODE} NUM_IMAGES=${NUM_IMAGES} STEPS_LIST=${STEPS_LIST}"

for STEPS in ${STEPS_LIST}; do
    run_eval "checkpoints/base/JiT-B.pth" "JiT_B-base-${START_SUPPORT_MODE}" "${STEPS}"
done

for KL_WEIGHT in ${KL_LIST}; do
    KL_TAG="${KL_WEIGHT/./p}"
    EXP_NAME="JiT_B-sphere-vae-start-tc0p30-kw${KL_TAG}-pre${PRE_STEPS}-post1k-${SLURM_JOB_ID:-manual}"
    echo "[$(date --iso-8601=seconds)] train KL=${KL_WEIGHT} exp=${EXP_NAME}"

    torchrun --nproc_per_node="${GPUS_PER_NODE}" --master_port="${MASTER_PORT}" \
        main_jit_vae_start.py \
        --project "${TRAIN_PROJECT}" \
        --exp_name "${EXP_NAME}" \
        --output_dir "${RUN_ROOT}" \
        --batch_size "${BATCH_SIZE}" \
        "${DATA_ARGS[@]}" \
        --load_from checkpoints/base/JiT-B.pth \
        --model JiT_B --rope_2d --learned_pe --legacy_time_convention \
        --cfg 3.0 --interval_min 0.1 --interval_max 1.0 \
        --start_support_mode "${START_SUPPORT_MODE}" \
        --ema_type edm \
        --num_sampling_steps 1 \
        --print_freq 20 --milestone_interval 0 --save_freq 2 \
        --epochs "${EPOCHS}" --steps_per_epoch "${STEPS_PER_EPOCH}" --warmup_epochs 1 \
        --lr 1e-5 --lr_sched cosine --min_lr 0.0 \
        --vae_start_tc "${TC}" \
        --vae_start_pre_steps "${PRE_STEPS}" \
        --vae_start_kl_weight "${KL_WEIGHT}" \
        --vae_start_cycle_weight 1.0 \
        --vae_start_hidden 64 \
        --vae_start_lr 2e-4 \
        --vae_start_sample_mode posterior \
        --train_t_min 0.02 --train_t_max 1.0 \
        --num_workers "${NUM_WORKERS}" \
        --disable_wandb

    MASTER_PORT=$(( MASTER_PORT + 1 ))

    CKPT_DIR="${RUN_ROOT}/${TRAIN_PROJECT}/${EXP_NAME}/checkpoints"
    CKPT_PATH="$(find "${CKPT_DIR}" -maxdepth 1 -name 'step_*.pth' | sort | tail -1)"
    if [[ -z "${CKPT_PATH}" ]]; then
        echo "[ERR] no checkpoint found in ${CKPT_DIR}" >&2
        exit 3
    fi

    for STEPS in ${STEPS_LIST}; do
        run_eval "${CKPT_PATH}" "${EXP_NAME}" "${STEPS}"
    done
done

echo "[$(date --iso-8601=seconds)] finished JiT-B sphere VAE-start post1k probe"
