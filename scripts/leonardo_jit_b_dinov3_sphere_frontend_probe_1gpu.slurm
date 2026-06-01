#!/bin/bash
#SBATCH --job-name=jitB_dinofe1
#SBATCH --account=EUHPC_D32_028
#SBATCH --partition=boost_usr_prod
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --exclude=lrdn2272,lrdn3214,lrdn2118
#SBATCH --output=/leonardo_scratch/large/userexternal/dxu00000/2027-ICLR-PixelGen/FD-Loss/runs/slurm/%x-%j.out
#SBATCH --error=/leonardo_scratch/large/userexternal/dxu00000/2027-ICLR-PixelGen/FD-Loss/runs/slurm/%x-%j.err

set -euo pipefail

cd /leonardo_scratch/fast/EUHPC_D32_028/dongli/FD-Loss

FRONTEND_PROBE="${FRONTEND_PROBE:-feature_norm}"
POST_STEPS="${POST_STEPS:-5000}"
GLOBAL_BSZ="${GLOBAL_BSZ:-80}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"

export VAE_START_ENCODER_TYPE="${VAE_START_ENCODER_TYPE:-dino_patch_sphere}"
export DINO_BRIDGE_TYPE="${DINO_BRIDGE_TYPE:-vit_decoder}"
export START_SUPPORT_MODE="${START_SUPPORT_MODE:-pixel_sphere}"
export CYCLE_L1_WEIGHT="${CYCLE_L1_WEIGHT:-1.0}"
export CYCLE_L2_WEIGHT="${CYCLE_L2_WEIGHT:-1.0}"
export CYCLE_LPIPS_WEIGHT="${CYCLE_LPIPS_WEIGHT:-1.0}"
export SIGREG_WEIGHT="${SIGREG_WEIGHT:-1.0}"
export SIGREG_NUM_SLICES="${SIGREG_NUM_SLICES:-32}"
export SIGREG_MAX_SAMPLES="${SIGREG_MAX_SAMPLES:-1024}"
export VAE_START_VIS_EVERY="${VAE_START_VIS_EVERY:-500}"
export INF_EVERY="${INF_EVERY:-500}"
export INF_IMAGES="${INF_IMAGES:-16}"
export PRE_STEPS="${PRE_STEPS:-0}"
export POST_STEPS GLOBAL_BSZ GRAD_ACCUM_STEPS

case "${FRONTEND_PROBE}" in
    full768_bridge_only)
        export DINO_FEATURE_NORM="${DINO_FEATURE_NORM:-none}"
        export DINO_PROJECT_DIM="${DINO_PROJECT_DIM:-0}"
        export LATENT_SIGREG_WEIGHT="${LATENT_SIGREG_WEIGHT:-0.0}"
        export FREEZE_GENERATION_MODEL_POST=1
        TAG="v0p1p8-full768-bridgeonly"
        ;;
    feature_norm)
        export DINO_FEATURE_NORM="${DINO_FEATURE_NORM:-sample_channel}"
        export DINO_PROJECT_DIM="${DINO_PROJECT_DIM:-0}"
        export LATENT_SIGREG_WEIGHT="${LATENT_SIGREG_WEIGHT:-0.0}"
        TAG="featnorm-${DINO_FEATURE_NORM}"
        ;;
    project128)
        export DINO_FEATURE_NORM="${DINO_FEATURE_NORM:-none}"
        export DINO_PROJECT_DIM="${DINO_PROJECT_DIM:-128}"
        export LATENT_SIGREG_WEIGHT="${LATENT_SIGREG_WEIGHT:-0.0}"
        TAG="proj${DINO_PROJECT_DIM}"
        ;;
    project128_bridge_only)
        export DINO_FEATURE_NORM="${DINO_FEATURE_NORM:-none}"
        export DINO_PROJECT_DIM="${DINO_PROJECT_DIM:-128}"
        export LATENT_SIGREG_WEIGHT="${LATENT_SIGREG_WEIGHT:-0.0}"
        export FREEZE_GENERATION_MODEL_POST=1
        TAG="v0p1p9-project128-bridgeonly"
        ;;
    project128_latentsigreg)
        export DINO_FEATURE_NORM="${DINO_FEATURE_NORM:-none}"
        export DINO_PROJECT_DIM="${DINO_PROJECT_DIM:-128}"
        export LATENT_SIGREG_WEIGHT="${LATENT_SIGREG_WEIGHT:-1.0}"
        export LATENT_SIGREG_TARGET="${LATENT_SIGREG_TARGET:-clean}"
        TAG="proj${DINO_PROJECT_DIM}-latentsigreg${LATENT_SIGREG_WEIGHT}"
        ;;
    *)
        echo "[ERR] unknown FRONTEND_PROBE=${FRONTEND_PROBE}" >&2
        exit 2
        ;;
esac

export EXP_NAME="${EXP_NAME:-JiT_B-dinov3-patchsphere-vitbridge-${TAG}-sigreg1-l1l2lpips1-gbsz${GLOBAL_BSZ}-post${POST_STEPS}-0531-${SLURM_JOB_ID:-manual}}"

exec bash scripts/leonardo_jit_b_dinov3_sphere_joint_1gpu.slurm
