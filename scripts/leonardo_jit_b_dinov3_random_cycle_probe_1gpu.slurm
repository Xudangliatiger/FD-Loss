#!/bin/bash
#SBATCH --job-name=jitB_dinorc1
#SBATCH --account=EUHPC_D32_028
#SBATCH --partition=boost_usr_prod
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --exclude=lrdn2272,lrdn3214,lrdn2118
#SBATCH --output=/leonardo_scratch/large/userexternal/dxu00000/2027-ICLR-PixelGen/FD-Loss/runs/slurm/%x-%j.out
#SBATCH --error=/leonardo_scratch/large/userexternal/dxu00000/2027-ICLR-PixelGen/FD-Loss/runs/slurm/%x-%j.err

set -euo pipefail

BASE_CKPT="/leonardo_scratch/large/userexternal/dxu00000/2027-ICLR-PixelGen/FD-Loss/runs/jit_dinov3_sphere_joint/JiT_B-dinov3-patchsphere-vitbridge-sigreg1-l1l2lpips1-gbsz1024-post125k-0530/checkpoints/step_0018749.pth"

export VAE_START_ENCODER_TYPE="${VAE_START_ENCODER_TYPE:-dino_patch_sphere}"
export DINO_BRIDGE_TYPE="${DINO_BRIDGE_TYPE:-vit_decoder}"
export GLOBAL_BSZ="${GLOBAL_BSZ:-48}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
export POST_STEPS="${POST_STEPS:-3000}"
export STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-500}"
export WARMUP_EPOCHS="${WARMUP_EPOCHS:-1}"
export CYCLE_L1_WEIGHT="${CYCLE_L1_WEIGHT:-1.0}"
export CYCLE_L2_WEIGHT="${CYCLE_L2_WEIGHT:-1.0}"
export CYCLE_LPIPS_WEIGHT="${CYCLE_LPIPS_WEIGHT:-1.0}"
export SIGREG_WEIGHT="${SIGREG_WEIGHT:-1.0}"
export RANDOM_CYCLE_WEIGHT="${RANDOM_CYCLE_WEIGHT:-0.10}"
export RANDOM_CYCLE_BSZ="${RANDOM_CYCLE_BSZ:-12}"
export VAE_START_VIS_EVERY="${VAE_START_VIS_EVERY:-500}"
export INF_EVERY="${INF_EVERY:-500}"
export INF_IMAGES="${INF_IMAGES:-16}"
export LOAD_FROM="${LOAD_FROM:-${BASE_CKPT}}"
export VAE_CKPT="${VAE_CKPT:-${BASE_CKPT}}"
export EXP_NAME="${EXP_NAME:-JiT_B-dinov3-patchsphere-vitbridge-sigreg1-randcycle0p10-rcbsz12-gbsz48-post3000-0531-${SLURM_JOB_ID:-manual}}"

exec bash scripts/leonardo_jit_b_dinov3_sphere_joint_1gpu.slurm
