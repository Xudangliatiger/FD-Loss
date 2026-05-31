# ImageNet VAE-start Failure Analysis

Date: 2026-05-27

## Question

Why does the proposed startpoint-coupling idea behave much worse on ImageNet than on Fashion-MNIST?

The current ImageNet implementation is `main_jit_vae_start.py`.  It trains an
image-conditioned VAE startpoint only near the raw JiT start endpoint:

- FD-Loss JiT time convention: raw `t=0` is data, raw `t=1` is the noise/start endpoint.
- `vae_start_tc=0.30` means VAE starts are used for raw `t >= 0.70`.
- Evaluation in `eval_all_fds.py` does **not** use the VAE encoder.  It evaluates
  the post-trained JiT from random Gaussian starts.

Therefore the central risk is train-test mismatch:

```text
training near start: x_t = (1 - t) x0 + t z_vae(x0)
inference:           x_t starts from random eps
```

## Current Evidence

### 1. KL sweep shows high KL only reduces damage

Job `42738799` ran `pre_steps=1000`, `post_steps=1000`, `t_c=0.30`, 50k eval.

| KL | FID(JiT) | FID(ADM) | FDr | IS |
|---:|---:|---:|---:|---:|
| 1 | 347.16 | 345.98 | 205.94 | 1.4895 |
| 3 | 335.31 | 334.08 | 198.86 | 1.4201 |
| 5 | 331.72 | 330.54 | 196.75 | 1.2910 |
| 7 | 327.07 | 325.98 | 194.04 | 1.3176 |
| 9 | 323.29 | 322.26 | 191.82 | 1.3639 |

Reference:

| run | FID(JiT) | FID(ADM) | FDr | IS |
|---|---:|---:|---:|---:|
| base JiT-B | 308.64 | 307.90 | 183.27 | 1.7776 |
| continued gbsz64 10k | 299.99 | 299.39 | 178.21 | 1.9144 |
| cutoff-only gbsz64 10k | 272.99 | 272.67 | 162.30 | 1.5507 |
| VAE-start+cutoff KL=1 gbsz64 10k | 239.95 | 239.43 | 142.52 | 1.9813 |

Interpretation:

- Larger KL makes `z_vae` closer to Gaussian and monotonically improves the 1k probe.
- But even KL=9 is worse than base JiT-B.
- This means high KL is mostly reducing train-test mismatch damage; it is not yet
  creating a better sampleable start distribution.
- The 10k VAE-start run can improve, so ImageNet is not simply impossible. The
  failing condition is the short, high-KL probe and/or weak random-start adaptation.

### 2. Encoder diagnostics show image leakage into the startpoint

VAE-only diagnostics with frozen JiT show:

| run | cycle MSE z_start | random-start MSE | KL/dim | mu std | mu-x0 cosine | z_start std |
|---|---:|---:|---:|---:|---:|---:|
| KL=0.25, 5000 steps | 0.095 | 0.389 | 0.0865 | 0.364 | 0.495 | 1.055 |
| KL=1.0, 1000 steps | 0.141 | 0.463 | 0.0182 | 0.181 | 0.216 | 1.010 |
| KL=5.0, 1000 steps | 0.156 | 0.463 | 0.0042 | 0.091 | 0.571 | 1.003 |

The sampled start is numerically close to Gaussian in std, but the mean direction
is image-aligned.  This matches the visualizations: `mu(x)` and even sampled
`z_start` retain layout/color/edge information on ImageNet.

Interpretation:

- The encoder is not learning a clean sampleable latent prior.
- It learns a low-amplitude image-conditioned bias added to random noise.
- At inference, random Gaussian starts lack this matched image-conditioned bias.

### 3. The current mechanism updates JiT under two incompatible start distributions

In `main_jit_vae_start.py`, for each post-training batch:

```python
vae_mask = (t >= (1.0 - args.vae_start_tc))
start = vae_mask * vae_start + (1.0 - vae_mask) * eps
x_t = (1.0 - t) * x0 + t * start
```

For about `t_c` of training times, JiT learns from `z_vae(x0)`.  For the rest,
it learns from Gaussian `eps`.  But evaluation always starts from Gaussian `eps`.

On Fashion-MNIST this mismatch is weak because the image manifold is simple and
the startpoint bias mostly encodes coarse shape.  On ImageNet, the same shortcut
contains high-level layout/color information that cannot be sampled independently.

### 4. Shuffling the VAE start breaks the signal

Job `42794535` added a direct pairing test.  For each checkpoint, keep the same
batch marginal VAE start distribution but shuffle `mu/logvar` across images
before sampling `z_start`.  This preserves the approximate marginal prior while
destroying the image-start pairing.

| checkpoint | paired MSE | shuffled MSE | random MSE | mu-x0 cos | shuffled mu-x0 cos | z-x0 cos | shuffled z-x0 cos |
|---|---:|---:|---:|---:|---:|---:|---:|
| KL=1, post1k | 0.0533 | 0.5349 | 0.3867 | 0.7634 | 0.0972 | 0.1121 | 0.0142 |
| KL=9, post1k | 0.0797 | 0.5059 | 0.3921 | 0.7400 | 0.1071 | 0.0525 | 0.0082 |
| KL=1, post10k | 0.0318 | 0.5678 | 0.3686 | 0.8092 | 0.0740 | 0.1257 | 0.0138 |

This is the strongest evidence so far.  The useful signal is not just that
`z_start` has a better marginal distribution than Gaussian noise.  When the
same marginal VAE statistics are detached from their original image, reconstruction
becomes worse than a random Gaussian start.  The 10k checkpoint, which has the
best 1-step FID among the VAE-start runs, is also the most clearly pair-dependent.

Therefore the VAE-start branch is learning an image-conditioned shortcut:

```text
paired z_start(x)       -> useful endpoint code for x
shuffled z_start(x')    -> harmful endpoint code for x
random eps              -> ordinary JiT start
```

This explains why the method transfers poorly to random-start generation: the
paired endpoint code is accurate only when it comes from the same image.

## Working Hypothesis

The ImageNet failure is caused by a mismatch between:

1. **Paired, image-conditioned starts during training**, which can contain image
   information even when their marginal std is near 1.
2. **Unpaired Gaussian random starts during inference**, which do not contain the
   same information.

Raising KL moves the method back toward pure Gaussian starts and reduces this
mismatch, which explains the monotonic improvement from KL=1 to KL=9.  It does
not add useful coupling, so it remains worse than base JiT in the 1k probe.

The stronger 10k VAE-start result is real, but the shuffle test shows it is not
evidence that the VAE prior is inherently sampleable.  It likely combines longer
random-start adaptation/post-training with a paired endpoint shortcut that cannot
be used directly at inference.

### 5. Multi-step sampling does not remove the gap

Job `42793578` is a quick 1GPU/5k diagnostic for base, KL=1 post1k, and
KL=9 post1k at 1, 2, and 4 sampling steps.

| run | steps | FID(JiT) | FID(ADM) | FDr | IS |
|---|---:|---:|---:|---:|---:|
| base | 1 | 311.82 | 311.07 | 185.16 | 1.7587 |
| base | 2 | 80.72 | 80.34 | 47.82 | 17.9804 |
| base | 4 | 22.16 | 22.02 | 13.11 | 90.4656 |
| KL=1 post1k | 1 | 349.16 | 347.98 | 207.13 | 1.4934 |
| KL=1 post1k | 2 | 106.79 | 106.23 | 63.23 | 11.7145 |
| KL=1 post1k | 4 | 120.79 | 120.73 | 71.86 | 11.3080 |
| KL=9 post1k | 1 | 324.85 | 323.82 | 192.75 | 1.3645 |
| KL=9 post1k | 2 | 106.62 | 106.04 | 63.12 | 11.1035 |
| KL=9 post1k | 4 | 91.76 | 91.75 | 54.61 | 15.7113 |

This rules out the simple explanation that ImageNet is only too hard for one
step.  Base JiT-B improves dramatically with more steps.  The VAE-start
checkpoints also improve from 1 to 2 steps, but they remain far worse than base,
and KL=1 even degrades from 2 to 4 steps.  The learned VAE-start post-training
has altered the vector field in a way that hurts random-start integration, not
just one-step extrapolation.

### 6. Formal 50k multistep eval confirms the same failure

Job `42796368` completed the reporting-quality 50k eval for base JiT-B and
the KL sweep checkpoints at 2 and 4 sampling steps.  The failed predecessor
`42791416` landed on bad node `lrdn2118`; the completed rerun excluded
`lrdn2272,lrdn3214,lrdn2118`.

| run | steps | FID(JiT) | FID(ADM) | FDr | IS |
|---|---:|---:|---:|---:|---:|
| base | 2 | 73.91 | 73.54 | 43.77 | 20.8858 |
| base | 4 | 16.35 | 16.20 | 9.65 | 131.5321 |
| KL=1 | 2 | 101.55 | 100.99 | 60.11 | 12.2786 |
| KL=1 | 4 | 112.92 | 112.85 | 67.18 | 12.5932 |
| KL=3 | 2 | 96.97 | 96.44 | 57.41 | 12.5647 |
| KL=3 | 4 | 92.15 | 92.10 | 54.82 | 16.2918 |
| KL=5 | 2 | 103.77 | 103.16 | 61.40 | 11.3216 |
| KL=5 | 4 | 94.20 | 94.14 | 56.03 | 15.8135 |
| KL=7 | 2 | 103.05 | 102.46 | 60.99 | 11.2756 |
| KL=7 | 4 | 91.72 | 91.67 | 54.57 | 16.1013 |
| KL=9 | 2 | 100.14 | 99.58 | 59.27 | 11.8770 |
| KL=9 | 4 | 85.47 | 85.44 | 50.86 | 17.5765 |

The formal 50k result matches the quick 5k diagnostic:

- Base JiT-B improves strongly when moving from 1 step to 2/4 steps.
- Every VAE-start KL checkpoint is worse than the base model at both 2 and 4
  steps.
- KL=9 is the least damaging VAE-start variant, but it is still far behind the
  base 4-step result.

This confirms that the failure is not just one-step extrapolation.  VAE-start
post-training damages the random-start trajectory itself.

### 7. DINOv3 sphere bridge experiments

After the VAE-start failure, we tested a Sphere-Encoder-style variant that uses
frozen DINOv3 patch tokens as the paired start latent:

`DINOv3 patch tokens -> RMS sphere -> trainable ViT bridge -> pixel start -> JiT`.

Use the following `v0.1.x` IDs when discussing this DINO-sphere line.  The
version ID identifies the method configuration; job ids identify particular
executions.

| version | status | job | scale | DINO frontend | bridge / losses | purpose |
|---|---|---:|---|---|---|---|
| v0.1.0 | completed | mixed early jobs | 1GPU/5k | full patch sphere `[256,768]` | L2 or L1/L2/LPIPS, no SIGReg | early feasibility probes |
| v0.1.1 | completed | `43014272` | 1GPU/5k | full patch sphere `[256,768]` | paired L2 + clean bridge KL | test Gaussian moment matching on `B(z_clean)` |
| v0.1.2 | completed/cancelled | `43128742`, `43138054` | 1GPU/10k and 8GPU/125k | full patch sphere `[256,768]` | L1/L2/LPIPS + bridge-after SIGReg | main full-DINO sphere baseline |
| v0.1.3 | cancelled | `43460286`, `43478074` | 1GPU probes | full patch sphere `[256,768]` | v0.1.2 + random sphere cycle | test image-DINO consistency from random generated images |
| v0.1.4 | invalidated | `43539278`, `43573382` before commit `eaa2ed5` | 1GPU/5k | intended feature norm / project128, but actually full patch sphere | L1/L2/LPIPS + SIGReg | invalid ablation: frontend options were not passed into `dino_patch_sphere` |
| v0.1.5 | completed | `43584026` | 1GPU/5k, gbsz80 | full patch sphere `[256,768]` | L1/L2/LPIPS + bridge-after SIGReg | fixed full-768 single-GPU baseline |
| v0.1.6 | running | `43596056` | 1GPU/5k, gbsz80 | sample-channel-normalized patch sphere `[256,768]` | L1/L2/LPIPS + bridge-after SIGReg | real feature-normalization ablation after commit `eaa2ed5` |
| v0.1.7 | running/planned | `43596057`; 8GPU pending | 1GPU/5k gbsz80; 8GPU/125k gbsz1024 | projected patch sphere `[256,128] -> [256,768]` | L1/L2/LPIPS + bridge-after SIGReg | real project128 method after commit `eaa2ed5`; training recipe belongs in the run folder name |

`v0.1.4` should not be used as evidence for feature normalization or
projection.  A bug in `build_vae_start_encoder()` failed to pass
`dinov2_start_feature_norm` and `dinov2_start_project_dim` into
`DINOv2PatchSphereStartEncoder`; this was fixed in local commit `fa89511` and
Leonardo commit `eaa2ed5`.

The bridge and JiT are trained jointly while DINOv3 remains frozen.  The early
baseline uses only paired cycle losses.  Version v0.1.1 adds a deterministic
moment KL on the clean bridge start `B(z_clean)`:

`0.5 * (mean(B)^2 + var(B) - log var(B) - 1)`.

This is not a VAE KL: there is no learned variance.  It only pushes the
deterministic clean bridge start marginal toward standard Gaussian statistics.

| version | run | z-start MSE ↓ | clean-start MSE ↓ | clean bridge KL ↓ | clean start std | random sphere one-step std |
|---|---|---:|---:|---:|---:|---:|
| v0.1.0 | DINO sphere + L2 | 0.0743 | 0.0703 | - | 0.8560 | 0.2542 |
| v0.1.0 | DINO sphere + L1/L2/LPIPS | 0.0926 | 0.0899 | - | 0.8049 | 0.2737 |
| v0.1.1 | DINO sphere + L2 + clean KL 0.05 | 0.0743 | 0.0705 | 0.0002 | 1.0064 | 0.3800 |

Current fixed 1GPU comparison at intermediate checkpoints:

| version | job | step | loss ↓ | cycle loss ↓ | z-start MSE ↓ | clean-start MSE ↓ | random pixel std | random sphere std | random bridge one-step std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v0.1.5 | `43584026` | 5000 | 1.1575 | 1.0073 | 0.1434 | 0.0892 | 0.2061 | 0.2508 | 0.2563 |
| v0.1.6 | `43596056` | 2500 | 1.2281 | 1.0757 | 0.1661 | 0.1123 | 0.2327 | 0.2162 | 0.2527 |
| v0.1.7 | `43596057` | 2500 | 1.1265 | 0.9721 | 0.1256 | 0.1023 | 0.1757 | 0.1965 | 0.1684 |

The clean-start KL does exactly what it is designed to do: `B(z_clean)` becomes
nearly standard-normal in first and second moments, without hurting the paired
reconstruction.  However, the random sphere branch still produces mostly colored
grid/texture patterns rather than stable ImageNet objects.  Therefore the main
missing piece is not just pixel-start marginal Gaussianity.  The harder problem
is semantic transport from random sphere latents through the bridge into a
startpoint that JiT can decode as an image.

Training scale is not a separate method version in this table.  For example,
the 1GPU probe and the planned 8GPU scale-up are both `v0.1.7`; their folder
names should carry recipe suffixes such as `v0p1p7-gbsz80-post5000` or
`v0p1p7-gbsz1024-post125k`.

After fixing the DINO frontend option bug, v0.1.7 starts to separate from the
full-768 single-GPU baseline numerically: the projected latent has lower paired
cycle loss and lower random bridge one-step variance at step 2500.  This is not
yet a qualitative success: the bridge output still contains structured carrier
textures and random sphere samples are still blurry.  The 8GPU recipe should be
treated as a scale test for v0.1.7, not as proof that the projection solves the
sampleability problem by itself.

## Next Experiments

1. Add a random-start branch during post-training so the model sees the actual
   inference start distribution while still receiving VAE-start endpoint coupling.
2. Test a marginal-matched start prior: sample `eps + s * mu(x_perm)` or shuffle
   `mu(x)` across labels/images. If this fails, the benefit requires exact pairing
   and is not sampleable.
3. For DINO/sphere starts, add a semantic transport objective on random sphere
   latents rather than only matching bridge-start first and second moments.
4. If v0.1.7 remains better at step 5000, run the same v0.1.7 method with an
   8GPU/gbsz1024/post125k recipe, using the recipe in the run folder name.
