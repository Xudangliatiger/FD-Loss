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

## Pending Verification

One follow-up eval is still queued:

- `42791416`: formal 50k eval for base and KL sweep checkpoints at 2-step and 4-step.

The 5k diagnostic is already strong enough to reject the pure "one-step is too
hard" hypothesis.  The 50k run should be treated as confirmation and for final
reporting-quality numbers.

## Next Experiments

1. For the good 10k VAE-start checkpoint, run the same 1/2/4-step comparison.
2. Add a random-start branch during post-training so the model sees the actual
   inference start distribution while still receiving VAE-start endpoint coupling.
3. Test a marginal-matched start prior: sample `eps + s * mu(x_perm)` or shuffle
   `mu(x)` across labels/images. If this fails, the benefit requires exact pairing
   and is not sampleable.
