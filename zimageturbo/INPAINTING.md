# Z-Image Inpainting Architecture Guide

## Overview

Z-Image-Turbo uses two distinct inpainting strategies depending on whether ControlNet is active.

## Strategy 1: ControlNet Inpainting (Native)

When ControlNet is loaded and active, inpainting is handled **entirely by the ControlNet's 33-channel input**. No external callback is used.

### 33-Channel Input Format

```
[control_latent(16ch), mask(1ch), inpaint_latent(16ch)]
```

- **control_latent**: VAE-encoded control image (canny, pose, depth, etc.)
- **mask**: Inverted binary mask — `1 = keep`, `0 = regenerate`
- **inpaint_latent**: VAE-encoded inpaint image with mask applied

### Mask Preprocessing (ComfyUI Convention)

```python
# 1. Invert mask: UI mask is 1=regenerate, model expects 1=keep
mask_keep = (1.0 - mask).round()

# 2. Apply mask to inpaint image: preserve kept areas, gray-fill regenerate areas
inpaint_image = (original_image - 0.5) * mask_keep + 0.5

# 3. Encode to latent
inpaint_latent = vae.encode(inpaint_image)

# 4. Mask channel: inverted, downsampled to latent resolution via nearest
mask_channel = interpolate(1.0 - mask, latent_size, mode='nearest')
```

The `.round()` on the keep-mask ensures binary boundaries — no partial transparency leaks into the ControlNet input.

### Why No Callback

The ControlNet model was **trained** with this 33-channel conditioning. It learns to:
- Reconstruct kept areas from the inpaint latent
- Generate new content in masked areas guided by the control signal
- Blend boundaries naturally through its learned attention mechanism

Adding an external noise-masking callback **conflicts** with this — the callback overwrites latent values that the ControlNet has already adjusted, creating stripe artifacts at patch boundaries.

## Strategy 2: RePaint Callback (No ControlNet)

When ControlNet is NOT active, `ZImagePipeline` is pure text-to-image with no inpainting support. We use a `callback_on_step_end` to inject noise-level masking.

### Algorithm (Flow Matching RePaint)

At each denoising step `t` with sigma schedule `σ_t → 0`:

```
σ = scheduler.sigmas[step + 1]
noised_original = σ * noise + (1 - σ) * original_latent
latents = mask * latents + (1 - mask) * noised_original
```

Where `mask = 1` means regenerate, `mask = 0` means keep original.

This forces non-masked regions to follow the noise schedule of the original image, while masked regions are freely denoised by the model.

### Limitations

- No bidirectional feedback between masked/unmasked regions
- Hard mask boundaries can produce visible seams with few steps (8)
- Works best for simple inpainting without ControlNet guidance

## Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `guidance_scale` | 1.0 | Turbo model, no CFG amplification |
| `shift` | 3.0 | Scheduler sigma shift |
| `control_context_scale` | 0.75 | ControlNet strength (0.65–1.0 optimal) |
| `steps` | 8 | Turbo distilled model |
| `sampler` | res_multistep | 2nd-order multistep (exponential integrator) |

## Reference

- ComfyUI implementation: `comfy_extras/nodes_model_patch.py` → `ZImageControlPatch`
- ControlNet model: `comfy/ldm/lumina/controlnet.py` → `ZImage_Control`
- Mask convention: ComfyUI inverts mask before passing to model (`mask = 1.0 - mask`)
