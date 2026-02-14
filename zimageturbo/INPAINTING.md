# Z-Image Technical Guide

## 1. RAM Optimization — Meta-Device Init

### Problem

`Model.from_config()` allocates ALL parameters in fp32 first (~14GB for transformer, ~6GB for ControlNet).
Even with `assign=True` in `load_state_dict()`, Python's pymalloc on Windows **never returns freed pages to OS**.
The process RSS stays bloated with dead memory.

### Trial & Error

| Attempt | Approach | Result (RSS) | Verdict |
|---------|----------|-------------|---------|
| Baseline | `from_config()` + `load_state_dict(strict=False)` + `.to(cuda)` | ~24 GB | fp32 init → permanent ghost pages |
| #1 | Properties + immediate `.to("cuda")` per component + `gc.collect()` | 24.87 GB | Marginal. Dead pages already committed |
| #2 | `load_safetensors(device="cuda")` + `device_map="cuda"` | 32.29 GB | **Worse**. GPU→CPU→GPU double copy |
| #3 | `assign=True` in `load_state_dict` + `device_map="cuda"` for VAE/TE | 24.35 GB | Marginal. fp32 init is the real problem |
| **#4 (Final)** | **`init_empty_weights()` + `assign=True`** | **959 MB** | **96% reduction** |

### Why #2 Failed

`load_safetensors(device="cuda")` loads tensors directly to GPU. But `load_state_dict()` copies them back to CPU-allocated model parameters, then `.to("cuda")` copies them again. Triple allocation: GPU state_dict + CPU fp32 params + GPU final params.

### Solution (Current)

```python
from accelerate import init_empty_weights

# 1. Create model skeleton on meta device (ZERO memory)
with init_empty_weights():
    model = ZImageTransformer2DModel.from_config(config)

# 2. Fill with bf16 tensors from safetensors (assign=True swaps pointers)
model.load_state_dict(state_dict, strict=False, assign=True)
del state_dict; gc.collect()

# 3. Move to GPU (only bf16 tensors exist, ~7GB briefly on CPU)
model.to(device="cuda")
gc.collect()
```

Same pattern for ControlNet in `zimage_control.py`:
```python
with init_empty_weights():
    module = ZImageControlModule(**config)
module.load_state_dict(sd, strict=False, assign=True)
```

### Key Insight

`assign=True` makes `load_state_dict` **replace** parameter tensors instead of copying into them.
Combined with meta-device init (no real tensors to copy into), the model never touches system RAM beyond the transient safetensors read buffer.

### Results

| Metric | Before | After |
|--------|--------|-------|
| RSS (base + ControlNet) | 23,855 MB | 959 MB |
| SysRAM% | 56.8% | 21.3% |
| GPU VRAM | 26,143 MB | 26,143 MB (unchanged) |

VAE and Text Encoder use `from_pretrained(device_map="cuda")` which internally does the same meta-device pattern via accelerate.

---

## 2. Inpainting Architecture

### Strategy 1: ControlNet Inpainting (Native 33ch)

When ControlNet is active, inpainting is handled **entirely by the 33-channel input**. No callback.

**33-Channel Format**: `[control_latent(16), mask(1), inpaint_latent(16)]`

**Mask Preprocessing (ComfyUI Convention)**:
```python
mask_keep = (1.0 - mask).round()                        # invert + binarize
inpaint_image = (original_image - 0.5) * mask_keep + 0.5  # gray-fill masked areas
inpaint_latent = vae.encode(inpaint_image)
mask_channel = interpolate(1.0 - mask, latent_size, mode='nearest')
```

**Why no callback**: The ControlNet was **trained** with this conditioning. Adding a noise-masking callback overwrites latent values the ControlNet already adjusted → stripe artifacts at patch boundaries.

### Strategy 2: RePaint Callback (No ControlNet)

`ZImagePipeline` is pure text-to-image. We inject noise-level masking via `callback_on_step_end`:

```
σ = scheduler.sigmas[step + 1]
noised_original = σ * noise + (1 - σ) * original_latent
latents = mask * latents + (1 - mask) * noised_original
```

Limitations: No bidirectional feedback, hard seams with few steps (8).

---

## 3. Sampler: res_multistep

2nd-order exponential integrator based on ComfyUI's k_diffusion implementation.

- First step: Euler
- Subsequent steps: Uses previous model output for 2nd-order correction (phi1/phi2 functions)
- Last step (sigma < eps): Jump directly to denoised
- No extra model evaluations — reuses `_old_denoised` from previous step

Flow matching velocity → x0: `denoised = sample - sigma * model_output`

---

## 4. Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `guidance_scale` | 1.0 | Turbo model, no CFG amplification |
| `shift` | 3.0 | Scheduler sigma shift |
| `control_context_scale` | 0.75 | ControlNet strength (0.65–1.0 optimal) |
| `steps` | 8 | Turbo distilled model |
| `sampler` | res_multistep | 2nd-order multistep exponential integrator |

## 5. Reference

- ComfyUI ControlNet: `comfy_extras/nodes_model_patch.py` → `ZImageControlPatch`
- ComfyUI ControlNet model: `comfy/ldm/lumina/controlnet.py` → `ZImage_Control`
- Mask convention: ComfyUI inverts mask before passing to model
- RAM benchmark: `zimageturbo/test_ram.py` (headless, no Gradio)
