"""Headless RAM usage test — loads model + controlnet and reports memory at each stage.
Tests meta-device init to avoid fp32 allocation."""
import os, sys, gc, time, psutil

# Fix encoding
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def mb(bytes_val):
    return bytes_val / (1024 * 1024)

def report(label):
    proc = psutil.Process()
    rss = proc.memory_info().rss
    vm = psutil.virtual_memory()
    gpu_alloc = 0
    try:
        import torch
        if torch.cuda.is_available():
            gpu_alloc = torch.cuda.memory_allocated()
    except:
        pass
    print(f"[{label}] RSS={mb(rss):.0f}MB  SysRAM={vm.percent}%  GPU={mb(gpu_alloc):.0f}MB")
    return rss

def main():
    report("startup")

    import torch
    from safetensors.torch import load_file as load_safetensors
    from diffusers import AutoencoderKL, ZImageTransformer2DModel, ZImagePipeline
    from transformers import AutoTokenizer, Qwen3ForCausalLM
    from accelerate import init_empty_weights
    report("imports")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODELS_ROOT = os.path.join(BASE_DIR, "models")
    Z_IMAGE_ROOT = os.path.join(MODELS_ROOT, "z_image_turbo")
    TE_ROOT = os.path.join(MODELS_ROOT, "text_encoders")
    VAE_ROOT = os.path.join(MODELS_ROOT, "vae")
    TOKENIZER_ROOT = os.path.join(MODELS_ROOT, "tokenizer")

    from app_engine import ResMultistepFlowMatchScheduler

    weight_dtype = torch.bfloat16

    # --- 1. Transformer (meta-device init to avoid fp32 allocation) ---
    print("\n=== Loading Transformer (meta-device init) ===")
    tf_config_dir = os.path.join(Z_IMAGE_ROOT, "transformer")
    weight_path = os.path.join(tf_config_dir, "diffusion_pytorch_model.safetensors")

    raw_sd = load_safetensors(weight_path)
    report("safetensors loaded")

    from app_engine import _convert_z_image_state_dict
    needs_conv = any(".attention.qkv.weight" in k for k in raw_sd.keys())
    if needs_conv:
        raw_sd = _convert_z_image_state_dict(raw_sd)
    report("QKV converted")

    tf_config = ZImageTransformer2DModel.load_config(tf_config_dir)

    # Meta-device init: creates model structure with ZERO memory allocation
    with init_empty_weights():
        _transformer = ZImageTransformer2DModel.from_config(tf_config)
    report("from_config (meta device - zero alloc)")

    # assign=True: directly use state_dict bf16 tensors as parameters
    _transformer.load_state_dict(raw_sd, strict=False, assign=True)
    del raw_sd; gc.collect()
    report("load_state_dict + del raw_sd")

    _transformer.to(device="cuda")
    gc.collect()
    report("transformer → GPU")

    # --- 2. VAE ---
    print("\n=== Loading VAE ===")
    _vae = AutoencoderKL.from_pretrained(VAE_ROOT, torch_dtype=weight_dtype, device_map="cuda")
    gc.collect()
    report("VAE loaded (device_map=cuda)")

    # --- 3. Text Encoder ---
    print("\n=== Loading Text Encoder ===")
    _text_encoder = Qwen3ForCausalLM.from_pretrained(TE_ROOT, torch_dtype=weight_dtype, device_map="cuda")
    gc.collect()
    report("text_encoder loaded (device_map=cuda)")

    # --- 4. Tokenizer + Scheduler ---
    _tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ROOT)
    scheduler_dir = os.path.join(Z_IMAGE_ROOT, "scheduler")
    _scheduler = ResMultistepFlowMatchScheduler.from_pretrained(scheduler_dir)

    # --- 5. Pipeline ---
    print("\n=== Assembling Pipeline ===")
    pipe = ZImagePipeline(
        transformer=_transformer, vae=_vae, text_encoder=_text_encoder,
        tokenizer=_tokenizer, scheduler=_scheduler,
    )
    del _transformer, _vae, _text_encoder
    gc.collect()
    report("pipeline assembled")

    # --- 6. ControlNet ---
    print("\n=== Loading ControlNet ===")
    cn_root = os.path.join(MODELS_ROOT, "Personalized_Model")
    import glob as g
    cn_files = g.glob(os.path.join(cn_root, "*.safetensors"))
    if cn_files:
        cn_path = cn_files[0]
        print(f"Loading: {os.path.basename(cn_path)}")
        from zimage_control import load_controlnet_module, install_control_hooks
        ctrl = load_controlnet_module(cn_path, device='cuda', dtype=weight_dtype)
        report("controlnet loaded")
        install_control_hooks(pipe.transformer, ctrl)
        report("hooks installed")
    else:
        print("No ControlNet files found, skipping")

    print("\n" + "="*60)
    final_rss = report("FINAL")
    print(f"\nFinal RSS: {mb(final_rss):.0f} MB ({mb(final_rss)/1024:.2f} GB)")
    print("="*60)

    # Cleanup
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
