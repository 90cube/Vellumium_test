"""
Standalone ControlNet module for Z-Image.
Adapted from ComfyUI's ZImage_Control architecture.

Key design: base transformer stays on GPU untouched.
ControlNet is a separate module with forward hooks — no pipeline replacement.
"""

import torch
import torch.nn as nn
from diffusers.models.transformers.transformer_z_image import ZImageTransformerBlock


class ZImageControlBlock(ZImageTransformerBlock):
    """Control transformer block with before_proj (block_id=0) and after_proj."""

    def __init__(self, layer_id, dim, n_heads, n_kv_heads, norm_eps, qk_norm,
                 modulation=True, block_id=0):
        super().__init__(layer_id, dim, n_heads, n_kv_heads, norm_eps, qk_norm, modulation)
        self.block_id = block_id
        if block_id == 0:
            self.before_proj = nn.Linear(dim, dim)
        self.after_proj = nn.Linear(dim, dim)

    def forward(self, c, x, attn_mask=None, freqs_cis=None, adaln_input=None):
        if self.block_id == 0:
            c = self.before_proj(c) + x
        c = super().forward(c, attn_mask, freqs_cis, adaln_input)
        c_skip = self.after_proj(c)
        return c_skip, c


class ZImageControlModule(nn.Module):
    """Standalone ControlNet module. Only control-specific layers (~295 keys).
    Base transformer stays completely untouched on GPU."""

    def __init__(self, dim=3840, n_heads=30, n_kv_heads=30, norm_eps=1e-5, qk_norm=True,
                 n_control_layers=6, control_in_dim=16, additional_in_dim=0,
                 refiner_control=False, broken=False):
        super().__init__()
        self.dim = dim
        self.n_control_layers = n_control_layers
        self.additional_in_dim = additional_in_dim
        self.control_in_dim = control_in_dim
        self.refiner_control = refiner_control
        self.broken = broken

        # Control transformer blocks
        self.control_layers = nn.ModuleList([
            ZImageControlBlock(i, dim, n_heads, n_kv_heads, norm_eps, qk_norm, block_id=i)
            for i in range(n_control_layers)
        ])

        # Control image embedder (patchify + linear)
        patch_size = 2
        f_patch_size = 1
        x_embedder = nn.Linear(
            f_patch_size * patch_size * patch_size * (control_in_dim + additional_in_dim),
            dim, bias=True
        )
        self.control_all_x_embedder = nn.ModuleDict(
            {f"{patch_size}-{f_patch_size}": x_embedder}
        )

        # Noise refiner
        n_refiner_layers = 2
        if refiner_control:
            self.control_noise_refiner = nn.ModuleList([
                ZImageControlBlock(layer_id, dim, n_heads, n_kv_heads, norm_eps, qk_norm,
                                   modulation=True, block_id=layer_id)
                for layer_id in range(n_refiner_layers)
            ])
        else:
            self.control_noise_refiner = nn.ModuleList([
                ZImageTransformerBlock(layer_id, dim, n_heads, n_kv_heads, norm_eps, qk_norm,
                                       modulation=True)
                for layer_id in range(n_refiner_layers)
            ])

        # Runtime state (set before each generation)
        self._active = False
        self._control_scale = 0.75
        self._temp_data = None  # (current_ctrl_idx, (c_skip, control_ctx))
        self._embedded_control = None
        self._control_latent = None  # [B, C, H, W] raw VAE-encoded latent
        self._image_seq_len = 0  # Number of image tokens (after patchify)

    def embed_control(self, control_latent, freqs_cis, adaln_input):
        """Embed control latent and optionally run noise refiner.

        Args:
            control_latent: [B, C, H, W] VAE-encoded latent
            freqs_cis: IMAGE-ONLY position embeddings (x_freqs_cis, NOT unified)
            adaln_input: timestep embedding
        Returns:
            embedded control context [B, seq_len, dim]
        """
        patch_size = 2
        f_patch_size = 1
        pH = pW = patch_size
        B, C, H, W = control_latent.shape

        # Patchify: [B, C, H, W] -> [B, (H/2)*(W/2), 4*C]
        control = control_latent.view(B, C, H // pH, pH, W // pW, pW)
        control = control.permute(0, 2, 4, 3, 5, 1).flatten(3).flatten(1, 2)

        # Embed: -> [B, seq_len, dim]
        control = self.control_all_x_embedder[f"{patch_size}-{f_patch_size}"](control)

        # Run noise refiner (only if NOT refiner_control)
        if not self.refiner_control:
            for layer in self.control_noise_refiner:
                control = layer(
                    control, None,
                    freqs_cis[:control.shape[0], :control.shape[1]],
                    adaln_input
                )

        return control

    def forward_control_block(self, layer_id, control_ctx, x, attn_mask, freqs_cis, adaln_input):
        """Run a single control block. Returns (c_skip, updated_control_ctx)."""
        return self.control_layers[layer_id](
            control_ctx, x,
            attn_mask=None,
            freqs_cis=freqs_cis[:control_ctx.shape[0], :control_ctx.shape[1]],
            adaln_input=adaln_input
        )

    def forward_noise_refiner_block(self, layer_id, control_ctx, x, attn_mask, freqs_cis, adaln_input):
        """Run a noise refiner control block. Returns (c_skip, updated_control_ctx)."""
        if self.refiner_control:
            if self.broken:
                if layer_id == 0:
                    return self.control_noise_refiner[layer_id](
                        control_ctx, x, attn_mask=None,
                        freqs_cis=freqs_cis[:control_ctx.shape[0], :control_ctx.shape[1]],
                        adaln_input=adaln_input
                    )
                if layer_id > 0:
                    out = None
                    for i in range(1, len(self.control_noise_refiner)):
                        o, control_ctx = self.control_noise_refiner[i](
                            control_ctx, x, attn_mask=None,
                            freqs_cis=freqs_cis[:control_ctx.shape[0], :control_ctx.shape[1]],
                            adaln_input=adaln_input
                        )
                        if out is None:
                            out = o
                    return (out, control_ctx)
            else:
                return self.control_noise_refiner[layer_id](
                    control_ctx, x, attn_mask=None,
                    freqs_cis=freqs_cis[:control_ctx.shape[0], :control_ctx.shape[1]],
                    adaln_input=adaln_input
                )
        return (None, control_ctx)


# --- Config detection ---

def detect_controlnet_config(state_dict):
    """Auto-detect ControlNet configuration from weight keys."""
    # Count control layers
    max_layer_id = -1
    for k in state_dict.keys():
        if k.startswith('control_layers.'):
            layer_id = int(k.split('.')[1])
            if layer_id > max_layer_id:
                max_layer_id = layer_id
    n_control_layers = max_layer_id + 1

    # Detect additional_in_dim from embedder weight shape
    embedder_key = 'control_all_x_embedder.2-1.weight'
    additional_in_dim = 0
    if embedder_key in state_dict:
        in_features = state_dict[embedder_key].shape[1]
        # in_features = patch_size^2 * (control_in_dim + additional_in_dim) = 4 * total
        total_in_dim = in_features // 4
        if total_in_dim > 16:
            additional_in_dim = total_in_dim - 16

    # Detect refiner_control (control noise refiner blocks have after_proj)
    refiner_control = 'control_noise_refiner.0.after_proj.weight' in state_dict

    # Detect broken (refiner after_proj is all zeros)
    broken = False
    if refiner_control:
        ref_weight = state_dict.get('control_noise_refiner.0.after_proj.weight')
        if ref_weight is not None:
            broken = torch.count_nonzero(ref_weight) == 0

    return {
        'n_control_layers': n_control_layers,
        'additional_in_dim': additional_in_dim,
        'control_in_dim': 16,
        'refiner_control': refiner_control,
        'broken': broken,
    }


def load_controlnet_module(controlnet_path, device='cpu', dtype=torch.bfloat16):
    """Load ControlNet weights into standalone ZImageControlModule."""
    from safetensors.torch import load_file as load_safetensors
    from accelerate import init_empty_weights

    sd = load_safetensors(controlnet_path)
    config = detect_controlnet_config(sd)

    print(f"    Config: {config['n_control_layers']} ctrl layers, "
          f"additional_in_dim={config['additional_in_dim']}, "
          f"refiner={config['refiner_control']}, broken={config['broken']}")

    # Meta-device init avoids fp32 allocation; assign=True fills with actual tensors
    with init_empty_weights():
        module = ZImageControlModule(**config)
    missing, unexpected = module.load_state_dict(sd, strict=False, assign=True)

    if missing:
        print(f"    Missing keys: {len(missing)} (first 5: {missing[:5]})")
    if unexpected:
        print(f"    Unexpected keys: {len(unexpected)} (first 5: {unexpected[:5]})")
    if not missing and not unexpected:
        print(f"    All {len(sd)} control keys loaded perfectly")

    del sd
    module = module.to(dtype=dtype, device=device)
    return module


# --- Hook system ---
# CRITICAL: In diffusers, unified = [image_tokens, caption_tokens] (IMAGE FIRST).
# Control operates only on IMAGE portion at positions [0, image_seq).
# Noise refiner operates on image-only tokens before unification.

def _make_pre_hook(control_module):
    """Reset control state at start of each transformer forward (= each denoising step)."""
    def hook(module, args):
        if control_module._active:
            control_module._embedded_control = None
            control_module._temp_data = None
    return hook


def _make_main_block_hook(control_module, block_idx, n_blocks):
    """Forward hook for a main transformer block -- injects control signal.

    Main blocks process unified [image, caption] tokens (IMAGE FIRST in diffusers).
    Control operates only on the image portion at positions [0, image_seq).
    """
    def hook(module, args, output):
        cm = control_module
        if not cm._active:
            return

        x_input = args[0]    # unified: [B, image_seq + cap_seq, dim]
        freqs_cis = args[2]  # unified_freqs_cis: [B, image_seq + cap_seq, ...]
        adaln_input = args[3] if len(args) > 3 else None

        # Image tokens are FIRST in diffusers unified layout
        image_seq = cm._image_seq_len
        if image_seq <= 0:
            return

        # Extract image-only RoPE (first image_seq positions)
        x_freqs_cis = freqs_cis[:, :image_seq]

        # Lazy-embed control on first applicable block
        if cm._embedded_control is None and cm._control_latent is not None:
            cm._embedded_control = cm.embed_control(
                cm._control_latent.to(output.dtype).to(output.device),
                x_freqs_cis, adaln_input
            )

        if cm._embedded_control is None:
            return

        n_ctrl = cm.n_control_layers
        div = round(n_blocks / n_ctrl)
        cnet_idx = block_idx // div
        cnet_float = block_idx / div

        # Past all control blocks -- done
        if cnet_float > (n_ctrl - 1):
            cm._temp_data = None
            return

        # Initialize / re-initialize for main blocks
        if cm._temp_data is None or cm._temp_data[0] > cnet_idx:
            cm._temp_data = (-1, (None, cm._embedded_control))

        # Catch up: run skipped control blocks
        while cm._temp_data[0] < cnet_idx and (cm._temp_data[0] + 1) < n_ctrl:
            next_i = cm._temp_data[0] + 1
            c_ctx = cm._temp_data[1][1]
            # Image tokens are at the start [0, image_seq)
            c_skip, c_next = cm.forward_control_block(
                next_i, c_ctx,
                x_input[:, :c_ctx.shape[1]],
                attn_mask=None,
                freqs_cis=x_freqs_cis,
                adaln_input=adaln_input
            )
            cm._temp_data = (next_i, (c_skip, c_next))

        # Apply control at matching block boundaries
        td = cm._temp_data
        if td is not None and cnet_float == td[0]:
            c_skip = td[1][0]
            if c_skip is not None:
                # Add control signal to image portion (first positions)
                output[:, :c_skip.shape[1]] += c_skip * cm._control_scale
            if n_ctrl == td[0] + 1:
                cm._temp_data = None

    return hook


def _make_noise_refiner_hook(control_module, block_idx):
    """Forward hook for a noise refiner block (refiner_control mode).

    Noise refiner operates on image tokens only -- no offset needed.
    """
    def hook(module, args, output):
        cm = control_module
        if not cm._active or not cm.refiner_control:
            return

        x_input = args[0]    # image tokens only [B, image_seq, dim]
        freqs_cis = args[2]  # x_freqs_cis (image-only RoPE)
        adaln_input = args[3] if len(args) > 3 else None

        # Lazy-embed on first noise refiner block
        if block_idx == 0 and cm._embedded_control is None and cm._control_latent is not None:
            cm._embedded_control = cm.embed_control(
                cm._control_latent.to(output.dtype).to(output.device),
                freqs_cis, adaln_input
            )

        if cm._embedded_control is None:
            return

        # Initialize for noise refiner
        if block_idx == 0:
            cm._temp_data = (-1, (None, cm._embedded_control))

        c_ctx = cm._temp_data[1][1]
        c_skip, c_next = cm.forward_noise_refiner_block(
            block_idx, c_ctx,
            x_input[:, :c_ctx.shape[1]],
            attn_mask=None,
            freqs_cis=freqs_cis,
            adaln_input=adaln_input
        )
        cm._temp_data = (block_idx, (c_skip, c_next))

        if c_skip is not None:
            output[:, :c_skip.shape[1]] += c_skip * cm._control_scale

    return hook


def install_control_hooks(transformer, control_module):
    """Install control hooks on transformer. Non-destructive."""
    n_main = len(transformer.layers)
    hooks = []

    # Pre-hook: reset state each forward pass
    h = transformer.register_forward_pre_hook(_make_pre_hook(control_module))
    hooks.append(h)

    # Main block hooks
    for i, layer in enumerate(transformer.layers):
        h = layer.register_forward_hook(_make_main_block_hook(control_module, i, n_main))
        hooks.append(h)

    # Noise refiner hooks (refiner_control mode only)
    if control_module.refiner_control:
        for i, layer in enumerate(transformer.noise_refiner):
            h = layer.register_forward_hook(_make_noise_refiner_hook(control_module, i))
            hooks.append(h)

    transformer._control_hooks = hooks
    transformer._control_module = control_module

    refiner_info = f" + {len(transformer.noise_refiner)} refiner" if control_module.refiner_control else ""
    print(f"    Control hooks installed ({n_main} main{refiner_info} layers)")


def remove_control_hooks(transformer):
    """Remove all control hooks from transformer."""
    if hasattr(transformer, '_control_hooks'):
        for h in transformer._control_hooks:
            h.remove()
        del transformer._control_hooks

    if hasattr(transformer, '_control_module'):
        del transformer._control_module

    print("    Control hooks removed")
