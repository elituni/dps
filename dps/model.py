"""FFHQ diffusion prior with checkpoint-compatible UNet settings."""

import torch

from vendor.guided_diffusion.script_util import create_model


def load_model(checkpoint, device):
    model = create_model(
        image_size=256,
        num_channels=128,
        num_res_blocks=1,
        channel_mult="1,1,2,2,4,4",
        learn_sigma=True,
        class_cond=False,
        use_checkpoint=False,
        attention_resolutions="16",
        num_heads=4,
        num_head_channels=64,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        dropout=0.0,
        resblock_updown=True,
        use_fp16=False,
        use_new_attention_order=False,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict):
        for key in ("model", "state_dict", "ema", "params", "net"):
            if isinstance(state.get(key), dict):
                state = state[key]
                break
    clean = {}
    for key, value in state.items():
        if key.startswith("module."):
            key = key[len("module."):]
        if key.startswith("model."):
            key = key[len("model."):]
        clean[key] = value
    model.load_state_dict(clean, strict=True)
    model.requires_grad_(False)
    return model.to(device).eval()
