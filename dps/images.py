"""RGB image conversion between uint8 pixels and normalized tensors."""

import numpy as np
from PIL import Image
import torch


def load_image(path, device="cpu"):
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != (256, 256):
            image = image.resize((256, 256), Image.Resampling.LANCZOS)
        array = np.array(image, dtype=np.float32) / 255.0
    return (torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0) * 2 - 1).to(device)


def save_image(tensor, path):
    array = ((tensor[0].detach().cpu().clamp(-1, 1) + 1) / 2)
    array = (array.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    Image.fromarray(array).save(path)
