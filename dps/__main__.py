"""Reconstruct a Gaussian-blurred image using a pretrained diffusion prior."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from .diffusion import create_diffusion
from .images import load_image, save_image
from .model import load_model
from .operators import GaussianBlur
from .sampling import sample_posterior


def sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs/example"))
    parser.add_argument("--sigma-y", type=float, default=0.05)
    parser.add_argument("--guidance-scale", type=float, default=0.3)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", choices=("cpu", "cuda"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()
    for name in ("sigma_y", "guidance_scale"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and nonnegative")
    if not 2 <= args.steps <= 1000:
        parser.error("--steps must be between 2 and 1000")
    if not args.image.is_file() or not args.checkpoint.is_file():
        parser.error("Provide existing image and checkpoint files.")

    diffusion = create_diffusion(args.steps)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    model = load_model(args.checkpoint, args.device)
    blur = GaussianBlur().to(args.device)
    clean = load_image(args.image, args.device)
    with torch.no_grad():
        observation = blur(clean) + args.sigma_y * torch.randn_like(clean)
    estimate = sample_posterior(model, diffusion, blur, observation, args.guidance_scale)

    images = {"ground-truth": clean, "observation": observation, "estimate": estimate}
    for name, tensor in images.items():
        save_image(tensor, args.output / f"{name}.png")
    settings = {
        "seed": args.seed,
        "sigma_y": args.sigma_y,
        "guidance_scale": args.guidance_scale,
        "device": args.device,
        "steps": args.steps,
        "base_steps": 1000,
        "blur_kernel_size": 61,
        "blur_sigma": 4.0,
        "checkpoint_sha256": sha256(args.checkpoint),
        "image_sha256": sha256(args.image),
        "torch_version": torch.__version__,
        "guidance_objective": "per-image residual L2 norm",
        "reverse_variance": "learned_range",
        "clip_denoised": True,
    }
    (args.output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    print(f"Saved images and settings to {args.output}")


if __name__ == "__main__":
    main()
