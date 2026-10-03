"""DDPM reverse sampling with measurement guidance through the denoiser."""

import math

import torch
from tqdm.auto import tqdm


def measurement_gradient(x_t, x0_hat, measurement, operator):
    """Differentiate each image's residual norm through the clean estimate."""
    residual = operator(x0_hat) - measurement
    distances = torch.linalg.vector_norm(residual.flatten(1), dim=1)
    gradient = torch.autograd.grad(distances.sum(), x_t)[0]
    return gradient, distances.detach()


@torch.enable_grad()
def posterior_step(model, diffusion, x_t, t_idx, measurement, operator,
                   guidance_scale=0.3):
    """Draw a reverse proposal, then correct it using the current-state score."""
    state = x_t.detach().requires_grad_(True)
    timesteps = torch.full((state.shape[0],), t_idx,
                           device=state.device, dtype=torch.long)
    prediction = diffusion.p_mean_variance(model, state, timesteps,
                                          clip_denoised=True)
    gradient, distances = measurement_gradient(
        state, prediction["pred_xstart"], measurement, operator,
    )
    with torch.no_grad():
        proposal = prediction["mean"]
        if t_idx > 0:
            proposal = (proposal + torch.exp(0.5 * prediction["log_variance"])
                        * torch.randn_like(state))
        sample = proposal - guidance_scale * gradient
    return sample, distances


def sample_posterior(model, diffusion, operator, measurement,
                     guidance_scale=0.3, progress=True):
    """Reconstruct from noise while retaining one step's graph at a time."""
    if not math.isfinite(guidance_scale) or guidance_scale < 0:
        raise ValueError("guidance_scale must be finite and nonnegative.")
    measurement = measurement.detach()
    state = torch.randn_like(measurement)
    steps = tqdm(range(diffusion.num_timesteps - 1, -1, -1),
                 disable=not progress, desc="DPS")
    for t_idx in steps:
        state, distances = posterior_step(
            model, diffusion, state, t_idx, measurement, operator, guidance_scale,
        )
        if progress:
            steps.set_postfix(residual=f"{distances.mean().item():.3f}")
    return state
