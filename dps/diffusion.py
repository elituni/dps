"""Linear diffusion schedule and learned-range reverse variance."""

from vendor.guided_diffusion.gaussian_diffusion import (
    LossType,
    ModelMeanType,
    ModelVarType,
    get_named_beta_schedule,
)
from vendor.guided_diffusion.respace import SpacedDiffusion, space_timesteps


def create_diffusion(steps=1000):
    """Use the checkpoint's 1000-step schedule, optionally with respacing."""
    if isinstance(steps, bool) or not isinstance(steps, int) or not 2 <= steps <= 1000:
        raise ValueError("steps must be an integer between 2 and 1000.")
    return SpacedDiffusion(
        use_timesteps=space_timesteps(1000, [steps]),
        betas=get_named_beta_schedule("linear", 1000),
        model_mean_type=ModelMeanType.EPSILON,
        model_var_type=ModelVarType.LEARNED_RANGE,
        loss_type=LossType.MSE,
        rescale_timesteps=False,
    )
