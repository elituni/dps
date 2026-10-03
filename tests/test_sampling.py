import unittest

import numpy as np
import torch

from dps.diffusion import create_diffusion
from dps.operators import GaussianBlur
from dps.sampling import measurement_gradient, posterior_step, sample_posterior


class ToyDenoiser(torch.nn.Module):
    def __init__(self, variance=0.0):
        super().__init__()
        self.variance = variance
        self.timesteps = []

    def forward(self, x, t):
        self.timesteps.extend(t.tolist())
        return torch.cat((0.2 * x, torch.full_like(x, self.variance)), dim=1)


class SamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_guidance_includes_denoiser_derivative(self):
        x = torch.tensor([[[[0.2, -0.1], [0.3, 0.1]]]], dtype=torch.float64,
                         requires_grad=True)
        target = torch.full_like(x, 0.1)
        alpha = torch.tensor(0.6, dtype=x.dtype)

        def clean(state):
            epsilon = 0.2 * state.sin()
            return (state - (1 - alpha).sqrt() * epsilon) / alpha.sqrt()

        gradient, _ = measurement_gradient(x, clean(x), target, lambda z: 2 * z)
        delta = 1e-6
        numerical = torch.zeros_like(x)
        for index in range(x.numel()):
            plus, minus = x.detach().clone(), x.detach().clone()
            plus.flatten()[index] += delta
            minus.flatten()[index] -= delta
            numerical.flatten()[index] = (
                torch.linalg.vector_norm(2 * clean(plus) - target)
                - torch.linalg.vector_norm(2 * clean(minus) - target)
            ) / (2 * delta)
        torch.testing.assert_close(gradient, numerical, rtol=1e-6, atol=1e-8)

    def test_zero_residual_has_finite_zero_gradient(self):
        x = torch.ones(1, 1, 2, 2, requires_grad=True)
        gradient, distances = measurement_gradient(x, 2 * x, 2 * x.detach(), lambda z: z)
        self.assertEqual(distances.item(), 0)
        torch.testing.assert_close(gradient, torch.zeros_like(x))

    def test_batch_guidance_matches_individual_images(self):
        x = torch.randn(2, 1, 3, 3, requires_grad=True)
        gradient, _ = measurement_gradient(x, x.square(), torch.zeros_like(x), lambda z: z)
        for i in range(2):
            single = x[i:i + 1].detach().requires_grad_(True)
            expected, _ = measurement_gradient(single, single.square(), torch.zeros_like(single), lambda z: z)
            torch.testing.assert_close(gradient[i:i + 1], expected)

    def test_blur_gradient_matches_finite_difference_at_boundary(self):
        blur = GaussianBlur(ksize=3, sigma=1).double()
        x = torch.randn(1, 1, 4, 4, dtype=torch.float64, requires_grad=True)
        loss = blur(x).square().sum()
        gradient = torch.autograd.grad(loss, x)[0]
        delta = 1e-6
        plus, minus = x.detach().clone(), x.detach().clone()
        plus[0, 0, 0, 0] += delta
        minus[0, 0, 0, 0] -= delta
        numerical = (blur(plus).square().sum() - blur(minus).square().sum()) / (2 * delta)
        torch.testing.assert_close(gradient[0, 0, 0, 0], numerical)

    def test_learned_variance_uses_both_bounds(self):
        diffusion = create_diffusion()
        x = torch.zeros(1, 1, 2, 2)
        t = torch.tensor([500])
        for value, expected in ((-1, diffusion.posterior_variance[500]),
                                (1, diffusion.betas[500])):
            prediction = diffusion.p_mean_variance(ToyDenoiser(value), x, t)
            torch.testing.assert_close(prediction['variance'], torch.full_like(x, expected))

    def test_guidance_corrects_reverse_proposal(self):
        diffusion = create_diffusion()
        model = ToyDenoiser()
        x = torch.full((1, 1, 2, 2), 0.1)
        y = torch.zeros_like(x)
        state = x.detach().requires_grad_(True)
        prediction = diffusion.p_mean_variance(model, state, torch.tensor([200]))
        gradient, _ = measurement_gradient(state, prediction['pred_xstart'], y, lambda z: z)
        torch.manual_seed(12)
        proposal = prediction['mean'].detach() + prediction['log_variance'].detach().mul(0.5).exp() * torch.randn_like(x)
        torch.manual_seed(12)
        actual, _ = posterior_step(model, diffusion, x, 200, y, lambda z: z, 0.3)
        torch.testing.assert_close(actual, proposal - 0.3 * gradient)
        self.assertFalse(actual.requires_grad)

    def test_last_step_has_no_sampling_noise(self):
        diffusion = create_diffusion()
        x = torch.full((1, 1, 2, 2), 0.1)
        model = ToyDenoiser()
        first, _ = posterior_step(model, diffusion, x, 0, torch.zeros_like(x), lambda z: z)
        torch.manual_seed(100)
        second, _ = posterior_step(model, diffusion, x, 0, torch.zeros_like(x), lambda z: z)
        torch.testing.assert_close(first, second)

    def test_respacing_maps_model_to_original_timesteps(self):
        diffusion = create_diffusion(4)
        original = create_diffusion()
        np.testing.assert_allclose(diffusion.alphas_cumprod,
                                   original.alphas_cumprod[diffusion.timestep_map])
        model = ToyDenoiser()
        diffusion.p_mean_variance(model, torch.zeros(1, 1, 2, 2), torch.tensor([3]))
        self.assertEqual(model.timesteps, [999])

    def test_short_chain_is_finite_and_discards_graphs(self):
        diffusion = create_diffusion(4)
        estimate = sample_posterior(ToyDenoiser(), diffusion, lambda z: z,
                                    torch.zeros(1, 1, 4, 4), progress=False)
        self.assertTrue(torch.isfinite(estimate).all())
        self.assertFalse(estimate.requires_grad)
        self.assertIsNone(estimate.grad_fn)

    def test_activation_checkpoint_supports_frozen_weights(self):
        create_diffusion()
        from vendor.guided_diffusion.nn import checkpoint

        layer = torch.nn.Linear(3, 2).requires_grad_(False)
        x = torch.randn(1, 3, requires_grad=True)
        output = checkpoint(layer, (x,), layer.parameters(), True)
        gradient = torch.autograd.grad(output.sum(), x)[0]
        torch.testing.assert_close(gradient, layer.weight.sum(dim=0).unsqueeze(0))
        self.assertTrue(all(parameter.grad is None for parameter in layer.parameters()))


if __name__ == '__main__':
    unittest.main()
