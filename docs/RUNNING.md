# Running the example

From the repository directory, with Python 3.11 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m dps --image data/00002.png --checkpoint checkpoints/ffhq_10m.pt
```

Use your own image and the `ffhq_10m.pt` checkpoint linked in the [official DPS repository](https://github.com/DPS2022/diffusion-posterior-sampling#2-download-pretrained-checkpoint). Place it under `checkpoints/`; weights and input images aren't included in Git. Images are converted to RGB, resized to 256×256, and normalized to [-1, 1]. CUDA is used when available; a full run is slow on CPU, particularly with gradients through the UNet.

`--sigma-y` controls Gaussian measurement noise in normalized image units (default 0.05). `--guidance-scale` controls posterior guidance (default 0.3). `--steps` defaults to 1000; a smaller value, such as 100, respaces the original schedule and changes the sampling trajectory. `--seed`, `--device`, and `--output` are also available. Use a new output directory for each run. Outputs include three PNGs and settings with checkpoint/image hashes. Seeded results can still differ across devices or PyTorch versions.

The short `dps.ipynb` walkthrough imports the same modules and runs from this directory. Install Jupyter separately if you want to use it. Nothing installs packages or downloads files automatically.

## Sampling method

The clean-image estimate is the epsilon-parameterized Tweedie estimate:

```text
x0_hat = (x_t - sqrt(1 - alpha_bar_t) * epsilon_theta(x_t, t)) / sqrt(alpha_bar_t)
```

The diffusion utilities clip this estimate to [-1, 1], compute the DDPM posterior mean, and interpolate learned log variance between the posterior-variance and beta bounds. A reverse proposal is drawn from that distribution, with no random noise at the final step.

For measurement operator A and observation y, guidance uses the residual norm:

```text
r_t = ||y - A(x0_hat(x_t))||_2
x_previous = reverse_proposal - guidance_scale * grad_x_t(r_t)
```

This objective follows the Gaussian-noise `ps` path in the [official implementation](https://github.com/DPS2022/diffusion-posterior-sampling/blob/main/guided_diffusion/condition_methods.py). It normalizes the squared-residual gradient; the scale absorbs its magnitude rather than explicitly dividing by measurement-noise variance. Each image's norm is computed independently when batching. Autograd includes the denoiser derivative and blur's boundary behavior; no hand-written transpose approximation is needed. Graphs are discarded after each step.

The reverse proposal and learned-variance settings follow the [reference sampler](https://github.com/DPS2022/diffusion-posterior-sampling/blob/main/guided_diffusion/gaussian_diffusion.py). This implementation covers Gaussian deblurring, not the paper's full set of inverse problems. The retained blur has kernel size 61 and sigma 4.0 with reflection padding; the official deblurring config uses intensity 3.0. This is one reason to avoid claiming an exact experimental reproduction.

## Sources and changes

The project started from the ECE 498/598 MP3 template. Credits remain in the README; assignment instructions, grader gates, shell installs, download cells, duplicate scheduler setup, and global state have been removed from the code.

The original notebook used a residual push before the DDPM step, detached predicted noise, and fixed posterior variance. The current sampler instead differentiates through the denoiser, uses the learned-range variance, and applies guidance after the reverse proposal. These are substantive algorithm changes made during repository cleanup; the archived figure does not validate them. Full reconstruction quality and parameter tuning remain unverified.

The original archive stays outside the repository. `docs/archived_result.json` identifies the notebook and saved figure by SHA-256. The vendored inference files come from its guided-diffusion checkout, whose recorded commit is `22e0df8183507e13a7813f8d38d51b072ca1e67c`; unrelated training/evaluation files and nested Git history are omitted. The `nn.py` gradient-checkpoint helper uses PyTorch's non-reentrant implementation so gradients through frozen model weights work; other vendored files are unchanged. Original and current hashes are recorded in the metadata. Its existing license is retained. No new blanket license is assigned to the course template.

## Checks

```bash
python -m unittest discover -s tests -v
```

The CPU checks verify guidance with finite differences, reverse-update ordering, learned variance, respacing, and graph detachment. They do not establish face-reconstruction quality or reproduce paper metrics.
