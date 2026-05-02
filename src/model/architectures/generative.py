"""Generative AI: VAE, GAN, DDPM, DDIM, Score-SDE, LDM, Normalizing Flow, WaveNet.

Papers: Kingma 2013, Goodfellow 2014, Ho 2020, Song 2020, Rombach 2021, Dinh 2016, van den Oord 2016.
"""

from __future__ import annotations

import math
import random

from .registry import register


class VAE:
    """Variational Autoencoder (Kingma & Welling 2013)."""

    def __init__(self, input_dim: int = 784, latent_dim: int = 20) -> None:
        s = 1.0 / math.sqrt(input_dim)
        self.enc_mu = [[random.gauss(0, s) for _ in range(input_dim)] for _ in range(latent_dim)]
        self.enc_logvar = [
            [random.gauss(0, s) for _ in range(input_dim)] for _ in range(latent_dim)
        ]
        self.dec = [[random.gauss(0, 0.1) for _ in range(latent_dim)] for _ in range(input_dim)]

    def encode(self, x: list[float]) -> tuple[list[float], list[float]]:
        mu = [sum(self.enc_mu[i][j] * x[j] for j in range(len(x))) for i in range(len(self.enc_mu))]
        logvar = [
            sum(self.enc_logvar[i][j] * x[j] for j in range(len(x)))
            for i in range(len(self.enc_logvar))
        ]
        return mu, logvar

    def reparameterize(self, mu: list[float], logvar: list[float]) -> list[float]:
        return [mu[i] + math.exp(logvar[i] * 0.5) * random.gauss(0, 1) for i in range(len(mu))]

    def decode(self, z: list[float]) -> list[float]:
        return [sum(self.dec[i][j] * z[j] for j in range(len(z))) for i in range(len(self.dec))]

    def forward(self, x: list[float]) -> tuple[list[float], float]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        kl = -0.5 * sum(1 + lv - m**2 - math.exp(lv) for m, lv in zip(mu, logvar, strict=True))
        return recon, kl


register("generative.vae", VAE)


class GAN:
    """Generative Adversarial Network (Goodfellow et al. 2014)."""

    def __init__(self, latent_dim: int = 100, data_dim: int = 784) -> None:
        s_gen = 1.0 / math.sqrt(latent_dim)
        s_dis = 1.0 / math.sqrt(data_dim)
        self.G = [[random.gauss(0, s_gen) for _ in range(latent_dim)] for _ in range(data_dim)]
        self.D = [[random.gauss(0, s_dis) for _ in range(data_dim)] for _ in range(1)]

    def generate(self, z: list[float]) -> list[float]:
        return [sum(self.G[i][j] * z[j] for j in range(len(z))) for i in range(len(self.G))]

    def discriminate(self, x: list[float]) -> float:
        return 1.0 / (1.0 + math.exp(-sum(self.D[0][j] * x[j] for j in range(len(x)))))


register("generative.gan", GAN)


class DDPM:
    """Denoising Diffusion Probabilistic Model (Ho, Jain, Abbeel 2020)."""

    def __init__(self, data_dim: int = 784, n_steps: int = 1000) -> None:
        self.n_steps = n_steps
        self.betas = [1e-4 + (i / n_steps) * (0.02 - 1e-4) for i in range(n_steps)]
        self.alphas = [1.0 - b for b in self.betas]
        self.alpha_bar = [math.prod(self.alphas[: i + 1]) for i in range(n_steps)]
        s = 1.0 / math.sqrt(data_dim)
        self.denoise = [[random.gauss(0, s) for _ in range(data_dim + 1)] for _ in range(data_dim)]

    def forward_diffuse(self, x0: list[float], t: int) -> tuple[list[float], list[float]]:
        noise = [random.gauss(0, 1) for _ in range(len(x0))]
        xt = [
            math.sqrt(self.alpha_bar[t]) * x0[i] + math.sqrt(1.0 - self.alpha_bar[t]) * noise[i]
            for i in range(len(x0))
        ]
        return xt, noise

    def reverse_step(self, xt: list[float], t: int) -> list[float]:
        t_norm = t / self.n_steps
        input_vec = xt + [t_norm]
        predicted_noise = [
            sum(self.denoise[i][j] * input_vec[j] for j in range(len(input_vec)))
            for i in range(len(xt))
        ]
        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alpha_bar[t]
        coef1 = 1.0 / math.sqrt(alpha_t)
        coef2 = beta_t / math.sqrt(1.0 - alpha_bar_t)
        return [
            coef1 * (xt[i] - coef2 * predicted_noise[i]) + math.sqrt(beta_t) * random.gauss(0, 1)
            for i in range(len(xt))
        ]

    def sample(self, n_tokens: int = 784) -> list[float]:
        x = [random.gauss(0, 1) for _ in range(n_tokens)]
        for t in reversed(range(self.n_steps)):
            x = self.reverse_step(x, t)
        return x


register("generative.ddpm", DDPM)


class LatentDiffusion:
    """Latent Diffusion Model (Rombach et al. 2022). Diffusion in latent space."""

    def __init__(self, latent_dim: int = 64, data_dim: int = 784) -> None:
        self.vae = VAE(data_dim, latent_dim)
        self.diffusion = DDPM(latent_dim, n_steps=50)

    def sample(self, latent_shape: int = 64) -> list[float]:
        latent = self.diffusion.sample(latent_shape)
        return self.vae.decode(latent)


register("generative.ldm", LatentDiffusion)


class NormalizingFlow:
    """Real NVP Normalizing Flow (Dinh, Sohl-Dickstein, Bengio 2016)."""

    def __init__(self, dim: int = 2, n_layers: int = 4) -> None:
        self.dim = dim
        self.n_layers = n_layers
        s = 1.0 / math.sqrt(dim)
        self.scales = [[random.gauss(0, s) for _ in range(dim // 2)] for _ in range(n_layers)]
        self.translations = [[random.gauss(0, s) for _ in range(dim // 2)] for _ in range(n_layers)]

    def forward(self, x: list[float]) -> tuple[list[float], float]:
        log_det = 0.0
        for layer in range(self.n_layers):
            d = len(x) // 2
            x1, x2 = list(x[:d]), list(x[d:])
            s = [sum(self.scales[layer][j] * x1[j] for j in range(d)) for _ in range(d)]
            t = [sum(self.translations[layer][j] * x1[j] for j in range(d)) for _ in range(d)]
            x2 = [x2[i] * math.exp(s[i]) + t[i] for i in range(d)]
            log_det += sum(s)
            x = x1 + x2
        return x, log_det


register("generative.flow", NormalizingFlow)
