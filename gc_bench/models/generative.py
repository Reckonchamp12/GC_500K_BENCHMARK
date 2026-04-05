"""
GC-Bench · Generative / Probabilistic Models
=============================================
All models designed to handle the non-uniqueness of inverse design.

MDN      – Mixture Density Network (10 Gaussian components)
CVAE     – Conditional Variational Autoencoder (β-VAE variant)
RealNVP  – Normalizing Flow (8 affine coupling layers)
INN      – Invertible Neural Network (padded affine blocks)
DDPM     – Denoising Diffusion Probabilistic Model (T=200)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist_

from .deep import MLP


# ── Mixture Density Network ───────────────────────────────────────────────────

class MDN(nn.Module):
    """
    Mixture Density Network outputting a Gaussian mixture over geometry space.
    Handles multi-modal posteriors by learning K mixture components.

    Parameters
    ----------
    n_mix : number of Gaussian components (default 10)
    """

    def __init__(self, in_d: int, out_d: int, n_mix: int = 10, hid=(256, 256)):
        super().__init__()
        self.n_mix = n_mix
        self.out_d = out_d
        self.base  = MLP(in_d, 256, hid)
        self.pi    = nn.Linear(256, n_mix)
        self.mu    = nn.Linear(256, n_mix * out_d)
        self.sig   = nn.Linear(256, n_mix * out_d)

    def forward(self, x: torch.Tensor):
        h   = self.base(x)
        pi  = F.softmax(self.pi(h), dim=-1)
        mu  = self.mu(h).view(-1, self.n_mix, self.out_d)
        sig = F.softplus(self.sig(h)).view(-1, self.n_mix, self.out_d) + 1e-3
        return pi, mu, sig

    @torch.no_grad()
    def sample(self, x: torch.Tensor) -> torch.Tensor:
        """Sample one geometry per input condition."""
        pi, mu, sig = self.forward(x)
        B   = x.shape[0]
        idx = torch.multinomial(pi, 1).squeeze(-1)
        mu_s  = mu[torch.arange(B), idx]
        sig_s = sig[torch.arange(B), idx]
        return mu_s + sig_s * torch.randn_like(mu_s)

    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood of the mixture."""
        pi, mu, sig = self.forward(x)
        y_exp = y.unsqueeze(1).expand_as(mu)
        lp    = dist_.Normal(mu, sig).log_prob(y_exp).sum(-1) + torch.log(pi + 1e-8)
        return -torch.logsumexp(lp, dim=-1).mean()


# ── Conditional VAE ───────────────────────────────────────────────────────────

class CVAE(nn.Module):
    """
    Conditional Variational Autoencoder.
    Encodes geometry conditioned on optical targets; decodes from latent + condition.

    Parameters
    ----------
    cond_d : dimension of conditioning input (scalars or spectrum)
    x_d    : dimension of output to reconstruct (geometry)
    latent : latent space dimension
    beta   : KL-divergence weight (set in cvae_loss, default 0.5)
    """

    def __init__(self, cond_d: int, x_d: int, latent: int = 32, hid: int = 256):
        super().__init__()
        self.latent = latent
        self.enc    = nn.Sequential(
            nn.Linear(x_d + cond_d, hid), nn.GELU(),
            nn.Linear(hid, hid),          nn.GELU(),
        )
        self.mu_h = nn.Linear(hid, latent)
        self.lv_h = nn.Linear(hid, latent)
        self.dec  = nn.Sequential(
            nn.Linear(latent + cond_d, hid), nn.GELU(),
            nn.Linear(hid, hid),             nn.GELU(),
            nn.Linear(hid, x_d),
        )

    def encode(self, x, c):
        h = self.enc(torch.cat([x, c], dim=-1))
        return self.mu_h(h), self.lv_h(h)

    def reparam(self, mu, lv):
        return mu + torch.exp(0.5 * lv) * torch.randn_like(mu)

    def decode(self, z, c):
        return torch.sigmoid(self.dec(torch.cat([z, c], dim=-1)))

    def forward(self, x, c):
        mu, lv = self.encode(x, c)
        z      = self.reparam(mu, lv)
        return self.decode(z, c), mu, lv

    @torch.no_grad()
    def sample(self, c: torch.Tensor) -> torch.Tensor:
        return self.decode(torch.randn(c.shape[0], self.latent, device=c.device), c)


def cvae_loss(recon, x, mu, lv, beta: float = 0.5):
    return F.mse_loss(recon, x) + beta * (-0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp()))


# ── Normalizing Flow (RealNVP) ────────────────────────────────────────────────

class _AffineCoupling(nn.Module):
    def __init__(self, d: int, hid: int = 128):
        super().__init__()
        half   = d // 2
        self.s = MLP(d - half, half, (hid, hid))
        self.t = MLP(d - half, half, (hid, hid))

    def forward(self, x: torch.Tensor, rev: bool = False):
        x1, x2 = x.chunk(2, dim=-1)
        s, t   = torch.tanh(self.s(x1)), self.t(x1)
        if not rev:
            y2   = x2 * torch.exp(s) + t
            logdet = s.sum(-1)
        else:
            y2      = (x2 - t) * torch.exp(-s)
            logdet  = None
        return torch.cat([x1, y2], dim=-1), logdet


class RealNVP(nn.Module):
    """
    Conditional RealNVP normalizing flow.
    Transforms a Gaussian prior into the geometry posterior conditioned on optical targets.
    """

    def __init__(self, d: int, cond_d: int, n: int = 8, hid: int = 128):
        super().__init__()
        self.d     = d
        self.cp    = nn.Linear(cond_d, d)
        self.flows = nn.ModuleList([_AffineCoupling(d, hid) for _ in range(n)])

    def log_prob(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x  = x + 0.01 * self.cp(c)
        ld = torch.zeros(x.shape[0], device=x.device)
        for f in self.flows:
            x, l = f(x)
            ld  += l
        return dist_.Normal(0, 1).log_prob(x).sum(-1) + ld

    @torch.no_grad()
    def sample(self, c: torch.Tensor) -> torch.Tensor:
        z = torch.randn(c.shape[0], self.d, device=c.device)
        for f in reversed(self.flows):
            z, _ = f(z, rev=True)
        return torch.sigmoid(z - 0.01 * self.cp(c))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return -self.log_prob(x, c).mean()


# ── Invertible Neural Network ─────────────────────────────────────────────────

class _INNBlock(nn.Module):
    def __init__(self, d: int, hid: int = 128):
        super().__init__()
        half   = d // 2
        self.s = nn.Sequential(nn.Linear(half, hid), nn.GELU(), nn.Linear(hid, d - half), nn.Tanh())
        self.t = nn.Sequential(nn.Linear(half, hid), nn.GELU(), nn.Linear(hid, d - half))

    def forward(self, x: torch.Tensor, rev: bool = False) -> torch.Tensor:
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        s, t   = self.s(x1), self.t(x1)
        y2     = x2 * torch.exp(s) + t if not rev else (x2 - t) * torch.exp(-s)
        return torch.cat([x1, y2], dim=-1)


class INN(nn.Module):
    """
    Invertible Neural Network using affine coupling blocks.
    The input is padded to a larger dimension for expressivity.
    """

    def __init__(self, in_d: int = 100, out_d: int = 5, n_blocks: int = 6):
        super().__init__()
        self.pad_d    = max(in_d, out_d * 4)
        self.in_proj  = nn.Linear(in_d,       self.pad_d)
        self.out_proj = nn.Linear(self.pad_d,  out_d)
        self.blocks   = nn.ModuleList([_INNBlock(self.pad_d) for _ in range(n_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        for b in self.blocks:
            h = b(h)
        return self.out_proj(h)


# ── DDPM Diffusion Model ──────────────────────────────────────────────────────

class _DiffUNet(nn.Module):
    """Time- and condition-aware noise predictor for DDPM."""

    def __init__(self, d: int, cond_d: int, hid: int = 256, n_layers: int = 4):
        super().__init__()
        self.te = nn.Sequential(nn.Linear(1,       64), nn.GELU(), nn.Linear(64, 64))
        self.ce = nn.Sequential(nn.Linear(cond_d,  64), nn.GELU(), nn.Linear(64, 64))
        layers  = [nn.Linear(d + 128, hid), nn.GELU()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hid, hid), nn.GELU()]
        layers.append(nn.Linear(hid, d))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        te = self.te(t.float().view(-1, 1) / 200)
        ce = self.ce(c)
        return self.net(torch.cat([x, te, ce], dim=-1))


class DDPM(nn.Module):
    """
    Denoising Diffusion Probabilistic Model for conditional geometry sampling.
    Geometry is treated as the 'image'; optical targets are the conditioning signal.
    """

    def __init__(self, d: int, cond_d: int, T: int = 200):
        super().__init__()
        self.T   = T
        self.net = _DiffUNet(d, cond_d)
        betas     = torch.linspace(1e-4, 0.02, T)
        alphas    = 1 - betas
        self.register_buffer("betas",      betas)
        self.register_buffer("alphas",     alphas)
        self.register_buffer("alpha_bars", torch.cumprod(alphas, 0))

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        ab = self.alpha_bars[t].view(-1, 1)
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise, noise

    def p_loss(self, x0, c):
        t       = torch.randint(0, self.T, (x0.shape[0],), device=x0.device)
        xt, eps = self.q_sample(x0, t)
        return F.mse_loss(self.net(xt, t, c), eps)

    def forward(self, x0, c):
        return self.p_loss(x0, c)

    @torch.no_grad()
    def sample(self, c: torch.Tensor, shape: tuple) -> torch.Tensor:
        x = torch.randn(*shape, device=c.device)
        for t_ in reversed(range(self.T)):
            t  = torch.full((shape[0],), t_, device=c.device, dtype=torch.long)
            pn = self.net(x, t, c)
            x  = (x - (1 - self.alphas[t_]) / (1 - self.alpha_bars[t_]).sqrt() * pn) / self.alphas[t_].sqrt()
            if t_ > 0:
                x += self.betas[t_].sqrt() * torch.randn_like(x)
        return torch.sigmoid(x)
