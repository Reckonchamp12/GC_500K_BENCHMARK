"""
GC-Bench · Deep Learning Architectures
=======================================
All neural architectures used across Tasks 1–4.

Deterministic forward / inverse models
---------------------------------------
  MLP           – Fully-connected with BatchNorm (Task 1/2/3/4)
  ResNet        – Residual blocks          (Task 1/2/3/4)  ← best inverse
  FTTransformer – Feature-Tokenizer Transformer (Task 1/2/3/4)
  MLPMixer      – Token-mixing MLP          (Task 1)
  NeuralODE     – Continuous-dynamics model (Task 1)
  CNN1D         – 1-D convolutional encoder (Task 4)

Spectral / Operator-learning forward models
--------------------------------------------
  UNet1D        – Skip-connection encoder-decoder ← best spectrum
  FNO1d         – Fourier Neural Operator
  DeepONet      – Branch-trunk operator network
  NeuralField   – Implicit neural representation

Physics-informed
-----------------
  PINN_Scalar   – ResNet + physics residual (Task 1)
  PINN_Spectral – ResNet with sigmoid output (Task 2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional Neural ODE
try:
    from torchdiffeq import odeint_adjoint as odeint
    HAS_ODE = True
except ImportError:
    HAS_ODE = False


# ── Shared building blocks ────────────────────────────────────────────────────

class MLP(nn.Module):
    """Fully-connected network with BatchNorm + GELU + Dropout."""

    def __init__(self, in_d: int, out_d: int, hidden=(256, 256, 256), drop: float = 0.1):
        super().__init__()
        layers, prev = [], in_d
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(drop)]
            prev = h
        layers.append(nn.Linear(prev, out_d))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResBlock(nn.Module):
    """Pre-activation residual block with BatchNorm."""

    def __init__(self, d: int, drop: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d), nn.BatchNorm1d(d), nn.GELU(),
            nn.Dropout(drop), nn.Linear(d, d), nn.BatchNorm1d(d),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class ResNet(nn.Module):
    """Stack of residual blocks with linear input/output projection."""

    def __init__(self, in_d: int, out_d: int, width: int = 256, n_blocks: int = 4, drop: float = 0.1):
        super().__init__()
        self.proj   = nn.Linear(in_d, width)
        self.blocks = nn.Sequential(*[ResBlock(width, drop) for _ in range(n_blocks)])
        self.head   = nn.Linear(width, out_d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.proj(x)))


# ── Transformer ───────────────────────────────────────────────────────────────

class FTTransformer(nn.Module):
    """
    Feature-Tokenizer Transformer.
    Each scalar input feature is projected to a d_model-dim token.
    A learned [CLS] token aggregates global context.
    """

    def __init__(self, in_d: int, out_d: int, d_model: int = 128,
                 nhead: int = 4, n_layers: int = 4, drop: float = 0.1):
        super().__init__()
        self.tok_emb = nn.Linear(1, d_model)
        enc_layer    = nn.TransformerEncoderLayer(
            d_model, nhead, d_model * 4, drop, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)
        self.cls     = nn.Parameter(torch.randn(1, 1, d_model))
        self.head    = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, out_d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        t = self.tok_emb(x.unsqueeze(-1))                         # (B, in_d, d_model)
        t = torch.cat([self.cls.expand(B, -1, -1), t], dim=1)     # prepend CLS
        return self.head(self.encoder(t)[:, 0])                    # use CLS output


# ── MLP-Mixer ─────────────────────────────────────────────────────────────────

class _MixerLayer(nn.Module):
    def __init__(self, n_patches: int, d_model: int, mlp_dim: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.tok = nn.Sequential(nn.Linear(n_patches, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, n_patches))
        self.ch  = nn.Sequential(nn.Linear(d_model,   mlp_dim), nn.GELU(), nn.Linear(mlp_dim, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.tok(self.ln1(x).transpose(1, 2)).transpose(1, 2)
        return x + self.ch(self.ln2(x))


class MLPMixer(nn.Module):
    """MLP-Mixer adapted for tabular input."""

    def __init__(self, in_d: int, out_d: int, n_patches: int = 8, d_model: int = 64, n_layers: int = 4):
        super().__init__()
        self.emb  = nn.Linear(in_d, n_patches * d_model)
        self.np_  = n_patches
        self.dm   = d_model
        self.layers = nn.Sequential(*[_MixerLayer(n_patches, d_model, d_model * 2) for _ in range(n_layers)])
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, out_d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        t = self.emb(x).view(B, self.np_, self.dm)
        return self.head(self.layers(t).mean(dim=1))


# ── Neural ODE ────────────────────────────────────────────────────────────────

class _ODEFunc(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 128), nn.Tanh(), nn.Linear(128, d))

    def forward(self, t, x):  # noqa: ARG002
        return self.net(x)


class NeuralODE(nn.Module):
    """Neural ODE with Euler fallback when torchdiffeq is unavailable."""

    def __init__(self, in_d: int, out_d: int, latent: int = 64):
        super().__init__()
        self.enc  = nn.Linear(in_d, latent)
        self.odef = _ODEFunc(latent)
        self.head = nn.Linear(latent, out_d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.enc(x)
        if HAS_ODE:
            z = odeint(self.odef, z, torch.tensor([0.0, 1.0], device=x.device), method="euler")[-1]
        else:
            # 4-step Euler integration
            for _ in range(4):
                z = z + 0.25 * self.odef(None, z)
        return self.head(z)


# ── 1-D CNN ───────────────────────────────────────────────────────────────────

class CNN1D(nn.Module):
    """
    1-D convolutional encoder for spectrum input.
    Accepts (B, L) or (B, 1, L) tensors.
    """

    def __init__(self, seq: int = 100, out_d: int = 5, ch: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, ch,     kernel_size=5, padding=2), nn.GELU(),
            nn.Conv1d(ch, ch*2,  kernel_size=5, padding=2), nn.GELU(),
            nn.Conv1d(ch*2, ch*4, kernel_size=3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(ch * 4, out_d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.head(self.conv(x).squeeze(-1))


# ── UNet1D (best forward-spectrum model) ──────────────────────────────────────

class UNet1D(nn.Module):
    """
    Lightweight UNet-style decoder with skip connection.
    Outperforms all other models on forward-spectrum: CosSim=0.9996.
    """

    def __init__(self, in_d: int = 5, out_d: int = 100, base: int = 64):
        super().__init__()
        self.enc  = nn.Linear(in_d, base * 4)
        self.dec  = nn.Sequential(
            nn.Linear(base * 4, 256), nn.GELU(),
            nn.Linear(256,      512), nn.GELU(),
            nn.Linear(512,      out_d),
        )
        self.skip = nn.Sequential(nn.Linear(in_d, 64), nn.GELU(), nn.Linear(64, out_d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dec(self.enc(x)) + 0.1 * self.skip(x)


# ── Fourier Neural Operator ───────────────────────────────────────────────────

class _SpectralConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, modes: int):
        super().__init__()
        self.modes = modes
        scale = 1.0 / (in_ch * out_ch)
        self.W = nn.Parameter(scale * torch.rand(in_ch, out_ch, modes, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, N = x.shape
        xf = torch.view_as_real(torch.fft.rfft(x))
        m = min(self.modes, xf.shape[-2])
        xr, xi = xf[..., :m, 0], xf[..., :m, 1]
        wr, wi = self.W[..., :m, 0], self.W[..., :m, 1]
        out_r = torch.einsum("bim,iom->bom", xr, wr) - torch.einsum("bim,iom->bom", xi, wi)
        out_i = torch.einsum("bim,iom->bom", xr, wi) + torch.einsum("bim,iom->bom", xi, wr)
        out_f = torch.zeros(B, self.W.shape[1], xf.shape[-2], 2, device=x.device)
        out_f[..., :m, :] = torch.stack([out_r, out_i], dim=-1)
        return torch.fft.irfft(torch.view_as_complex(out_f), n=N)


class FNO1d(nn.Module):
    """Fourier Neural Operator for 1-D spectral prediction."""

    def __init__(self, in_d: int = 5, out_d: int = 100, modes: int = 16, width: int = 32, n_layers: int = 4):
        super().__init__()
        self.lift = nn.Linear(in_d + 1, width)
        self.sp   = nn.ModuleList([_SpectralConv1d(width, width, modes) for _ in range(n_layers)])
        self.wc   = nn.ModuleList([nn.Conv1d(width, width, 1)           for _ in range(n_layers)])
        self.proj = nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Linear(128, 1))
        self.out_d = out_d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        grid = torch.linspace(0, 1, self.out_d, device=x.device)
        h = self.lift(
            torch.cat([
                x.unsqueeze(1).expand(-1, self.out_d, -1),
                grid.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1),
            ], dim=-1)
        ).permute(0, 2, 1)
        for s, w in zip(self.sp, self.wc):
            h = F.gelu(s(h) + w(h))
        return self.proj(h.permute(0, 2, 1)).squeeze(-1)


# ── DeepONet ──────────────────────────────────────────────────────────────────

class DeepONet(nn.Module):
    """
    Deep Operator Network (Lu et al., 2021).
    branch(geometry) · trunk(sensor_coords) + bias.
    """

    def __init__(self, br_in: int = 5, tr_in: int = 1, out_d: int = 100, hid: int = 128, p: int = 64):
        super().__init__()
        self.branch = MLP(br_in, p, (hid, hid))
        self.trunk  = MLP(tr_in, p, (hid, hid))
        self.bias   = nn.Parameter(torch.zeros(out_d))
        self.out_d  = out_d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b       = self.branch(x)
        sensors = torch.linspace(0, 1, self.out_d, device=x.device)
        t       = self.trunk(sensors.unsqueeze(-1))
        return b @ t.T + self.bias


# ── Neural Field ──────────────────────────────────────────────────────────────

class NeuralField(nn.Module):
    """
    Implicit neural representation conditioned on geometry.
    Queries transmittance at arbitrary wavelength coordinates.
    """

    def __init__(self, cond_d: int = 5, hid: int = 128, n_layers: int = 4, out_d: int = 100):
        super().__init__()
        in_d = cond_d + 1
        layers = [nn.Linear(in_d, hid), nn.GELU()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hid, hid), nn.GELU()]
        layers.append(nn.Linear(hid, 1))
        self.net   = nn.Sequential(*layers)
        self.out_d = out_d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B      = x.shape[0]
        coords = torch.linspace(0, 1, self.out_d, device=x.device)
        xe     = x.unsqueeze(1).expand(-1, self.out_d, -1)
        ce     = coords.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1)
        inp    = torch.cat([xe, ce], dim=-1).view(-1, x.shape[1] + 1)
        return torch.sigmoid(self.net(inp).view(B, self.out_d))


# ── Physics-Informed Networks ─────────────────────────────────────────────────

def physics_forward_torch(xr: torch.Tensor) -> torch.Tensor:
    """
    Differentiable analytical approximation of the grating coupler response.
    Used as a regularisation term in PINN training.
    """
    p, ff, etch, ox, tsi = xr.unbind(-1)
    n_si  = 3.48
    nslab = n_si * (1 - 0.2 * torch.exp(-tsi / 150))
    ng    = n_si * ff + 1.0 * (1 - ff)
    fetch = 1 - 0.5 * (etch / tsi)
    ncomb = nslab * fetch + ng * (1 - fetch)
    fox   = 1 - 0.3 * torch.exp(-ox / 1000)
    neff  = ncomb * fox
    return torch.stack([
        p * neff,
        30 + 20 * (1 - ff) + 10 * (etch / 100),
        neff,
        0.8 * (1 - 0.1 * (etch / tsi)),
    ], dim=-1)


class PINN_Scalar(nn.Module):
    """ResNet backbone with physics residual loss for scalar prediction."""

    def __init__(self, in_d: int = 5, out_d: int = 4, width: int = 256, n_blocks: int = 4):
        super().__init__()
        self.net = ResNet(in_d, out_d, width, n_blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PINN_Spectral(nn.Module):
    """ResNet backbone with sigmoid output for spectral prediction."""

    def __init__(self, in_d: int = 5, out_d: int = 100, width: int = 512, n_blocks: int = 6):
        super().__init__()
        self.net = ResNet(in_d, out_d, width, n_blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))
