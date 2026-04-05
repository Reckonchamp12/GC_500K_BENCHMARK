"""
GC-Bench · Configuration
========================
Central dataclass holding all hyperparameters.
Override via YAML (configs/default.yaml) or CLI flags.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class CFG:
    # ── Data ──────────────────────────────────────────────────────────────
    data_path: str = "/kaggle/input/datasets/drahulray/syn-data/gc_500k_20251213_152525.h5"
    seed: int = 42

    # ── Training ──────────────────────────────────────────────────────────
    batch_size: int = 4096
    dl_epochs: int = 60
    lr: float = 1e-3
    wd: float = 1e-4
    patience: int = 10

    # ── Model-specific ────────────────────────────────────────────────────
    gp_subset: int = 3000      # GP trains on a random subset (memory)
    ensemble_n: int = 5         # Number of members in DeepEnsemble
    diff_steps: int = 200       # DDPM diffusion timesteps
    flow_layers: int = 8        # RealNVP coupling layers
    fid_dim: int = 64           # Feature dim for FID-style metrics

    # ── Evaluation ────────────────────────────────────────────────────────
    dtw_sub: int = 10           # Sub-sampling factor for DTW (speed)
    noise_sigmas: List[float] = field(
        default_factory=lambda: [0.001, 0.005, 0.01, 0.02, 0.05]
    )

    # ── Schema ────────────────────────────────────────────────────────────
    param_names: List[str] = field(default_factory=lambda: [
        "period_nm",
        "fill_factor",
        "etch_depth_nm",
        "oxide_thickness_nm",
        "si_thickness_nm",
    ])
    scalar_names: List[str] = field(default_factory=lambda: [
        "lambda_center_nm",
        "bandwidth_um",
        "n_eff",
        "peak_transmission",
    ])

    # ── Checkpointing ─────────────────────────────────────────────────────
    force_retrain: bool = True   # False → load existing checkpoints
    out_dir: str = "/kaggle/working/gc_bench"
