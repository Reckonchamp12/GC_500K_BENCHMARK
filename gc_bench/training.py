"""
GC-Bench · Training Utilities
==============================
Generic PyTorch training loop, DataLoader factory, multi-GPU wrapping,
and batched prediction — all shared across all four benchmarks.
"""

import time
import gc
from copy import deepcopy
from pathlib import Path
from typing import Callable, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .config import CFG

# ── Device setup ──────────────────────────────────────────────────────────────

N_GPUS = torch.cuda.device_count()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def print_device_info() -> None:
    print(f"GPUs: {N_GPUS}  device: {DEVICE}")
    for i in range(N_GPUS):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")


def wrap(model: nn.Module) -> nn.Module:
    """Move model to device; use DataParallel if multiple GPUs are available."""
    return nn.DataParallel(model.to(DEVICE)) if N_GPUS > 1 else model.to(DEVICE)


# ── DataLoader factory ────────────────────────────────────────────────────────

def make_loader(
    X: np.ndarray,
    Y: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(Y, dtype=torch.float32),
    )
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=2, pin_memory=True, drop_last=False,
    )


# ── Generic training loop ─────────────────────────────────────────────────────

def train_torch(
    model: nn.Module,
    ldr_tr: DataLoader,
    ldr_va: DataLoader,
    loss_fn: Callable = nn.MSELoss(),
    epochs: Optional[int] = None,
    patience: Optional[int] = None,
    name: str = "model",
    ckpt: Optional[Path] = None,
    force: bool = True,
    cfg: Optional[CFG] = None,
) -> Tuple[nn.Module, List[float], List[float]]:
    """
    Train `model` with early stopping.

    Parameters
    ----------
    model   : wrapped (possibly DataParallel) nn.Module
    ldr_tr  : training DataLoader
    ldr_va  : validation DataLoader
    loss_fn : loss callable
    epochs  : max epochs (defaults to cfg.dl_epochs)
    patience: early-stop patience (defaults to cfg.patience)
    name    : display name for tqdm
    ckpt    : path to save/load checkpoint
    force   : if False and ckpt exists, skip training and load
    cfg     : CFG instance for default values

    Returns
    -------
    (model, train_losses, val_losses)
    """
    _epochs  = epochs   or (cfg.dl_epochs if cfg else 60)
    _patience = patience or (cfg.patience if cfg else 10)
    _batch   = cfg.batch_size if cfg else 4096

    if not force and ckpt and ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        print(f"    ↳ loaded checkpoint: {ckpt.name}")
        return model, [], []

    lr = cfg.lr if cfg else 1e-3
    wd = cfg.wd if cfg else 1e-4

    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=_epochs)

    best_val, best_state, wait = 1e9, None, 0
    htr: List[float] = []
    hva: List[float] = []

    pbar = tqdm(range(1, _epochs + 1), desc=f"  {name}", leave=False)
    for ep in pbar:
        # ── Train ──────────────────────────────────────────────────────
        model.train()
        tl = 0.0
        for xb, yb in ldr_tr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            if isinstance(out, tuple):
                out = out[0]
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            tl += loss.item() * len(xb)
        tl /= len(ldr_tr.dataset)

        # ── Validate ───────────────────────────────────────────────────
        model.eval()
        vl = 0.0
        with torch.no_grad():
            for xb, yb in ldr_va:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                out = model(xb)
                if isinstance(out, tuple):
                    out = out[0]
                vl += loss_fn(out, yb).item() * len(xb)
        vl /= len(ldr_va.dataset)
        sched.step()

        htr.append(tl)
        hva.append(vl)

        if vl < best_val:
            best_val, best_state, wait = vl, deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= _patience:
                pbar.set_postfix(stop="early")
                break

        pbar.set_postfix(tr=f"{tl:.4f}", va=f"{vl:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    if ckpt is not None:
        torch.save(model.state_dict(), ckpt)

    return model, htr, hva


@torch.no_grad()
def predict(model: nn.Module, X_np: np.ndarray, batch_size: int = 8192) -> np.ndarray:
    """Batched inference — returns numpy array."""
    model.eval()
    dl = DataLoader(
        TensorDataset(torch.tensor(X_np, dtype=torch.float32)),
        batch_size=batch_size, shuffle=False,
    )
    out = []
    for (xb,) in dl:
        r = model(xb.to(DEVICE))
        if isinstance(r, tuple):
            r = r[0]
        out.append(r.cpu().numpy())
    return np.concatenate(out)


def free_memory(*models) -> None:
    """Delete models and flush GPU cache."""
    for m in models:
        del m
    gc.collect()
    torch.cuda.empty_cache()


# ── Deep Ensemble wrapper ─────────────────────────────────────────────────────

class DeepEnsemble:
    """
    Train N independent models and aggregate their predictions.
    Provides mean and standard deviation for uncertainty estimation.
    """

    def __init__(self, model_cls, model_kwargs: dict, n: int = 5):
        self.models = [model_cls(**model_kwargs) for _ in range(n)]
        self.n = n

    def fit(
        self,
        ldr_tr: DataLoader,
        ldr_va: DataLoader,
        name: str,
        ckpt_paths: List[Path],
        cfg: Optional[CFG] = None,
        force: bool = True,
    ) -> Tuple[List[List[float]], List[List[float]]]:
        all_tr, all_va = [], []
        for i, m in enumerate(self.models):
            m = wrap(m)
            m, htr, hva = train_torch(
                m, ldr_tr, ldr_va,
                name=f"{name}[{i}]",
                ckpt=ckpt_paths[i],
                force=force,
                cfg=cfg,
            )
            self.models[i] = m
            all_tr.append(htr)
            all_va.append(hva)
        return all_tr, all_va

    @torch.no_grad()
    def predict_mean_std(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        preds = np.stack([predict(m, X) for m in self.models])
        return preds.mean(0), preds.std(0)
