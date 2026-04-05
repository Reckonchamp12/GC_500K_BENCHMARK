#!/usr/bin/env python3
"""
GC-Bench · Benchmark 1: Forward Scalar
=======================================
Geometry (5 params) → 4 scalar optical metrics.

Usage
-----
    python scripts/bench_forward_scalar.py [--data /path/to/data.h5] [--no-force]
"""

import argparse
import gc
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
import lightgbm as lgb
import torch

from gc_bench import (
    CFG, load_data, wrap, make_loader, train_torch, predict, free_memory,
    metrics_fwd_scalar, print_fwd_scalar_table, DeepEnsemble, print_device_info,
)
from gc_bench.models import (
    MLP, ResNet, FTTransformer, MLPMixer, NeuralODE,
    PINN_Scalar, physics_forward_torch,
)
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(description="GC-Bench Benchmark 1: Forward Scalar")
    p.add_argument("--data",     default=None,  help="Path to HDF5 dataset")
    p.add_argument("--no-force", action="store_true", help="Load checkpoints instead of retraining")
    p.add_argument("--out-dir",  default="/kaggle/working/gc_bench", help="Output directory")
    return p.parse_args()


def main():
    args  = parse_args()
    FORCE = not args.no_force

    cfg = CFG(force_retrain=FORCE, out_dir=args.out_dir)
    if args.data:
        cfg.data_path = args.data

    print_device_info()
    from gc_bench.training import DEVICE

    OUTDIR = Path(cfg.out_dir); OUTDIR.mkdir(parents=True, exist_ok=True)
    CKPT   = OUTDIR / "checkpoints"; CKPT.mkdir(exist_ok=True)

    data = load_data(cfg)
    wl   = data["wl"]

    Xtr_s, Xva_s, Xte_s = data["Xtr_s"], data["Xva_s"], data["Xte_s"]
    Ys_tr, Ys_va, Ys_te = data["Ys_tr"], data["Ys_va"], data["Ys_te"]
    X_tr,  X_va,  X_te  = data["X_tr"],  data["X_va"],  data["X_te"]
    sx_inv = data["sx_inv"]

    ldr_tr = make_loader(Xtr_s, Ys_tr, cfg.batch_size)
    ldr_va = make_loader(Xva_s, Ys_va, cfg.batch_size, shuffle=False)
    in_s, out_s = Xtr_s.shape[1], Ys_tr.shape[1]

    RESULTS = {}

    print("\n" + "═" * 62)
    print("BENCHMARK 1 · FORWARD SCALAR  (5 params → 4 scalars)")
    print("═" * 62)

    # ── GP ──────────────────────────────────────────────────────────────────
    print("\n[GaussianProcess]")
    gp_idx = np.random.choice(len(Xtr_s), cfg.gp_subset, replace=False)
    t0 = time.time(); gp_preds = []
    for col in range(Ys_tr.shape[1]):
        gp = GaussianProcessRegressor(kernel=Matern(nu=2.5) + WhiteKernel(1e-3), n_restarts_optimizer=2)
        gp.fit(Xtr_s[gp_idx], Ys_tr[gp_idx, col])
        gp_preds.append(gp.predict(Xte_s))
    pred_gp = np.column_stack(gp_preds)
    m = metrics_fwd_scalar(Ys_te, pred_gp)
    RESULTS["GP"] = {"metrics": m, "time": time.time() - t0}
    print(f"  R²={m['R2'].mean():.4f}  MAE={m['MAE'].mean():.4f}")

    # ── Classical ───────────────────────────────────────────────────────────
    for mname, model in [
        ("LinearReg",    LinearRegression()),
        ("Ridge",        Ridge(alpha=1.0)),
        ("RandomForest", RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=cfg.seed)),
        ("XGBoost",      xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                                           tree_method="hist",
                                           device="cuda" if torch.cuda.is_available() else "cpu",
                                           n_jobs=-1, random_state=cfg.seed, verbosity=0)),
        ("LightGBM",     lgb.LGBMRegressor(n_estimators=300, num_leaves=63, learning_rate=0.05,
                                            n_jobs=-1, random_state=cfg.seed, verbose=-1)),
    ]:
        print(f"\n[{mname}]")
        t0 = time.time()
        try:
            model.fit(Xtr_s, Ys_tr)
        except Exception:
            model = MultiOutputRegressor(model); model.fit(Xtr_s, Ys_tr)
        pred = model.predict(Xte_s)
        m = metrics_fwd_scalar(Ys_te, pred)
        RESULTS[mname] = {"metrics": m, "time": time.time() - t0}
        print(f"  R²={m['R2'].mean():.4f}  MAE={m['MAE'].mean():.4f}")

    # ── DL models ────────────────────────────────────────────────────────────
    for mname, model in [
        ("MLP",          MLP(in_s, out_s, (256, 256, 256))),
        ("ResNet",       ResNet(in_s, out_s, 256, 4)),
        ("FTTransformer",FTTransformer(in_s, out_s, 128, 4, 4)),
        ("MLPMixer",     MLPMixer(in_s, out_s, 8, 64, 4)),
        ("NeuralODE",    NeuralODE(in_s, out_s, 64)),
    ]:
        print(f"\n[{mname}]")
        ckpt  = CKPT / f"fs_{mname}.pt"
        model = wrap(model)
        model, htr, hva = train_torch(model, ldr_tr, ldr_va, name=mname, ckpt=ckpt, force=FORCE, cfg=cfg)
        pred  = predict(model, Xte_s)
        m     = metrics_fwd_scalar(Ys_te, pred)
        RESULTS[mname] = {"metrics": m}
        print(f"  R²={m['R2'].mean():.4f}  MAE={m['MAE'].mean():.4f}")
        free_memory(model)

    # ── Deep Ensemble ─────────────────────────────────────────────────────────
    print("\n[DeepEnsemble]")
    ens    = DeepEnsemble(MLP, {"in_d": in_s, "out_d": out_s, "hidden": (256, 256, 256)}, n=cfg.ensemble_n)
    ckpts_ = [CKPT / f"fs_ens_{i}.pt" for i in range(cfg.ensemble_n)]
    ens.fit(ldr_tr, ldr_va, "Ens", ckpts_, cfg=cfg, force=FORCE)
    pred_e, pred_std = ens.predict_mean_std(Xte_s)
    m = metrics_fwd_scalar(Ys_te, pred_e)
    m["pred_std"] = float(pred_std.mean())
    RESULTS["DeepEnsemble"] = {"metrics": m}
    print(f"  R²={m['R2'].mean():.4f}  MAE={m['MAE'].mean():.4f}  std={m['pred_std']:.5f}")
    for m_ in ens.models:
        free_memory(m_)

    # ── Summary ───────────────────────────────────────────────────────────────
    print_fwd_scalar_table(RESULTS)
    return RESULTS


if __name__ == "__main__":
    main()
