#!/usr/bin/env python3
"""
GC-Bench · Benchmark 3: Inverse Scalar
========================================
4 optical scalars → 5 geometry parameters.

Usage
-----
    python scripts/bench_inverse_scalar.py [--data /path/to/data.h5] [--no-force]
"""

import argparse
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

from gc_bench import (
    CFG, load_data, wrap, make_loader, train_torch, predict, free_memory,
    metrics_inverse, print_inverse_table, print_device_info,
)
from gc_bench.models import (
    MLP, ResNet, FTTransformer,
    MDN, CVAE, cvae_loss, RealNVP,
    physics_forward_torch,
)
from gc_bench.training import DEVICE


def parse_args():
    p = argparse.ArgumentParser(description="GC-Bench Benchmark 3: Inverse Scalar")
    p.add_argument("--data",     default=None)
    p.add_argument("--no-force", action="store_true")
    p.add_argument("--out-dir",  default="/kaggle/working/gc_bench")
    return p.parse_args()


def _train_mdn(mdn, Xin_tr, Yo_tr, cfg, ckpt, force):
    if not force and ckpt.exists():
        mdn.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        return mdn
    opt  = torch.optim.AdamW(mdn.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    ds   = TensorDataset(torch.tensor(Xin_tr, dtype=torch.float32),
                         torch.tensor(Yo_tr,  dtype=torch.float32))
    dl   = DataLoader(ds, cfg.batch_size, True, num_workers=2, pin_memory=True)
    best_v, best_s, wait = 1e9, None, 0
    pbar = tqdm(range(1, cfg.dl_epochs + 1), desc="  MDN_inv", leave=False)
    for ep in pbar:
        mdn.train(); tl = 0
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE); opt.zero_grad()
            mod  = mdn.module if hasattr(mdn, "module") else mdn
            loss = mod.loss(xb, yb); loss.backward(); opt.step()
            tl  += loss.item() * len(xb)
        tl /= len(dl.dataset)
        if tl < best_v: best_v, best_s, wait = tl, deepcopy(mdn.state_dict()), 0
        else:
            wait += 1
            if wait >= cfg.patience: break
        pbar.set_postfix(loss=f"{tl:.4f}")
    mdn.load_state_dict(best_s); torch.save(mdn.state_dict(), ckpt)
    return mdn


def _train_cvae(cvae_m, X_cond_tr, X_out_tr, X_cond_va, X_out_va, cfg, ckpt, force):
    if not force and ckpt.exists():
        cvae_m.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        return cvae_m
    opt   = torch.optim.AdamW(cvae_m.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.dl_epochs)
    ds    = TensorDataset(torch.tensor(X_cond_tr, dtype=torch.float32),
                          torch.tensor(X_out_tr,  dtype=torch.float32))
    dl    = DataLoader(ds, cfg.batch_size, True, num_workers=2, pin_memory=True)
    Xc_va = torch.tensor(X_cond_va, dtype=torch.float32).to(DEVICE)
    Xo_va = torch.tensor(X_out_va,  dtype=torch.float32).to(DEVICE)
    best_v, best_s, wait = 1e9, None, 0
    pbar = tqdm(range(1, cfg.dl_epochs + 1), desc="  cVAE_inv", leave=False)
    for ep in pbar:
        cvae_m.train(); tl = 0
        for cb, xb in dl:
            cb, xb = cb.to(DEVICE), xb.to(DEVICE); opt.zero_grad(set_to_none=True)
            mod = cvae_m.module if hasattr(cvae_m, "module") else cvae_m
            recon, mu, lv = mod(xb, cb)
            loss = cvae_loss(recon, xb, mu, lv); loss.backward(); opt.step()
            tl += loss.item() * len(cb)
        tl /= len(dl.dataset); sched.step()
        cvae_m.eval()
        with torch.no_grad():
            mod = cvae_m.module if hasattr(cvae_m, "module") else cvae_m
            r, mu, lv = mod(Xo_va, Xc_va)
            vl = cvae_loss(r, Xo_va, mu, lv).item()
        if vl < best_v: best_v, best_s, wait = vl, deepcopy(cvae_m.state_dict()), 0
        else:
            wait += 1
            if wait >= cfg.patience: break
        pbar.set_postfix(tr=f"{tl:.4f}", va=f"{vl:.4f}")
    cvae_m.load_state_dict(best_s); torch.save(cvae_m.state_dict(), ckpt)
    return cvae_m


def _train_flow(flow, Yo_tr, Xin_tr, cfg, ckpt, force):
    if not force and ckpt.exists():
        flow.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        return flow
    opt  = torch.optim.AdamW(flow.parameters(), lr=3e-4, weight_decay=cfg.wd)
    ds   = TensorDataset(torch.tensor(Yo_tr,  dtype=torch.float32),
                         torch.tensor(Xin_tr, dtype=torch.float32))
    dl   = DataLoader(ds, cfg.batch_size, True, num_workers=2, pin_memory=True)
    best_v, best_s, wait = 1e9, None, 0
    pbar = tqdm(range(1, cfg.dl_epochs + 1), desc="  Flow_inv", leave=False)
    for ep in pbar:
        flow.train(); tl = 0
        for xb, cb in dl:
            xb, cb = xb.to(DEVICE), cb.to(DEVICE); opt.zero_grad()
            mod  = flow.module if hasattr(flow, "module") else flow
            loss = mod(xb, cb)
            if torch.isfinite(loss): loss.backward(); opt.step()
            tl += loss.item() * len(xb) if torch.isfinite(loss) else 0
        tl /= len(dl.dataset)
        if tl < best_v: best_v, best_s, wait = tl, deepcopy(flow.state_dict()), 0
        else:
            wait += 1
            if wait >= cfg.patience: break
        pbar.set_postfix(loss=f"{tl:.4f}")
    flow.load_state_dict(best_s); torch.save(flow.state_dict(), ckpt)
    return flow


def main():
    args  = parse_args()
    FORCE = not args.no_force

    cfg = CFG(force_retrain=FORCE, out_dir=args.out_dir)
    if args.data:
        cfg.data_path = args.data

    print_device_info()

    OUTDIR = Path(cfg.out_dir); OUTDIR.mkdir(parents=True, exist_ok=True)
    CKPT   = OUTDIR / "checkpoints"; CKPT.mkdir(exist_ok=True)

    data = load_data(cfg)
    Xin_tr, Xin_va, Xin_te = data["Xin_tr"], data["Xin_va"], data["Xin_te"]
    Yo_tr,  Yo_va,  Yo_te  = data["Yo_tr"],  data["Yo_va"],  data["Yo_te"]
    X_tr,   X_te            = data["X_tr"],   data["X_te"]
    Ys_tr                   = data["Ys_tr"]
    sy_inv                  = data["sy_inv"]
    sx_inv                  = data["sx_inv"]

    ldr_tr = make_loader(Xin_tr, Yo_tr, cfg.batch_size)
    ldr_va = make_loader(Xin_va, Yo_va, cfg.batch_size, shuffle=False)
    in_inv, out_inv = Xin_tr.shape[1], Yo_tr.shape[1]

    RESULTS = {}

    print("\n" + "=" * 62)
    print("BENCHMARK 3 · INVERSE SCALAR  (4 scalars → 5 geo params)")
    print("=" * 62)

    # ── Classical ────────────────────────────────────────────────────────────
    for mname, model in [
        ("XGBoost_inv", xgb.XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05, tree_method="hist",
            device="cuda" if torch.cuda.is_available() else "cpu",
            n_jobs=-1, random_state=cfg.seed, verbosity=0)),
        ("RF_inv", RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=cfg.seed)),
    ]:
        print(f"\n[{mname}]")
        t0 = time.time(); model.fit(Xin_tr, X_tr)
        pred = model.predict(Xin_te)
        m    = metrics_inverse(X_te, pred)
        RESULTS[mname] = {"metrics": m, "time": time.time() - t0}
        print(f"  SR_strict={m['SR_strict']:.1f}%  SR_relaxed={m['SR_relaxed']:.1f}%  MAE={m['MAE'].mean():.3f}")

    # ── DL deterministic ─────────────────────────────────────────────────────
    for mname, model in [
        ("MLP_inv",    MLP(in_inv, out_inv, (256, 256, 256))),
        ("ResNet_inv", ResNet(in_inv, out_inv, 256, 4)),
        ("FTT_inv",    FTTransformer(in_inv, out_inv, 64, 4, 3)),
    ]:
        print(f"\n[{mname}]")
        ckpt  = CKPT / f"is_{mname}.pt"
        model = wrap(model)
        model, htr, hva = train_torch(model, ldr_tr, ldr_va, name=mname, ckpt=ckpt, force=FORCE, cfg=cfg)
        pred  = sy_inv.inverse_transform(predict(model, Xin_te))
        m     = metrics_inverse(X_te, pred)
        RESULTS[mname] = {"metrics": m}
        print(f"  SR_strict={m['SR_strict']:.1f}%  SR_relaxed={m['SR_relaxed']:.1f}%  MAE={m['MAE'].mean():.3f}")
        free_memory(model)

    # ── MDN ──────────────────────────────────────────────────────────────────
    print("\n[MDN_inv]")
    mdn = wrap(MDN(in_inv, out_inv, n_mix=10))
    mdn = _train_mdn(mdn, Xin_tr, Yo_tr, cfg, CKPT / "is_MDN.pt", FORCE)
    mdn.eval()
    with torch.no_grad():
        mod = mdn.module if hasattr(mdn, "module") else mdn
        pn  = mod.sample(torch.tensor(Xin_te, dtype=torch.float32).to(DEVICE)).cpu().numpy()
    pred_mdn = sy_inv.inverse_transform(pn)
    m_mdn = metrics_inverse(X_te, pred_mdn)
    RESULTS["MDN_inv"] = {"metrics": m_mdn}
    print(f"  SR_strict={m_mdn['SR_strict']:.1f}%  MAE={m_mdn['MAE'].mean():.3f}")
    free_memory(mdn)

    # ── cVAE ─────────────────────────────────────────────────────────────────
    print("\n[cVAE_inv]")
    cvae_inv = wrap(CVAE(in_inv, out_inv, 32, 256))
    cvae_inv = _train_cvae(cvae_inv, Xin_tr, Yo_tr, Xin_va, Yo_va, cfg, CKPT / "is_cVAE.pt", FORCE)
    cvae_inv.eval()
    with torch.no_grad():
        mod    = cvae_inv.module if hasattr(cvae_inv, "module") else cvae_inv
        pred_n = mod.sample(torch.tensor(Xin_te, dtype=torch.float32).to(DEVICE)).cpu().numpy()
    pred_cv = sy_inv.inverse_transform(pred_n)
    m_cv = metrics_inverse(X_te, pred_cv)
    RESULTS["cVAE_inv"] = {"metrics": m_cv}
    print(f"  SR_strict={m_cv['SR_strict']:.1f}%  MAE={m_cv['MAE'].mean():.3f}")
    free_memory(cvae_inv)

    # ── Normalizing Flow ──────────────────────────────────────────────────────
    print("\n[Flow_inv]")
    flow_inv = wrap(RealNVP(out_inv, in_inv, cfg.flow_layers, 128))
    flow_inv = _train_flow(flow_inv, Yo_tr, Xin_tr, cfg, CKPT / "is_Flow.pt", FORCE)
    flow_inv.eval()
    with torch.no_grad():
        mod    = flow_inv.module if hasattr(flow_inv, "module") else flow_inv
        pred_n = mod.sample(torch.tensor(Xin_te, dtype=torch.float32).to(DEVICE)).cpu().numpy()
    pred_fl = sy_inv.inverse_transform(pred_n)
    m_fl = metrics_inverse(X_te, pred_fl)
    RESULTS["Flow_inv"] = {"metrics": m_fl}
    print(f"  SR_strict={m_fl['SR_strict']:.1f}%  MAE={m_fl['MAE'].mean():.3f}")
    free_memory(flow_inv)

    # ── PINN Inverse ──────────────────────────────────────────────────────────
    print("\n[PINN_inv]")
    pinn_inv = wrap(ResNet(in_inv, out_inv, 256, 4))
    ckpt_pi  = CKPT / "is_PINN.pt"
    if not FORCE and ckpt_pi.exists():
        pinn_inv.load_state_dict(torch.load(ckpt_pi, map_location=DEVICE))
    else:
        opt_pi   = torch.optim.AdamW(pinn_inv.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
        sched_pi = torch.optim.lr_scheduler.CosineAnnealingLR(opt_pi, T_max=cfg.dl_epochs)
        ds_pi    = TensorDataset(
            torch.tensor(Xin_tr, dtype=torch.float32),
            torch.tensor(Yo_tr,  dtype=torch.float32),
            torch.tensor(Ys_tr,  dtype=torch.float32),
        )
        dl_pi  = DataLoader(ds_pi, cfg.batch_size, True, num_workers=2, pin_memory=True)
        best_v, best_s, wait = 1e9, None, 0
        pbar = tqdm(range(1, cfg.dl_epochs + 1), desc="  PINN_inv", leave=False)
        for ep in pbar:
            pinn_inv.train(); tl = 0
            for xb, yb, ys in dl_pi:
                xb, yb, ys = xb.to(DEVICE), yb.to(DEVICE), ys.to(DEVICE)
                opt_pi.zero_grad(set_to_none=True)
                pred_g = pinn_inv(xb)
                if isinstance(pred_g, tuple): pred_g = pred_g[0]
                pred_g_raw = torch.tensor(
                    sy_inv.inverse_transform(pred_g.detach().cpu().numpy()),
                    dtype=torch.float32).to(DEVICE)
                pred_sc   = physics_forward_torch(pred_g_raw)
                pred_sc_n = torch.tensor(
                    sx_inv.transform(pred_sc.detach().cpu().numpy()),
                    dtype=torch.float32).to(DEVICE)
                loss = F.mse_loss(pred_g, yb) + 0.1 * F.mse_loss(pred_sc_n, xb)
                loss.backward(); opt_pi.step(); tl += loss.item() * len(xb)
            tl /= len(dl_pi.dataset); sched_pi.step()
            if tl < best_v: best_v, best_s, wait = tl, deepcopy(pinn_inv.state_dict()), 0
            else:
                wait += 1
                if wait >= cfg.patience: break
            pbar.set_postfix(loss=f"{tl:.4f}")
        pinn_inv.load_state_dict(best_s); torch.save(pinn_inv.state_dict(), ckpt_pi)
    pred_pi = sy_inv.inverse_transform(predict(pinn_inv, Xin_te))
    m_pi = metrics_inverse(X_te, pred_pi)
    RESULTS["PINN_inv"] = {"metrics": m_pi}
    print(f"  SR_strict={m_pi['SR_strict']:.1f}%  MAE={m_pi['MAE'].mean():.3f}")
    free_memory(pinn_inv)

    print_inverse_table(RESULTS, task="Benchmark 3 · Inverse Scalar")
    return RESULTS


if __name__ == "__main__":
    main()
