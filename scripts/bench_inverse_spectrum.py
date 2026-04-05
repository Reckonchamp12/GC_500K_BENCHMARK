#!/usr/bin/env python3
"""
GC-Bench · Benchmark 4: Inverse Spectrum
==========================================
100-point transmittance spectrum → 5 geometry parameters.
Includes deep ensembles, tandem networks, cVAE, diffusion, and noise robustness.

Usage
-----
    python scripts/bench_inverse_spectrum.py [--data /path/to/data.h5] [--no-force]
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
from sklearn.metrics import mean_absolute_error

from gc_bench import (
    CFG, load_data, wrap, make_loader, train_torch, predict, free_memory,
    metrics_inverse, print_inverse_table, print_device_info, success_rate,
    DeepEnsemble,
)
from gc_bench.metrics import metrics_inv_spectrum
from gc_bench.models import (
    MLP, ResNet, FTTransformer, CNN1D,
    CVAE, cvae_loss, INN, DDPM,
)
from gc_bench.training import DEVICE


def parse_args():
    p = argparse.ArgumentParser(description="GC-Bench Benchmark 4: Inverse Spectrum")
    p.add_argument("--data",     default=None)
    p.add_argument("--no-force", action="store_true")
    p.add_argument("--out-dir",  default="/kaggle/working/gc_bench")
    return p.parse_args()


def _train_cvae(cvae_m, X_cond_tr, X_out_tr, X_cond_va, X_out_va, cfg, ckpt, force, name):
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
    pbar = tqdm(range(1, cfg.dl_epochs + 1), desc=f"  {name}", leave=False)
    for ep in pbar:
        cvae_m.train(); tl = 0
        for cb, xb in dl:
            cb, xb = cb.to(DEVICE), xb.to(DEVICE); opt.zero_grad(set_to_none=True)
            mod = cvae_m.module if hasattr(cvae_m, "module") else cvae_m
            recon, mu, lv = mod(xb, cb)
            loss = cvae_loss(recon, xb, mu, lv); loss.backward(); opt.step()
            tl += loss.item() * len(cb)
        tl /= len(ds); sched.step()
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
    Xsp_tr, Xsp_va, Xsp_te = data["Xsp_tr"], data["Xsp_va"], data["Xsp_te"]
    Yg_tr,  Yg_va,  Yg_te  = data["Yg_tr"],  data["Yg_va"],  data["Yg_te"]
    X_te                    = data["X_te"]
    Ysp_te                  = data["Ysp_te"]
    sy_sp2g                 = data["sy_sp2g"]
    # Also need forward data for tandem
    Xtr_s, Xva_s            = data["Xtr_s"], data["Xva_s"]
    Ysp_tr_n, Ysp_va_n      = data["Ysp_tr_n"], data["Ysp_va_n"]

    ldr_tr = make_loader(Xsp_tr, Yg_tr, cfg.batch_size)
    ldr_va = make_loader(Xsp_va, Yg_va, cfg.batch_size, shuffle=False)
    in_spg, out_spg = Xsp_tr.shape[1], Yg_tr.shape[1]

    RESULTS = {}
    NOISE_RESULTS = {}

    print("\n" + "═" * 62)
    print("BENCHMARK 4 · INVERSE SPECTRUM  (100-pt spec → 5 geo params)")
    print("═" * 62)

    # ── DL deterministic ─────────────────────────────────────────────────────
    for mname, model in [
        ("MLP_sp",    MLP(in_spg, out_spg, (512, 256, 128))),
        ("ResNet_sp", ResNet(in_spg, out_spg, 256, 4)),
        ("CNN1D_sp",  CNN1D(in_spg, out_spg)),
        ("FTT_sp",    FTTransformer(in_spg, out_spg, 128, 4, 4)),
        ("INN_sp",    INN(in_spg, out_spg, 6)),
    ]:
        print(f"\n[{mname}]")
        ckpt  = CKPT / f"isp_{mname}.pt"
        model = wrap(model)
        model, htr, hva = train_torch(model, ldr_tr, ldr_va, name=mname, ckpt=ckpt, force=FORCE, cfg=cfg)
        pred  = sy_sp2g.inverse_transform(predict(model, Xsp_te))
        m     = metrics_inv_spectrum(X_te, pred, Ysp_te)
        RESULTS[mname] = {"metrics": m}
        print(
            f"  SR_strict={m['SR_strict']:.1f}%  SR_relaxed={m['SR_relaxed']:.1f}%"
            f"  MAE={m['MAE'].mean():.3f}  CondNum={m['CondNum']:.3f}"
        )
        free_memory(model)

    # ── Deep Ensemble ─────────────────────────────────────────────────────────
    print("\n[DeepEns_sp]")
    ens_sp  = DeepEnsemble(CNN1D, {"seq": in_spg, "out_d": out_spg}, n=cfg.ensemble_n)
    ckpts_e = [CKPT / f"isp_ens_{i}.pt" for i in range(cfg.ensemble_n)]
    ens_sp.fit(ldr_tr, ldr_va, "EnsInvSp", ckpts_e, cfg=cfg, force=FORCE)
    pred_ens_n, pred_std = ens_sp.predict_mean_std(Xsp_te)
    pred_ens = sy_sp2g.inverse_transform(pred_ens_n)
    m_ens = metrics_inv_spectrum(X_te, pred_ens, Ysp_te)
    m_ens["uncertainty"] = float(pred_std.mean())
    RESULTS["DeepEns_sp"] = {"metrics": m_ens}
    print(f"  SR_strict={m_ens['SR_strict']:.1f}%  MAE={m_ens['MAE'].mean():.3f}  Uncertainty={m_ens['uncertainty']:.4f}")
    for md_ in ens_sp.models:
        free_memory(md_)

    # ── Tandem Network ────────────────────────────────────────────────────────
    print("\n[Tandem]")
    fwd_tan = wrap(MLP(5, 100, (512, 512, 512))); ckpt_tf = CKPT / "isp_tandem_fwd.pt"
    lf  = make_loader(Xtr_s, Ysp_tr_n, cfg.batch_size)
    lfv = make_loader(Xva_s, Ysp_va_n, cfg.batch_size, shuffle=False)
    fwd_tan, _, _ = train_torch(fwd_tan, lf, lfv, name="tandem_fwd",
                                 epochs=cfg.dl_epochs // 2, ckpt=ckpt_tf, force=FORCE, cfg=cfg)
    fwd_tan.eval()
    for p in fwd_tan.parameters():
        p.requires_grad_(False)

    inv_tan = wrap(CNN1D(in_spg, out_spg)); ckpt_ti = CKPT / "isp_tandem_inv.pt"
    if not FORCE and ckpt_ti.exists():
        inv_tan.load_state_dict(torch.load(ckpt_ti, map_location=DEVICE))
    else:
        opt_t = torch.optim.AdamW(inv_tan.parameters(), lr=cfg.lr)
        best_v, best_s, wait = 1e9, None, 0
        pbar = tqdm(range(1, cfg.dl_epochs + 1), desc="  Tandem", leave=False)
        for ep in pbar:
            inv_tan.train(); tl = 0
            for xb, yb in ldr_tr:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE); opt_t.zero_grad(set_to_none=True)
                gp = inv_tan(xb)
                if isinstance(gp, tuple): gp = gp[0]
                sp = fwd_tan(gp)
                if isinstance(sp, tuple): sp = sp[0]
                loss = F.mse_loss(gp, yb) + 0.5 * F.mse_loss(sp, xb)
                loss.backward(); opt_t.step(); tl += loss.item() * len(xb)
            tl /= len(ldr_tr.dataset)
            if tl < best_v: best_v, best_s, wait = tl, deepcopy(inv_tan.state_dict()), 0
            else:
                wait += 1
                if wait >= cfg.patience: break
            pbar.set_postfix(loss=f"{tl:.4f}")
        inv_tan.load_state_dict(best_s); torch.save(inv_tan.state_dict(), ckpt_ti)
    pred_tan = sy_sp2g.inverse_transform(predict(inv_tan, Xsp_te))
    m_tan = metrics_inv_spectrum(X_te, pred_tan, Ysp_te)
    RESULTS["Tandem"] = {"metrics": m_tan}
    print(f"  SR_strict={m_tan['SR_strict']:.1f}%  MAE={m_tan['MAE'].mean():.3f}")
    free_memory(fwd_tan, inv_tan)

    # ── cVAE ─────────────────────────────────────────────────────────────────
    print("\n[cVAE_sp]")
    cvae_sp = wrap(CVAE(in_spg, out_spg, 32, 256))
    cvae_sp = _train_cvae(cvae_sp, Xsp_tr, Yg_tr, Xsp_va, Yg_va, cfg, CKPT / "isp_cVAE.pt", FORCE, "cVAE_sp")
    cvae_sp.eval()
    with torch.no_grad():
        mod = cvae_sp.module if hasattr(cvae_sp, "module") else cvae_sp
        pn  = mod.sample(torch.tensor(Xsp_te, dtype=torch.float32).to(DEVICE)).cpu().numpy()
    pred_cvs = sy_sp2g.inverse_transform(pn)
    m_cvs = metrics_inv_spectrum(X_te, pred_cvs, Ysp_te)
    RESULTS["cVAE_sp"] = {"metrics": m_cvs}
    print(f"  SR_strict={m_cvs['SR_strict']:.1f}%  MAE={m_cvs['MAE'].mean():.3f}")
    free_memory(cvae_sp)

    # ── Diffusion ─────────────────────────────────────────────────────────────
    print("\n[Diffusion_sp]")
    ddpm_sp = wrap(DDPM(out_spg, in_spg, T=cfg.diff_steps)); ckpt_dd = CKPT / "isp_Diffusion.pt"
    if not FORCE and ckpt_dd.exists():
        ddpm_sp.load_state_dict(torch.load(ckpt_dd, map_location=DEVICE))
    else:
        opt_dd = torch.optim.AdamW(ddpm_sp.parameters(), lr=1e-4)
        ds_dd  = TensorDataset(torch.tensor(Yg_tr,  dtype=torch.float32),
                               torch.tensor(Xsp_tr, dtype=torch.float32))
        dl_dd  = DataLoader(ds_dd, cfg.batch_size, True, num_workers=2, pin_memory=True)
        best_v, best_s, wait = 1e9, None, 0
        pbar = tqdm(range(1, min(cfg.dl_epochs, 30) + 1), desc="  Diffusion_sp", leave=False)
        for ep in pbar:
            ddpm_sp.train(); tl = 0
            for xb, cb in dl_dd:
                xb, cb = xb.to(DEVICE), cb.to(DEVICE); opt_dd.zero_grad()
                mod  = ddpm_sp.module if hasattr(ddpm_sp, "module") else ddpm_sp
                loss = mod(xb, cb); loss.backward(); opt_dd.step(); tl += loss.item() * len(xb)
            tl /= len(ds_dd)
            if tl < best_v: best_v, best_s, wait = tl, deepcopy(ddpm_sp.state_dict()), 0
            else:
                wait += 1
                if wait >= cfg.patience: break
            pbar.set_postfix(loss=f"{tl:.4f}")
        ddpm_sp.load_state_dict(best_s); torch.save(ddpm_sp.state_dict(), ckpt_dd)
    ddpm_sp.eval()
    with torch.no_grad():
        mod    = ddpm_sp.module if hasattr(ddpm_sp, "module") else ddpm_sp
        pn_dd  = mod.sample(torch.tensor(Xsp_te, dtype=torch.float32).to(DEVICE), (len(Xsp_te), out_spg)).cpu().numpy()
    pred_dd = sy_sp2g.inverse_transform(pn_dd)
    m_dd = metrics_inv_spectrum(X_te, pred_dd, Ysp_te)
    RESULTS["Diffusion_sp"] = {"metrics": m_dd}
    print(f"  SR_strict={m_dd['SR_strict']:.1f}%  MAE={m_dd['MAE'].mean():.3f}")
    free_memory(ddpm_sp)

    # ── Noise Robustness ──────────────────────────────────────────────────────
    print("\n[Noise Robustness — ResNet_sp]")
    nr_m = wrap(ResNet(in_spg, out_spg, 256, 4))
    nr_m, _, _ = train_torch(nr_m, ldr_tr, ldr_va, name="NR_ResNet",
                              ckpt=CKPT / "isp_noise_resnet.pt", force=FORCE, cfg=cfg)
    for sig in cfg.noise_sigmas:
        noisy   = Xsp_te + np.random.randn(*Xsp_te.shape).astype(np.float32) * sig
        pred_nr = sy_sp2g.inverse_transform(predict(nr_m, noisy))
        sr  = success_rate(X_te, pred_nr)
        mae = mean_absolute_error(X_te, pred_nr, multioutput="raw_values").mean()
        NOISE_RESULTS[f"sig={sig}"] = {"SR": sr, "MAE": mae, "sigma": sig}
        print(f"  σ={sig:<7}  SR={sr:.1f}%  MAE={mae:.4f}")
    free_memory(nr_m)

    # ── Summary ───────────────────────────────────────────────────────────────
    print_inverse_table(RESULTS, task="Benchmark 4 · Inverse Spectrum")

    print("\n\n  Noise Robustness Summary")
    print(f"  {'σ':<8} {'SR%':>6}  {'MAE':>9}")
    print("  " + "─" * 26)
    for k, v in NOISE_RESULTS.items():
        print(f"  {v['sigma']:<8}  {v['SR']:>5.1f}%  {v['MAE']:>9.4f}")

    return RESULTS, NOISE_RESULTS


if __name__ == "__main__":
    main()
