#!/usr/bin/env python3
"""
GC-Bench · Benchmark 2: Forward Spectrum
=========================================
Geometry (5 params) → full 100-point transmittance spectrum.

Usage
-----
    python scripts/bench_forward_spectrum.py [--data /path/to/data.h5] [--no-force]
"""

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import torch

from gc_bench import (
    CFG, load_data, wrap, make_loader, train_torch, predict, free_memory,
    metrics_fwd_spectrum, print_fwd_spectrum_table, print_device_info,
)
from gc_bench.models import (
    MLP, ResNet, FTTransformer,
    UNet1D, FNO1d, DeepONet, NeuralField, PINN_Spectral,
)


def parse_args():
    p = argparse.ArgumentParser(description="GC-Bench Benchmark 2: Forward Spectrum")
    p.add_argument("--data",     default=None)
    p.add_argument("--no-force", action="store_true")
    p.add_argument("--out-dir",  default="/kaggle/working/gc_bench")
    return p.parse_args()


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
    wl   = data["wl"]
    ssp  = data["ssp"]

    Xtr_s, Xva_s, Xte_s           = data["Xtr_s"],    data["Xva_s"],    data["Xte_s"]
    Ysp_tr_n, Ysp_va_n, Ysp_te_n  = data["Ysp_tr_n"], data["Ysp_va_n"], data["Ysp_te_n"]
    Ysp_te                         = data["Ysp_te"]

    ldr_tr = make_loader(Xtr_s, Ysp_tr_n, cfg.batch_size)
    ldr_va = make_loader(Xva_s, Ysp_va_n, cfg.batch_size, shuffle=False)
    in_sp, out_sp = Xtr_s.shape[1], Ysp_tr_n.shape[1]

    RESULTS = {}

    print("\n" + "═" * 62)
    print("BENCHMARK 2 · FORWARD SPECTRUM  (5 params → 100-pt spectrum)")
    print("═" * 62)

    model_registry = {
        "MLP_spec":           MLP(in_sp, out_sp, (512, 512, 512)),
        "ResNet_spec":        ResNet(in_sp, out_sp, 512, 6),
        "FTTransformer_spec": FTTransformer(in_sp, out_sp, 128, 4, 4),
        "UNet1D":             UNet1D(in_sp, out_sp),
        "FNO":                FNO1d(in_sp, out_sp, modes=16, width=32, n_layers=4),
        "DeepONet":           DeepONet(in_sp, 1, out_sp, 128, 64),
        "NeuralField":        NeuralField(in_sp, 128, 4, out_sp),
        "PINN_Spectral":      PINN_Spectral(in_sp, out_sp, 512, 6),
    }

    for mname, model in model_registry.items():
        print(f"\n[{mname}]")
        ckpt  = CKPT / f"sp_{mname}.pt"
        model = wrap(model)
        model, htr, hva = train_torch(
            model, ldr_tr, ldr_va, name=mname, ckpt=ckpt, force=FORCE, cfg=cfg
        )
        pred_n = predict(model, Xte_s)
        pred   = np.clip(ssp.inverse_transform(pred_n), 0, 1)
        m      = metrics_fwd_spectrum(Ysp_te, pred, wl, sub=cfg.dtw_sub)
        RESULTS[mname] = {"metrics": m}
        print(
            f"  MSE={m['MSE']:.5f}  CosSim={m['CosSim']:.4f}"
            f"  DTW={m['DTW']:.3f}  PkWL={m['PkWL_nm']:.2f}nm"
        )
        free_memory(model)

    print_fwd_spectrum_table(RESULTS)
    return RESULTS


if __name__ == "__main__":
    main()
