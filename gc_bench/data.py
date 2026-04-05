"""
GC-Bench · Data Loading & Preprocessing
========================================
Loads the HDF5 dataset, builds train/val/test splits,
and prepares all normalised arrays used by each benchmark.
"""

import time
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import CFG


def load_data(cfg: CFG) -> dict:
    """
    Load the grating-coupler HDF5 dataset and return a single dict
    containing raw arrays, split indices, scalers, and all normalised views.

    Returns
    -------
    dict with keys:
        wl              : wavelength array (nm)
        X_params        : (N, 5) geometry matrix
        Y_scalars       : (N, 4) scalar-metric matrix
        Y_spectra       : (N, 100) transmittance matrix
        idx_tr/va/te    : split indices
        X_tr/va/te      : raw geometry splits
        Ys_tr/va/te     : raw scalar splits
        Ysp_tr/va/te    : raw spectral splits
        Xtr_s / ...     : geometry normalised (for forward models)
        Xin_tr / ...    : scalars normalised  (for inverse-scalar models)
        Xsp_tr / ...    : spectra normalised  (for inverse-spectrum models)
        Yo_tr / ...     : geometry normalised (target side of inverse)
        Yg_tr / ...     : geometry normalised (target side of inv-spectrum)
        Ysp_tr_n / ...  : spectra normalised  (target side of fwd-spectrum)
        sx / sy / ssp   : scalers
        sx_inv / sy_inv : inverse-scalar scalers
        sx_sp2g / sy_sp2g: inverse-spectrum scalers
    """
    print("\n=== Loading dataset ===")
    t0 = time.time()
    with h5py.File(cfg.data_path, "r") as f:
        T_full  = f["T"][:]
        wl      = f["wavelengths_um"][:] * 1000        # → nm
        valid   = f["valid"][:]
        params  = {k: f[f"parameters/{k}"][:] for k in cfg.param_names}
        scalars = {k: f[f"metrics/{k}"][:]    for k in cfg.scalar_names}

    N = len(T_full)
    print(f"  Loaded {N:,} samples in {time.time()-t0:.1f}s  "
          f"valid={valid.mean()*100:.1f}%  "
          f"λ=[{wl[0]:.0f}, {wl[-1]:.0f}] nm")

    X_params  = np.column_stack([params[k]  for k in cfg.param_names ]).astype(np.float32)
    Y_scalars = np.column_stack([scalars[k] for k in cfg.scalar_names]).astype(np.float32)
    Y_spectra = T_full.astype(np.float32)

    # ── Splits ──────────────────────────────────────────────────────────
    np.random.seed(cfg.seed)
    idx = np.arange(N)
    idx_tv, idx_te = train_test_split(idx, test_size=0.15, random_state=cfg.seed)
    idx_tr, idx_va = train_test_split(idx_tv, test_size=0.15/0.85, random_state=cfg.seed)
    print(f"  train:{len(idx_tr):,}  val:{len(idx_va):,}  test:{len(idx_te):,}")

    def split(a):
        return a[idx_tr], a[idx_va], a[idx_te]

    X_tr,  X_va,  X_te  = split(X_params)
    Ys_tr, Ys_va, Ys_te = split(Y_scalars)
    Ysp_tr,Ysp_va,Ysp_te = split(Y_spectra)

    # ── Forward scalers ─────────────────────────────────────────────────
    sx  = StandardScaler().fit(X_tr)
    sy  = StandardScaler().fit(Ys_tr)
    ssp = StandardScaler().fit(Ysp_tr)

    Xtr_s, Xva_s, Xte_s           = sx.transform(X_tr),   sx.transform(X_va),   sx.transform(X_te)
    Ytr_s, Yva_s, Yte_s            = sy.transform(Ys_tr),  sy.transform(Ys_va),  sy.transform(Ys_te)
    Ysp_tr_n, Ysp_va_n, Ysp_te_n  = ssp.transform(Ysp_tr),ssp.transform(Ysp_va),ssp.transform(Ysp_te)

    # ── Inverse-scalar scalers (input=scalars, output=geometry) ─────────
    sx_inv = StandardScaler().fit(Ys_tr)
    sy_inv = StandardScaler().fit(X_tr)
    Xin_tr, Xin_va, Xin_te = (sx_inv.transform(Ys_tr),
                                sx_inv.transform(Ys_va),
                                sx_inv.transform(Ys_te))
    Yo_tr, Yo_va, Yo_te    = (sy_inv.transform(X_tr),
                                sy_inv.transform(X_va),
                                sy_inv.transform(X_te))

    # ── Inverse-spectrum scalers (input=spectra, output=geometry) ───────
    sx_sp2g = StandardScaler().fit(Ysp_tr)
    sy_sp2g = StandardScaler().fit(X_tr)
    Xsp_tr, Xsp_va, Xsp_te = (sx_sp2g.transform(Ysp_tr),
                                sx_sp2g.transform(Ysp_va),
                                sx_sp2g.transform(Ysp_te))
    Yg_tr, Yg_va, Yg_te    = (sy_sp2g.transform(X_tr),
                                sy_sp2g.transform(X_va),
                                sy_sp2g.transform(X_te))

    return dict(
        wl=wl, valid=valid,
        X_params=X_params, Y_scalars=Y_scalars, Y_spectra=Y_spectra,
        idx_tr=idx_tr, idx_va=idx_va, idx_te=idx_te,
        # Raw splits
        X_tr=X_tr,   X_va=X_va,   X_te=X_te,
        Ys_tr=Ys_tr, Ys_va=Ys_va, Ys_te=Ys_te,
        Ysp_tr=Ysp_tr, Ysp_va=Ysp_va, Ysp_te=Ysp_te,
        # Forward-scalar (normalised geometry → normalised scalars)
        Xtr_s=Xtr_s, Xva_s=Xva_s, Xte_s=Xte_s,
        Ytr_s=Ytr_s, Yva_s=Yva_s, Yte_s=Yte_s,
        # Forward-spectrum (normalised geometry → normalised spectrum)
        Ysp_tr_n=Ysp_tr_n, Ysp_va_n=Ysp_va_n, Ysp_te_n=Ysp_te_n,
        # Inverse-scalar
        Xin_tr=Xin_tr, Xin_va=Xin_va, Xin_te=Xin_te,
        Yo_tr=Yo_tr,   Yo_va=Yo_va,   Yo_te=Yo_te,
        # Inverse-spectrum
        Xsp_tr=Xsp_tr, Xsp_va=Xsp_va, Xsp_te=Xsp_te,
        Yg_tr=Yg_tr,   Yg_va=Yg_va,   Yg_te=Yg_te,
        # Scalers (for inverse_transform)
        sx=sx, sy=sy, ssp=ssp,
        sx_inv=sx_inv, sy_inv=sy_inv,
        sx_sp2g=sx_sp2g, sy_sp2g=sy_sp2g,
    )
