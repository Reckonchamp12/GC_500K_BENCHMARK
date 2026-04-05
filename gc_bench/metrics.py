"""
GC-Bench · Evaluation Metrics
==============================
All metric functions used across the four benchmarks.

Forward Scalar  : metrics_fwd_scalar
Forward Spectrum: metrics_fwd_spectrum  (+ helpers: sam, cosine_sim, dtw, ...)
Inverse         : metrics_inverse, success_rate, sr_per_param
Inverse Spectrum: metrics_inv_spectrum  (adds condition-number proxy)
"""

import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.neighbors import NearestNeighbors

try:
    from dtaidistance import dtw as dtw_lib
    HAS_DTW = True
except ImportError:
    HAS_DTW = False


# ── Shared helpers ────────────────────────────────────────────────────────────

def _sdiv(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Safe element-wise division."""
    return a / (np.abs(b) + eps)


def cosine_sim(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Per-sample cosine similarity. Shape: (N,)."""
    return (
        np.sum(A * B, axis=1)
        / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-8)
    )


def sam(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Spectral Angle Mapper in degrees. Shape: (N,)."""
    d = np.sum(A * B, axis=1)
    return np.degrees(
        np.arccos(
            np.clip(d / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-8), -1, 1)
        )
    )


def success_rate(yt: np.ndarray, yp: np.ndarray, tol: float = 0.05) -> float:
    """Fraction of samples where ALL parameters are within ±tol (relative)."""
    return (np.abs(_sdiv(yt - yp, yt)) * 100 < tol * 100).all(axis=1).mean() * 100


def sr_per_param(yt: np.ndarray, yp: np.ndarray, tol: float = 0.05) -> np.ndarray:
    """Per-parameter success rate at relative tolerance tol."""
    return (np.abs(_sdiv(yt - yp, yt)) * 100 < tol * 100).mean(axis=0) * 100


# ── Task 1: Forward Scalar ────────────────────────────────────────────────────

def metrics_fwd_scalar(yt: np.ndarray, yp: np.ndarray) -> dict:
    """
    Compute all forward-scalar metrics.

    Parameters
    ----------
    yt : (N, 4) ground-truth scalars
    yp : (N, 4) predicted scalars

    Returns
    -------
    dict with keys: R2, MAE, RMSE, MAPE, Corr, MaxE, dB_err
    """
    r2   = r2_score(yt, yp, multioutput="raw_values")
    mae  = mean_absolute_error(yt, yp, multioutput="raw_values")
    rmse = np.sqrt(mean_squared_error(yt, yp, multioutput="raw_values"))
    mape = np.mean(np.abs(_sdiv(yt - yp, yt)) * 100, axis=0)
    corr = np.array([np.corrcoef(yt[:, i], yp[:, i])[0, 1] for i in range(yt.shape[1])])
    maxe = np.max(np.abs(yt - yp), axis=0)
    # dB error on peak_transmission (column index 3)
    dberr = np.mean(
        np.abs(
            10 * np.log10(np.clip(yp[:, 3], 1e-8, None))
            - 10 * np.log10(np.clip(yt[:, 3], 1e-8, None))
        )
    )
    return {"R2": r2, "MAE": mae, "RMSE": rmse, "MAPE": mape,
            "Corr": corr, "MaxE": maxe, "dB_err": dberr}


def print_fwd_scalar_table(results: dict) -> None:
    """Pretty-print forward-scalar results table (no inference time)."""
    hdr = f"{'Model':<18}{'R²':>8}{'MAE':>9}{'RMSE':>9}{'MAPE%':>9}{'Corr':>8}{'dB_err':>9}"
    print(f"\n{'─'*70}")
    print(hdr)
    print('─' * 70)
    for mn, res in results.items():
        m = res["metrics"]
        print(
            f"{mn:<18}"
            f"{m['R2'].mean():>8.4f}"
            f"{m['MAE'].mean():>9.4f}"
            f"{m['RMSE'].mean():>9.4f}"
            f"{m['MAPE'].mean():>9.2f}"
            f"{m['Corr'].mean():>8.4f}"
            f"{float(m['dB_err']):>9.4f}"
        )


# ── Task 2: Forward Spectrum ──────────────────────────────────────────────────

def _compute_dtw(A: np.ndarray, B: np.ndarray, sub: int = 10) -> float:
    if not HAS_DTW:
        return float("nan")
    A2, B2 = A[:, ::sub], B[:, ::sub]
    return float(np.mean([dtw_lib.distance(a, b) for a, b in zip(A2, B2)]))


def _peak_wl_err(yt: np.ndarray, yp: np.ndarray, wl: np.ndarray) -> float:
    return float(np.mean(np.abs(wl[np.argmax(yt, 1)] - wl[np.argmax(yp, 1)])))


def _bw_err(yt: np.ndarray, yp: np.ndarray, wl: np.ndarray, thr: float = 0.5) -> float:
    dwl = wl[1] - wl[0]
    def bw(sp): return (sp > thr * sp.max(1, keepdims=True)).sum(1) * dwl
    return float(np.mean(np.abs(bw(yt) - bw(yp))))


def _pw_corr(yt: np.ndarray, yp: np.ndarray) -> float:
    return float(np.mean([np.corrcoef(yt[:, i], yp[:, i])[0, 1] for i in range(yt.shape[1])]))


def metrics_fwd_spectrum(
    yt: np.ndarray, yp: np.ndarray, wl: np.ndarray, sub: int = 10
) -> dict:
    """
    Compute all forward-spectrum metrics.

    Parameters
    ----------
    yt  : (N, 100) ground-truth transmittance
    yp  : (N, 100) predicted transmittance
    wl  : (100,) wavelength axis in nm
    sub : sub-sampling factor for DTW
    """
    return {
        "MSE":     float(np.mean((yt - yp) ** 2)),
        "MAE":     float(np.mean(np.abs(yt - yp))),
        "CosSim":  float(cosine_sim(yt, yp).mean()),
        "SAM_deg": float(sam(yt, yp).mean()),
        "DTW":     _compute_dtw(yt, yp, sub),
        "PkWL_nm": _peak_wl_err(yt, yp, wl),
        "BW_nm":   _bw_err(yt, yp, wl),
        "MaxPW":   float(np.max(np.abs(yt - yp), 0).mean()),
        "PW_Corr": _pw_corr(yt, yp),
    }


def print_fwd_spectrum_table(results: dict) -> None:
    """Pretty-print forward-spectrum results table (no inference time)."""
    hdr = (f"{'Model':<22}{'MSE':>9}{'MAE':>9}{'CosSim':>9}"
           f"{'SAM°':>8}{'DTW':>9}{'PkWL_nm':>9}{'BW_nm':>9}{'PW_Corr':>9}")
    print(f"\n{'─'*92}")
    print(hdr)
    print('─' * 92)
    for mn, res in results.items():
        m = res["metrics"]
        print(
            f"{mn:<22}"
            f"{m['MSE']:>9.5f}{m['MAE']:>9.5f}"
            f"{m['CosSim']:>9.4f}{m['SAM_deg']:>8.2f}"
            f"{m['DTW']:>9.3f}{m['PkWL_nm']:>9.2f}"
            f"{m['BW_nm']:>9.2f}{m['PW_Corr']:>9.4f}"
        )


# ── Tasks 3 & 4: Inverse ─────────────────────────────────────────────────────

def metrics_inverse(
    yt: np.ndarray,
    yp: np.ndarray,
    tol_s: float = 0.05,
    tol_r: float = 0.10,
) -> dict:
    """
    Compute all inverse-design metrics.

    Parameters
    ----------
    yt    : (N, 5) ground-truth geometry
    yp    : (N, 5) predicted geometry
    tol_s : strict tolerance (default 5%)
    tol_r : relaxed tolerance (default 10%)
    """
    return {
        "MAE":          mean_absolute_error(yt, yp, multioutput="raw_values"),
        "RMSE":         np.sqrt(mean_squared_error(yt, yp, multioutput="raw_values")),
        "MedAE":        np.median(np.abs(yt - yp), axis=0),
        "MRE":          np.mean(np.abs(_sdiv(yt - yp, yt)), axis=0),
        "SR_strict":    success_rate(yt, yp, tol_s),
        "SR_relaxed":   success_rate(yt, yp, tol_r),
        "SR_per_param": sr_per_param(yt, yp, tol_s),
    }


def condition_number_proxy(X: np.ndarray, Y: np.ndarray, n: int = 300) -> float:
    """
    Local ill-posedness proxy: median ||ΔX|| / ||ΔY|| over nearest-neighbour pairs.
    Small values indicate that nearby spectra map to very different geometries.
    """
    n = min(n, len(X))
    idx_ = np.random.choice(len(X), n, replace=False)
    X_, Y_ = X[idx_], Y[idx_]
    nn = NearestNeighbors(n_neighbors=2).fit(X_)
    _, ids = nn.kneighbors(X_)
    dX = X_[ids[:, 1]] - X_
    dY = Y_[ids[:, 1]] - Y_
    return float(np.median(np.linalg.norm(dX, axis=1) / (np.linalg.norm(dY, axis=1) + 1e-8)))


def metrics_inv_spectrum(
    yt: np.ndarray,
    yp: np.ndarray,
    y_spec_input: np.ndarray,
    tol_s: float = 0.05,
    tol_r: float = 0.10,
) -> dict:
    """Inverse-spectrum metrics (extends metrics_inverse with CondNum)."""
    m = metrics_inverse(yt, yp, tol_s, tol_r)
    m["CondNum"] = condition_number_proxy(y_spec_input, yp)
    return m


def print_inverse_table(results: dict, task: str = "Inverse") -> None:
    """Pretty-print inverse results table (no inference time)."""
    hdr = (f"{'Model':<18}{'SR_strict':>11}{'SR_relax':>10}"
           f"{'MAE':>9}{'RMSE':>9}{'MedAE':>9}{'MRE':>9}")
    print(f"\n{'─'*78}")
    print(f"  {task}")
    print('─' * 78)
    print(hdr)
    print('─' * 78)
    for mn, res in results.items():
        m = res["metrics"]
        cond = f"{m.get('CondNum', float('nan')):>10.3f}" if "CondNum" in m else ""
        print(
            f"{mn:<18}"
            f"{m['SR_strict']:>11.1f}"
            f"{m['SR_relaxed']:>10.1f}"
            f"{m['MAE'].mean():>9.4f}"
            f"{m['RMSE'].mean():>9.4f}"
            f"{m['MedAE'].mean():>9.4f}"
            f"{m['MRE'].mean():>9.4f}"
            f"{cond}"
        )
