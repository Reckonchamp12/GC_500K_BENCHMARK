"""
GC-Bench · Visualization Utilities
=====================================
All plotting helpers used across benchmark reports.
Both light and dark themes are supported.

Figures produced
----------------
  A  R² heatmap                  (fwd scalar)
  B  Hexbin predicted-vs-true    (fwd scalar · ResNet)
  C  Violin MAE per target       (fwd scalar)
  D  Radar chart                 (fwd scalar)
  E  Pareto R² vs time           (fwd scalar)
  F  Spectral gallery            (fwd spectrum)
  G  Per-wavelength MAE heatmap  (fwd spectrum)
  H  Metrics radar               (fwd spectrum)
  I  Learning curves             (fwd spectrum)
  J  CosSim vs DTW scatter       (fwd spectrum)
  K  Per-parameter MAE heatmap   (inv scalar)
  L  Per-parameter SR% heatmap   (inv scalar)
  M  SR vs tolerance curves      (inv scalar)
  N  Strict vs relaxed SR        (inv scalar)
  O  Pareto SR vs time           (inv scalar)
  P  SR lollipop + SR heatmap    (inv spectrum)
  Q  SR vs MAE scatter           (inv spectrum)
  R  Noise robustness curves     (inv spectrum)
  S  Per-param MAE heatmap       (inv spectrum)
  T  Pareto SR vs time           (inv spectrum)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

from .metrics import cosine_sim

# ── Theme definitions ─────────────────────────────────────────────────────────

THEMES: Dict[str, Dict] = {
    "light": {
        "bg": "#FAFAFA", "fig_bg": "#FFFFFF", "fg": "#1a1a2e",
        "grid": "#E8E8E8", "spine": "#CCCCCC",
        "cmap": "viridis", "cmap_div": "RdBu_r",
        "pal": ["#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#F4A261",
                "#264653", "#8338EC", "#06D6A0", "#FB8500", "#023047",
                "#A8DADC", "#457B9D"],
    },
    "dark": {
        "bg": "#0d1117", "fig_bg": "#161b22", "fg": "#E6EDF3",
        "grid": "#21262d", "spine": "#30363d",
        "cmap": "plasma", "cmap_div": "coolwarm",
        "pal": ["#FF6B6B", "#74C0FC", "#63E6BE", "#FFD43B", "#FFA94D",
                "#A9E34B", "#CC5DE8", "#20C997", "#FF922B", "#339AF0",
                "#F06595", "#748FFC"],
    },
}


def _apply_theme(fig, axes, th: Dict) -> None:
    fig.patch.set_facecolor(th["fig_bg"])
    for ax in np.array(axes).flatten():
        if ax is None:
            continue
        ax.set_facecolor(th["bg"])
        ax.tick_params(colors=th["fg"], which="both")
        ax.xaxis.label.set_color(th["fg"])
        ax.yaxis.label.set_color(th["fg"])
        ax.title.set_color(th["fg"])
        for sp in ax.spines.values():
            sp.set_edgecolor(th["spine"])
        ax.grid(True, color=th["grid"], alpha=0.45, lw=0.6)
        ax.set_axisbelow(True)


def save_figure(name: str, th_name: str, fig, figdir: Path) -> None:
    p = figdir / f"{name}_{th_name}.png"
    fig.savefig(p, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  → {p.name}")


# ── A: R² heatmap ─────────────────────────────────────────────────────────────

def plot_r2_heatmap(results: dict, scalar_names: List[str], figdir: Path) -> None:
    for thn, th in THEMES.items():
        models_ = list(results.keys())
        r2mat   = np.array([results[mn]["metrics"]["R2"] for mn in models_])
        fig, ax = plt.subplots(figsize=(10, max(4, len(models_) * 0.55)))
        im = ax.imshow(r2mat, cmap=th["cmap"], vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(scalar_names)))
        ax.set_xticklabels(scalar_names, rotation=30, ha="right", color=th["fg"], fontsize=9)
        ax.set_yticks(range(len(models_)))
        ax.set_yticklabels(models_, color=th["fg"], fontsize=9)
        for i in range(len(models_)):
            for j in range(len(scalar_names)):
                v = r2mat[i, j]
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8,
                        color="white" if v < 0.6 else "black")
        plt.colorbar(im, ax=ax, label="R²")
        ax.set_facecolor(th["bg"]); fig.patch.set_facecolor(th["fig_bg"])
        ax.set_title("Forward Scalar — Per-Model R² Heatmap",
                     color=th["fg"], fontsize=13, fontweight="bold")
        ax.tick_params(colors=th["fg"]); plt.tight_layout()
        save_figure("A_fwd_scalar_r2_heatmap", thn, fig, figdir)


# ── F: Spectral gallery ───────────────────────────────────────────────────────

def plot_spectral_gallery(
    wl: np.ndarray,
    Ysp_te: np.ndarray,
    pred_sp: np.ndarray,
    figdir: Path,
    n_panels: int = 12,
    title: str = "Spectral Prediction Gallery",
) -> None:
    rng  = np.random.default_rng(0)
    idxs = rng.choice(len(Ysp_te), n_panels, replace=False)
    nr, nc = (n_panels + 3) // 4, 4

    for thn, th in THEMES.items():
        fig, axes = plt.subplots(nr, nc, figsize=(18, nr * 3.5))
        fig.patch.set_facecolor(th["fig_bg"])
        fig.suptitle(title, fontsize=14, color=th["fg"], fontweight="bold", y=1.01)

        for i, ax in enumerate(axes.flatten()):
            idx = idxs[i]; ax.set_facecolor(th["bg"])
            ax.plot(wl, Ysp_te[idx], color=th["pal"][0], lw=2, label="True", zorder=3)
            ax.plot(wl, pred_sp[idx], color=th["pal"][1], lw=1.5, ls="--", label="Pred", zorder=4, alpha=0.9)
            ax.fill_between(wl, Ysp_te[idx], pred_sp[idx], color=th["pal"][2], alpha=0.12)
            ax.set_xlim(wl[0], wl[-1]); ax.set_ylim(-0.05, 1.1)
            ax.set_xlabel("λ (nm)", fontsize=8, color=th["fg"])
            ax.set_ylabel("T", fontsize=8, color=th["fg"])
            ax.tick_params(colors=th["fg"], labelsize=7)
            for sp in ax.spines.values():
                sp.set_edgecolor(th["spine"])
            ax.grid(True, color=th["grid"], alpha=0.35, lw=0.5)
            cos_ = float(cosine_sim(Ysp_te[idx:idx+1], pred_sp[idx:idx+1]))
            ax.set_title(f"#{idx}  CosSim={cos_:.3f}", fontsize=8, color=th["fg"], pad=4)
            if i == 0:
                ax.legend(fontsize=7, facecolor=th["bg"], labelcolor=th["fg"], edgecolor=th["spine"])
        plt.tight_layout()
        save_figure("F_spec_gallery", thn, fig, figdir)


# ── Learning curves ────────────────────────────────────────────────────────────

def plot_learning_curves(
    curves: Dict[str, Dict],
    prefix: str,
    title: str,
    fig_name: str,
    figdir: Path,
) -> None:
    keys_ = [(k, v) for k, v in curves.items() if v.get("tr") and len(v["tr"]) > 2 and k.startswith(prefix)]
    if not keys_:
        return
    n = len(keys_); nr = max(1, (n + 2) // 3); nc = 3

    for thn, th in THEMES.items():
        fig, axes = plt.subplots(nr, nc, figsize=(16, nr * 4))
        axes_f    = np.array(axes).flatten()
        fig.patch.set_facecolor(th["fig_bg"])

        for i, (k, v) in enumerate(keys_):
            ax = axes_f[i]; ax.set_facecolor(th["bg"])
            ep = range(1, len(v["tr"]) + 1)
            ax.semilogy(ep, v["tr"], color=th["pal"][0], lw=2, label="Train")
            ax.semilogy(ep, v["va"], color=th["pal"][1], lw=2, label="Val", linestyle="--")
            ax.set_xlabel("Epoch", color=th["fg"], fontsize=8)
            ax.set_ylabel("Loss",  color=th["fg"], fontsize=8)
            ax.set_title(k.replace(prefix, ""), color=th["fg"], fontsize=10, fontweight="bold")
            ax.tick_params(colors=th["fg"], labelsize=7)
            for sp in ax.spines.values():
                sp.set_edgecolor(th["spine"])
            ax.grid(True, color=th["grid"], alpha=0.4)
            if i == 0:
                ax.legend(fontsize=8, facecolor=th["bg"], labelcolor=th["fg"], edgecolor=th["spine"])

        for i in range(len(keys_), len(axes_f)):
            axes_f[i].set_visible(False)

        fig.suptitle(title, color=th["fg"], fontsize=14, fontweight="bold")
        plt.tight_layout()
        save_figure(fig_name, thn, fig, figdir)


# ── Noise robustness curves ───────────────────────────────────────────────────

def plot_noise_robustness(noise_results: dict, figdir: Path) -> None:
    sigs  = [v["sigma"] for v in noise_results.values()]
    srs_  = [v["SR"]    for v in noise_results.values()]
    maes_ = [v["MAE"]   for v in noise_results.values()]

    for thn, th in THEMES.items():
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.patch.set_facecolor(th["fig_bg"])
        for ax, y_, ylabel, title in zip(
            axes,
            [srs_, maes_],
            ["SR (%)", "MAE"],
            ["Success Rate vs Noise σ", "MAE vs Noise σ"],
        ):
            ax.set_facecolor(th["bg"])
            ax.plot(sigs, y_, color=th["pal"][0], lw=2.5, marker="o", markersize=8, zorder=5)
            ax.fill_between(sigs, y_, alpha=0.15, color=th["pal"][0])
            ax.set_xlabel("Noise σ (added to spectrum)", color=th["fg"])
            ax.set_ylabel(ylabel, color=th["fg"])
            ax.set_title(title, color=th["fg"], fontsize=12, fontweight="bold")
            ax.tick_params(colors=th["fg"])
            for sp in ax.spines.values():
                sp.set_edgecolor(th["spine"])
            ax.grid(True, color=th["grid"], alpha=0.4)
        plt.tight_layout()
        save_figure("R_noise_robustness", thn, fig, figdir)
