"""
GC-Bench
========
A comprehensive ML benchmark for silicon grating coupler inverse design.

Quick start
-----------
>>> from gc_bench import CFG, load_data
>>> cfg  = CFG(data_path="/path/to/gc_500k.h5")
>>> data = load_data(cfg)
"""

from .config   import CFG
from .data     import load_data
from .metrics  import (
    metrics_fwd_scalar,
    metrics_fwd_spectrum,
    metrics_inverse,
    metrics_inv_spectrum,
    print_fwd_scalar_table,
    print_fwd_spectrum_table,
    print_inverse_table,
    cosine_sim,
    sam,
    success_rate,
    sr_per_param,
)
from .training import (
    wrap,
    make_loader,
    train_torch,
    predict,
    free_memory,
    DeepEnsemble,
    DEVICE,
    N_GPUS,
    print_device_info,
)

__version__ = "1.0.0"
__all__ = [
    "CFG", "load_data",
    "metrics_fwd_scalar", "metrics_fwd_spectrum",
    "metrics_inverse", "metrics_inv_spectrum",
    "print_fwd_scalar_table", "print_fwd_spectrum_table", "print_inverse_table",
    "cosine_sim", "sam", "success_rate", "sr_per_param",
    "wrap", "make_loader", "train_torch", "predict", "free_memory",
    "DeepEnsemble", "DEVICE", "N_GPUS", "print_device_info",
]
