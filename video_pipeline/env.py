"""Environment configuration for the video colorization pipeline."""

from __future__ import annotations

import multiprocessing as mp
import os

import torch

LOGICAL = mp.cpu_count() or 8

# Configure BLAS/OpenMP thread counts
os.environ["OMP_NUM_THREADS"] = str(LOGICAL)
os.environ["MKL_NUM_THREADS"] = str(LOGICAL)
os.environ["OPENBLAS_NUM_THREADS"] = str(LOGICAL)
os.environ["NUMEXPR_NUM_THREADS"] = str(LOGICAL)

# Configure PyTorch threading
torch.set_num_threads(LOGICAL // 2)
torch.set_num_interop_threads(LOGICAL // 2)

try:
    import intel_extension_for_pytorch as _ipex  # type: ignore[import-not-found]

    HAS_IPEX = True
except ImportError:  # pragma: no cover - optional dependency
    _ipex = None
    HAS_IPEX = False

# Re-export the optional module for consumers that need it.
ipex = _ipex

__all__ = ["LOGICAL", "HAS_IPEX", "ipex"]
