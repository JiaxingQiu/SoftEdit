"""Post-training calibration baselines (batch-only editors).

    ts      — Temperature Scaling (Guo et al., ICML 2017)
    bc      — Batch Calibration (Zhou et al., ICLR 2024)
    ats     — Adaptive Temperature Scaling (Xie et al., EMNLP 2024)
    ts_ret  — per-edit temperature, nearest-key retrieval
    bc_ret  — per-edit first-token distribution, nearest-key retrieval
"""
from .ts import TSSoft
from .bc import BCSoft
from .ats import ATSSoft
from .ts_ret import TSRetSoft
from .bc_ret import BCRetSoft

__all__ = ["TSSoft", "BCSoft", "ATSSoft", "TSRetSoft", "BCRetSoft"]
