"""Shared SignalResult contract — frozen after hour 1. Do not edit without syncing all workstreams."""
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class SignalResult:
    signal_name: str
    score: float                          # 0.0-1.0, higher = more suspicious
    applicable: bool                      # False if preconditions unmet (no face, no glint, bad EXIF, etc.)
    evidence_image: Optional[np.ndarray]  # BGR image for the UI gallery, or None
    note: str                             # short human-readable explanation
