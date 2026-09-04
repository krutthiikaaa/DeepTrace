"""
Workstream C2 — verdict aggregation.
Maps SignalResults into category scores (ai_generated / face_swapped / edited)
and applies the verdict logic described in docs/PLAN.md.
"""
from typing import Dict, List

from shared.types import SignalResult


def aggregate(signal_results: List[SignalResult], thresholds: Dict) -> Dict:
    """
    TODO (Workstream C2):
    - weighted average per category over applicable signals only
    - quality gate fail or <2 applicable signals -> Uncertain
    - max category >= high_threshold and beats runner-up by >= margin -> that category
    - all categories < low_threshold -> Genuine
    - else -> Uncertain
    See docs/PLAN.md "Scoring & Verdict Logic" for the full spec.
    """
    raise NotImplementedError("Workstream C2: implement aggregation + verdict logic")
