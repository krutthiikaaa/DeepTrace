"""
Workstream C2 — orchestration entrypoint.
run() is the single function every other workstream (especially Workstream D's
app.py) calls. Stub this early (hour 0-2) with fake SignalResults so the other
workstreams can integrate against a working backend immediately.
"""
from typing import Dict


def run(image_path: str) -> Dict:
    """
    TODO (Workstream C2): load image -> call each signals.*.analyze(), each
    wrapped in its own try/except (one bad signal must never crash the whole
    request) -> scoring.aggregate() -> return result dict.
    """
    raise NotImplementedError("Workstream C2: implement pipeline orchestration")
