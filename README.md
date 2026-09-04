# DeepTrace — KYC-Lens

Explainable Face & Deepfake Forensics for KYC photo verification. A 24-hour hackathon build producing a 4-way verdict (**Genuine / AI-Generated / Face-Swapped / Edited**) plus an honest **Uncertain — Route to Tier-2 Review** state — backed by inspectable visual evidence (FFT spectra, ELA heatmaps, eye-glint crops, noise maps), not a black-box score.

See [docs/PRD.md](docs/PRD.md) for the product requirements and [docs/PLAN.md](docs/PLAN.md) for the full implementation plan.

## Workstreams & Ownership

| Workstream | Branch | Owns |
|---|---|---|
| A — Frequency & Compression Forensics | `feature/a-frequency` | `signals/fft_signal.py`, `signals/ela_signal.py` |
| B — Biometric & Sensor-Noise Forensics | `feature/b-biometric` | `signals/pupil_signal.py`, `signals/noise_signal.py` |
| C1 — Quality Gate & Stretch ML Signal | `feature/c1-quality-ml` | `signals/exif_blur_gate.py`, `signals/synthetic_vit_signal.py` |
| C2 — Aggregation, Pipeline & Integration Backend | `feature/c2-scoring-pipeline` | `scoring.py`, `pipeline.py`, `configs/thresholds.yaml`, `requirements.txt` |
| D — UI, Evidence & Demo | `feature/d-ui` | `app.py`, `test_images/`, `eval/run_eval.py`, `docs/demo_script.md` |

**Rule: no two people ever edit the same file.** `shared/types.py` and `configs/thresholds.yaml` are the only shared touchpoints — both frozen/append-only after hour 1 (see docs/PLAN.md). Work on your own `feature/<workstream>` branch; merge to `main` at the hour-10 and hour-14 checkpoints.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # only needed if a gated model is added later
```

## Run

```bash
python app.py
```

## Test

```bash
pytest tests/
```

Each signal is also runnable standalone for self-testing without the full pipeline:

```bash
python -m signals.fft_signal path/to/image.jpg
```
