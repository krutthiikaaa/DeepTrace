# KYC-Lens — Implementation Plan

## Context

24-hour, 5-person hackathon build. An earlier design proposed fine-tuning two ViT models on FaceForensics++/Celeb-DF/140k-faces — rejected as overscoped for this timeline (FF++ official access has a ~1 week approval lag; fine-tuning/eval needs a labeled dataset pipeline no 24hr team can safely build and validate). Adopted instead: five classical CV forensic signals, no training, no GPU, CPU-only, millisecond inference, plus one pretrained (not fine-tuned) deep-learning signal added to patch the main coverage gap against diffusion-generated images.

Goal: an auditable "forensic lab," not a black-box score — a 4-way verdict (**Genuine / AI-Generated / Face-Swapped / Edited**) plus an honest **Uncertain — Route to Tier-2 Review** state, backed by visual evidence (heatmaps, eye crops, spectra) a KYC compliance reviewer can actually inspect.

The build is split into **5 fully independent, parallelizable workstreams** with a frozen shared interface contract defined up front, so all 5 people can code, test, and merge without blocking each other for the entire 24 hours.

## Why classical-first, not learned-first

- No dataset-access or licensing friction.
- No GPU dependency; every signal runs in milliseconds on a laptop CPU.
- The heatmaps/crops **are** the forensic evidence — not a Grad-CAM approximation of a black box.
- Known gap: FFT-based artifact detection targets GAN-era upsampling checkerboard patterns; modern diffusion models don't reliably produce that signature. Patched by the stretch signal (workstream C1) rather than over-engineering FFT heuristics in one day.

## Shared Contract — write this FIRST, before splitting up (Hour 0-1, all 5 people together)

Everything else depends on this being stable. One 30-60 min joint session defines it, then it's frozen — changing it later requires a quick sync since all 5 workstreams depend on it. See `shared/types.py` for the `SignalResult` dataclass and `configs/thresholds.yaml` for the tunable verdict thresholds/weights.

Every signal module exposes exactly one pure function: `def analyze(image: np.ndarray) -> SignalResult`. No global state, no hidden I/O, no side effects — this is what makes each signal independently testable and swappable.

Shared repo conventions agreed in this same session: `ruff` for lint/format (run before every commit), type hints on every function signature, snake_case for files/functions, PascalCase for dataclasses, `logging` module (not `print`) for anything beyond a signal's own standalone CLI test.

**No two people ever edit the same file.** `shared/types.py` and `configs/thresholds.yaml` are the only shared touchpoints, and both are frozen/append-only after hour 1.

## The 5 Workstreams

### Workstream A — Frequency & Compression Forensics
- **Owns**: `signals/fft_signal.py`, `signals/ela_signal.py`, `tests/test_fft_signal.py`, `tests/test_ela_signal.py`
- **FFT Frequency Spectrum**: grayscale -> 2D FFT -> log-magnitude spectrum; score from periodic peak energy in the radial/azimuthal profile (checkerboard grid = GAN upsampling artifact). Evidence: the spectrum image, peaks annotated.
- **ELA**: re-save at fixed JPEG quality (90), pixel-diff vs. original, amplify, connected-component analysis for localized hot blobs (spliced regions differ from uniform global recompression noise). Evidence: the ELA heatmap.
- Each function: pure, `image: np.ndarray -> SignalResult`, no disk/network I/O inside the function itself.
- Unit tests: 2-3 fixture images (one clean photo, one known-GAN sample, one JPEG-edited sample) asserting score direction, not exact values.

### Workstream B — Biometric & Sensor-Noise Forensics
- **Owns**: `signals/pupil_signal.py`, `signals/noise_signal.py`, `tests/test_pupil_signal.py`, `tests/test_noise_signal.py`
- **Corneal Pupil Reflections**: MediaPipe Face Mesh (iris refinement) locates both eyes; find brightest specular glint per eye; compare glint position/size/color left vs. right. Return `applicable=False` (never force a score) when an eye/glint isn't detectable - sunglasses, extreme angle, glare.
- **Noise Consistency**: high-frequency residual (median-filter subtraction) noise variance, face region vs. background/border region - face-swap composites often mismatch here at the blend boundary.
- Same purity/testability rules as Workstream A.

### Workstream C1 — Quality Gate & Stretch ML Signal
- **Owns**: `signals/exif_blur_gate.py`, `signals/synthetic_vit_signal.py` (stretch), `tests/test_exif_blur_gate.py`, `tests/test_synthetic_vit_signal.py`
- **EXIF & Blur/Quality Gate**: EXIF editing-software tags / missing camera metadata, variance-of-Laplacian blur score, resolution floor. A **quality gate**, not a fraud category - its output tells `scoring.py` (Workstream C2) how much to trust the other signals and can force Uncertain on unusable input.
- **Stretch - `synthetic_vit_signal.py`**: `buildborderless/CommunityForensics-DeepfakeDet-ViT`, pretrained-only (no fine-tuning), lazy-imports `torch`/`transformers`/`timm` behind try/except so its absence never breaks the core pipeline. Attempted once the core signals are integrating cleanly (checkpoint at hour 14). Pre-download the model checkpoint before the event.

### Workstream C2 — Aggregation, Pipeline & Integration Backend
- **Owns**: `scoring.py`, `pipeline.py`, `configs/thresholds.yaml`, `requirements.txt`, `tests/test_scoring.py`, `tests/test_pipeline.py`
- **`scoring.py`**: maps signals to categories (weighted average over `applicable` signals only), applies verdict logic:
  - Quality gate fails, or <2 signals applicable -> **Uncertain - Route to Tier-2 Review**
  - `max(category) >= HIGH_THRESHOLD` and beats runner-up by `>= MARGIN` -> that category
  - all categories `< LOW_THRESHOLD` -> **Genuine**
  - else -> **Uncertain** (conflicting signals - honest uncertainty, never force a guess)
- **`pipeline.py`**: orchestrates load image -> run each signal (from A, B, and C1) wrapped in its own try/except (one bad signal must never crash the whole request) -> `scoring.py` -> result dict. Exposes `def run(image_path: str) -> dict` as the one function everyone else (Workstream D) calls.
- **Critical early deliverable**: stub `pipeline.py`/`scoring.py` with fake `SignalResult`s in hour 0-2 - this is what lets A, B, C1, and D all integrate against a working (if fake) backend immediately, rather than waiting on real signals.

### Workstream D — UI, Evidence & Demo
- **Owns**: `app.py`, `test_images/`, `eval/run_eval.py`, `docs/demo_script.md`
- **`app.py`**: Gradio dashboard calling `pipeline.run()` - verdict badge, evidence gallery (FFT spectrum, ELA heatmap, eye-glint crop comparison, noise map), confidence bar.
- Collects ~15-30 test images (real selfies, known AI-generated faces, face-swap samples, Photoshopped/edited images), roughly labeled - used both for the live demo and as `eval/run_eval.py`'s validation set.
- `eval/run_eval.py`: runs `test_images/` through `pipeline.run()`, prints per-category hit/miss.

## Git Workflow

- One branch per workstream: `feature/a-frequency`, `feature/b-biometric`, `feature/c1-quality-ml`, `feature/c2-scoring-pipeline`, `feature/d-ui`.
- `main` must always run end-to-end - Workstream C2's stubbed `pipeline.py` from hour 0-2 guarantees this even before real signals land.
- Small, frequent commits; merge to `main` at the two integration checkpoints (hour 10, hour 14) rather than one big merge at the end.
- Whoever touches `shared/types.py` or `configs/thresholds.yaml` pings the other 4 - these are the only shared-state risk.

## Timeline

- **Hr 0-1**: all 5 together - agree the `SignalResult` contract, `thresholds.yaml` schema, repo/lint conventions.
- **Hr 1-2**: Workstream C2 stubs `pipeline.py`/`scoring.py` with fake signals; Workstream D wires `app.py` against the stub.
- **Hr 2-10**: A, B, and C1 build their real signals in parallel, each testable standalone via `python -m signals.fft_signal path/to.jpg`.
- **Hr 10 (checkpoint)**: merge A + B + C1's real signals into `main`, replacing C2's stubs.
- **Hr 10-14**: C2 wires real signals into `scoring.py`/`pipeline.py`; D polishes UI against real output.
- **Hr 14 (checkpoint)**: C1 attempts stretch `synthetic_vit_signal.py`; run `eval/run_eval.py` against `test_images/`, C2 hand-tunes `configs/thresholds.yaml`.
- **Hr 14-22**: polish, edge cases (no face, corrupt upload, sunglasses -> correctly Uncertain), pitch prep.
- **Hr 22-24**: rehearse demo, capture fallback screenshots, buffer.

## Best practices baked in

- **Interface-first design**: the `SignalResult` contract is agreed and frozen before any signal code is written.
- **Pure functions**: every signal is `image -> SignalResult`, no hidden state - trivially unit-testable in isolation and swappable without touching the pipeline.
- **Fail-soft, never fail-hard**: `applicable: False` instead of exceptions for missing preconditions; `pipeline.py` wraps each signal call in try/except so one broken signal never crashes a live demo request.
- **Config over hardcoding**: thresholds/weights live in `configs/thresholds.yaml`, tunable under time pressure without touching signal code.
- **Feature-flagged stretch work**: the deep-learning signal is fully optional and isolated (lazy import, try/except).
- **Stub-first integration**: the backend is stubbed end-to-end in hour 1-2 so UI and integration work start immediately.
- **Ownership boundaries = merge-conflict avoidance**: file-level ownership per workstream, with only two frozen/append-only shared files.

## Secrets, Credentials & Sensitive Data

- **No paid/authenticated API is required for the MVP.** `buildborderless/CommunityForensics-DeepfakeDet-ViT` is a public, ungated Hugging Face checkpoint - no API token needed. If a gated/private model is ever swapped in, load a `HUGGINGFACE_TOKEN` from the environment, never hardcode it.
- **`.env` + `.gitignore` from hour 0**: any credential goes in a local `.env`, loaded via `python-dotenv`, never committed. Commit only `.env.example` with placeholder keys.
- **`.gitignore` includes**: `.env`, `models/`, `data/raw/`, and any runtime upload cache - none of these belong in git history.
- **Test images are not real KYC documents.** `test_images/` must be public/synthetic sources only - never a real person's actual ID card, passport, or personal selfie without consent.
- **Don't persist uploaded images beyond the request.** Process in memory, discard after returning a verdict.
- **`share=True` in Gradio** exposes the endpoint publicly for as long as it runs - turn it off after a supervised demo window.
- **Pitch angle**: "no external API calls, everything runs on-device" is itself a KYC-relevant compliance advantage - no biometric image data ever leaves the machine.

## Verification plan

- Each signal script runnable standalone (`python -m signals.<name> path/to.jpg`) printing its `SignalResult`.
- `tests/` - pytest per signal/scoring/pipeline module, runnable independently by each workstream.
- `eval/run_eval.py` - full-pipeline pass over `test_images/`, per-category accuracy.
- Manual smoke test: launch `app.py`, upload one clearly-real and one clearly-fake sample; upload a sunglasses/blurry photo and confirm it lands in Uncertain.

## Resilience notes (hackathon-specific)

- Every signal returns `applicable: false` gracefully on missing preconditions - never throws.
- The stretch ViT signal is fully optional/isolated; pre-download its checkpoint before the event; the core 5 signals must demo correctly with it disabled.
- Thresholds live in config, not code, so last-minute tuning needs no redeploy.
