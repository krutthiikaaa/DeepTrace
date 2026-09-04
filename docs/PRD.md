# PRD: KYC-Lens — Explainable Face & Deepfake Forensics for KYC

## 1. Problem Statement

KYC (Know Your Customer) onboarding relies on photo verification (selfie vs. ID, liveness checks) that is increasingly vulnerable to three distinct fraud vectors: **fully AI-generated synthetic faces**, **face-swap deepfakes**, and **manually edited/spliced photos**. Existing black-box fraud scores give compliance teams a number with no way to audit *why* an image was flagged — both a trust problem (reviewers can't verify the model) and a compliance problem (regulators increasingly expect explainable decisioning). Most teams also can't deploy GPU-heavy learned models at photo-verification volume without cost/latency tradeoffs.

## 2. Goals

- Produce a **4-way verdict** — Genuine / AI-Generated / Face-Swapped / Edited — plus an honest **Uncertain / Route to Tier-2 Review** state, instead of a bare probability score.
- Every verdict is backed by **inspectable visual evidence** (frequency spectra, ELA heatmaps, eye-glint crops, noise maps) a human reviewer can independently judge.
- Run at **CPU-only, millisecond latency**, no GPU, no training pipeline, no dataset-licensing dependency.
- Ship a working, demoable system within a **24-hour, 5-person hackathon window**.

## 3. Non-Goals

- Not a production-grade, adversarially-hardened fraud system — this is a hackathon MVP demonstrating the approach, not a hardened deployment.
- Not attempting full liveness detection (blink/motion challenge-response) — static single-image forensics only.
- Not training or fine-tuning any model — the one deep-learning component used is pretrained, inference-only, and fully optional.
- Not handling video input in v1 — single still images only.

## 4. Target Users

- **KYC compliance reviewer**: needs a verdict plus evidence they can visually confirm or override, not a black box.
- **Fraud/product team evaluating the tool**: needs to see it work against a small labeled test set, not just a live demo.
- **Hackathon judges**: need to understand the method's reasoning and honest limitations in a 2-3 minute pitch.

## 5. User Stories

1. *As a reviewer*, I upload a photo and get a clear verdict badge (Genuine/AI-Generated/Face-Swapped/Edited/Uncertain) so I know what action to take.
2. *As a reviewer*, I see the specific evidence (heatmap, eye crop, spectrum) that produced the verdict, so I can sanity-check the system's reasoning before acting on it.
3. *As a reviewer*, when the photo is low-quality, blurry, or ambiguous, I get routed to **Uncertain / Tier-2 Review** rather than a falsely confident answer — false confidence is worse than admitted uncertainty.
4. *As a product owner*, I can see aggregate accuracy against a labeled test set, not just anecdotal demo photos, so I can gauge whether this is production-track-worthy.

## 6. Functional Requirements

| # | Requirement | Signal(s) |
|---|---|---|
| FR1 | Detect frequency-domain artifacts characteristic of GAN upsampling | FFT spectrum |
| FR2 | Detect corneal specular-reflection mismatch between eyes | Pupil/iris (MediaPipe) |
| FR3 | Detect localized recompression artifacts from splicing/editing | ELA |
| FR4 | Detect sensor/noise-pattern mismatch between face and background | Noise consistency |
| FR5 | Gate all of the above on image quality (blur, resolution, EXIF plausibility) | EXIF/blur gate |
| FR6 (stretch) | Detect diffusion-generated images that FFT alone misses | Pretrained ViT (inference-only) |
| FR7 | Aggregate all applicable signals into one of 5 verdicts with a stated confidence | Scoring/verdict logic |
| FR8 | Render verdict + full evidence gallery in a web UI | Gradio dashboard |
| FR9 | Run the full pipeline against a labeled test set and report per-category accuracy | Eval script |

## 7. Non-Functional Requirements

- **Latency**: sub-second end-to-end on a laptop CPU (excluding the optional stretch model's first cold load).
- **Reliability**: no single signal's failure (missing face, corrupt EXIF, no glint) may crash the request — always degrade to `applicable: false`, never throw.
- **Explainability**: every verdict must be traceable to specific, human-viewable evidence — no opaque scores presented alone.
- **Configurability**: verdict thresholds/weights must be tunable via config file, not code changes, so they can be calibrated against test data under time pressure.
- **Modularity**: each forensic signal must be independently runnable, testable, and swappable behind one shared interface, so 5 people can build in parallel with zero file-level conflicts.

## 8. Scope

**MVP (must-ship)**: FR1-5, FR7, FR8 — the 5 classical signals, verdict aggregation, and the demo UI. This alone is a complete, coherent product.

**Stretch (nice-to-have, cut first if behind schedule)**: FR6 (pretrained diffusion-detection signal), FR9 polish beyond a basic console report.

## 9. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| FFT signal misses diffusion-era fakes (no checkerboard artifact) | Documented as a known limitation; FR6 stretch signal patches it if time allows |
| Pupil/glint signal fails on sunglasses/side-angle/low-res photos | Never forces a score — returns `applicable: false`, correctly reduces confidence rather than guessing |
| Live demo gets an unexpected/adversarial upload | Fail-soft design (see NFR reliability) plus pre-captured fallback screenshots |
| Venue wifi fails during the stretch model's checkpoint download | Pre-download the model before the event; core MVP has zero network dependency at runtime |
| 5 people produce conflicting/incompatible code | Shared `SignalResult` interface frozen in hour 1; strict file-level ownership per workstream |

## 10. Success Metrics (for this hackathon)

- All 5 MVP signals integrated and producing a verdict end-to-end by hour 14.
- `eval/run_eval.py` shows correct-direction verdicts on a majority of the ~15-30 collected labeled test images (a hackathon-scale sanity check, not a formal accuracy bar).
- Demo runs live without crashing on at least one deliberately "hard" test image (sunglasses, low light, edited photo) and correctly returns Uncertain rather than a wrong confident verdict.

## 11. Team & Timeline

Maps directly to the 5 parallel workstreams (Frequency & Compression / Biometric & Noise / Quality Gate & Stretch ML / Aggregation & Pipeline / UI & Evidence) and the hour-by-hour build schedule in [PLAN.md](PLAN.md).

## 12. Out of Scope / Future Work

Video/liveness challenge-response, adversarial robustness testing, production deployment hardening, fine-tuning the stretch model on FaceForensics++/Celeb-DF once dataset access clears, multi-language EXIF/metadata edge cases.
