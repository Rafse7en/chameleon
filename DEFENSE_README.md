# Chameleon Defense: quality-gated face-local experiment

This is a local experimental tool for reducing similarity in two face-feature models while keeping the photo visually useful. It is **not a proven anti-training or anti-generation defense**. No result guarantees protection from future models, face swaps, reference-image generation, LoRA/DreamBooth, fine-tuning, denoising, or commercial services.

Double-click `run_defense.bat`, choose a mode, and select photos. You can also drag photos onto the launcher. Processing stays on this computer. The original is never overwritten.

Each input produces:

- `_defense_MODE.png`: processed, metadata-free PNG.
- `_defense_MODE.json`: measurements and every accepted/rejected checkpoint.
- `_defense_MODE_preview.png`: original/processed side-by-side view.

## What changed in experimental-0.2

The earlier prototype failed its visual acceptance test: even Gentle applied patterned texture across the full image and silently reduced the long edge to 1024 pixels. This version changes that behavior:

- Source resolution is preserved by default. Downscaling happens only when `--max-side` is explicitly supplied.
- Optimization is restricted to soft, landmark-guided face ovals. Pixels outside those regions remain byte-for-byte unchanged.
- The perturbation combines a low-resolution, spatially coherent color field with a small feathered subpixel warp. It does not apply a full-frame noise pattern.
- Every two iterations, the saved-pixel candidate is tested for face-region PSNR, SSIM, high-frequency residual energy, residual periodicity, and any background change.
- A failing candidate is rejected even when its model score is better. Repeated failures stop the run early. If nothing passes, the unchanged image is saved and the report says `quality_gate_fallback`.
- If no face is detected, the output is an unchanged clean copy and the report says `no_face_no_change`. The tool does not pretend that scene optimization provides face privacy.
- Optimization uses both VGGFace2-trained and CASIA-WebFace-trained InceptionResnetV1 models across original, JPEG-85, half-size/resampled, and smoothed face crops.

## Modes and gates

The additive bound is not a bound on total pixel difference because a subpixel warp can move pixels across an edge. The report records the actual maximum pixel change as well as the perceptual measurements.

| Mode | Additive bound | Warp bound | Minimum face PSNR | Minimum face SSIM | Maximum HF residual |
|---|---:|---:|---:|---:|---:|
| Gentle | 3/255 | 0.75 px | 42.5 dB | 0.995 | 1.20/255 |
| Balanced | 5/255 | 1.50 px | 37.5 dB | 0.990 | 2.20/255 |
| Strong | 7/255 | 2.25 px | 32.0 dB | 0.944 | 3.20/255 |

Start with **Gentle**. Strong intentionally permits a visible local face alteration; it is no longer an almost-imperceptible setting. Previews include magnified original/processed face details so this can be judged. Inspect the generated preview and full-resolution output before sharing it. The automated gate catches broad texture, periodic residuals, and excessive numerical change, but it cannot replace human visual review—especially on skin, eyes, teeth, fine hair, and high-contrast face edges.

## Public-sample result

On the 512×512 scikit-image astronaut portrait, all three modes preserved resolution and changed no pixels outside the face mask:

| Mode | Face PSNR | Face SSIM | VGGFace2 cosine | CASIA cosine |
|---|---:|---:|---:|---:|
| Gentle | 43.8 dB | 0.9967 | 0.931 | 0.965 |
| Balanced | 38.9 dB | 0.9904 | 0.867 | 0.927 |
| Strong | 33.3 dB | 0.9628 | 0.627 | 0.787 |

Lower cosine means less similarity in that particular model's embedding space. These are **not privacy percentages or identity decision thresholds**. Both face models participate in optimization, so these are in-ensemble measurements, not independent held-out validation. The high whole-photo similarity is expected because the background is intentionally preserved.

The public sample looks much cleaner than experimental-0.1, but its face-feature disruption is also much more modest. That tradeoff is deliberate and honestly reported. A visually clean score change on two models does not establish resistance to likeness learning or generation.

## Technical notes

- Face detection uses MTCNN on a bounded-size copy, then maps boxes and five landmarks back to source resolution.
- Face crops are fixed from the original detection. The processed result is not independently redetected or realigned during evaluation.
- Checkpoint selection uses the same two face models later reported. MobileNetV3 remains a whole-photo diagnostic only; it is not an identity evaluator.
- JPEG uses a real encoded forward pass with a straight-through gradient approximation. Resize and smoothing paths are differentiable.
- New PNG files omit source metadata. Transparent inputs are composited over white.
- Existing results are never overwritten. A repeat run automatically uses `_2`, `_3`, and so on for its output, report, and preview.
- `--no-preview` suppresses the side-by-side sheet. `--max-side N` opts into resizing. The default has no resolution cap.

## Validation

Run from the application directory:

```bat
.venv-defense\Scripts\python.exe test_defense.py
```

The regression suite covers exact pixel roundtrip, batched real-JPEG forward/gradient behavior, bounded optimization, exact background preservation, rejection of a checker-pattern residual, existing-output preservation, and resolution-preserving no-face behavior.

The original engine and launcher remain separate. `run_chameleon.bat` still runs the older pipeline; `run_defense.bat` runs this quality-gated experiment.
