# Chameleon

Chameleon is a pair of local, experimental image-processing tools from CyberSage Technologies.

The project explores image perturbation, metadata removal, face-local feature disruption, and strict visual-quality checks. It is a research tool, not a proven privacy product.

## Important limits

Chameleon does **not** guarantee protection against:

- AI training or fine-tuning
- LoRA, DreamBooth, or other likeness learning
- Reference-image generation or face swaps
- Face recognition or future recognition models
- Commercial image or identity services

The Engine does not optimize a loss against a recognition or generative model. Its optional YuNet check measures face detection only.

Defense does optimize against two face-feature models, but its reported face scores are measured on those same models. They are not independent validation, privacy percentages, identity thresholds, or proof of protection.

Read the detailed notes before relying on any result.

## The two tools

### Chameleon Engine v2.1.1

`Chameleon_Engine.py` is the original full-frame pipeline. It can:

- Apply frequency, chroma, structured-noise, and grain treatments
- Apply stronger face-region geometry in its Strong and Max modes
- Remove source metadata by saving fresh pixel data
- Report PSNR as a pixel-fidelity measurement
- Optionally check whether YuNet face detection changes after processing

Modes: `subtle`, `balanced`, `strong`, and `max`.

### Chameleon Defense experimental-0.2

`Chameleon_Defense.py` is the newer quality-gated pipeline. It:

- Detects faces with MTCNN and works only inside soft face regions
- Leaves pixels outside the face mask unchanged
- Combines a smooth color field with a bounded subpixel warp
- Rejects checkpoints that fail face PSNR, SSIM, high-frequency, periodicity, or background-preservation checks
- Preserves source resolution unless `--max-side` is explicitly supplied
- Writes a PNG, a JSON measurement report, and an optional comparison preview
- Saves an unchanged copy and reports `no_face_no_change` when no face is detected

Modes: `gentle`, `balanced`, and `strong`.

The detailed Defense design, quality gates, and sample measurements are in [DEFENSE_README.md](DEFENSE_README.md).

## Quick start

Full installation instructions are in [SETUP.md](SETUP.md).

### Engine on Windows

Double-click `run_chameleon.bat` and drop images onto it, or run:

```bat
python Chameleon_Engine.py input.jpg -o output.jpg --mode balanced --verify
```

The launcher checks for NumPy, Pillow, and OpenCV and offers to install them if they are missing.

### Defense on Windows

Create the Defense environment by following [SETUP.md](SETUP.md), then either double-click `run_defense.bat` or run:

```bat
.venv-defense\Scripts\python.exe Chameleon_Defense.py photo.jpg --mode gentle
```

Defense can process more than one image in a single command. Run it without image arguments to open a file picker.

## Outputs

Engine outputs use the input name plus `_chameleon` unless an output path is supplied.

For each Defense input, the default outputs are:

- `NAME_defense_MODE.png`: processed image
- `NAME_defense_MODE.json`: settings, measurements, checkpoints, and status
- `NAME_defense_MODE_preview.png`: original and processed comparison

Defense never overwrites an existing result set. A repeated run uses a numbered suffix.

Original images are not overwritten by either tool.

## Tests

Engine audit tests:

```bat
python -m unittest -v test_chameleon_audit.py
```

Defense regression tests:

```bat
.venv-defense\Scripts\python.exe -m unittest -v test_defense.py
```

## Data handling

Image processing happens locally. On first use, the tools may download public model files:

- The Engine may download the YuNet ONNX detector into `Chameleon_Toolkit`.
- Defense may download pretrained model weights into `Chameleon_Toolkit\torch`.

Those downloaded files are runtime caches and are not part of this source package.

## Project layout

- `Chameleon_Engine.py`: original full-frame pipeline
- `Chameleon_Defense.py`: quality-gated face-local pipeline
- `run_chameleon.bat`: Windows launcher for the Engine
- `run_defense.bat`: Windows launcher for Defense
- `test_chameleon_audit.py`: Engine audit tests
- `test_defense.py`: Defense regression tests
- `requirements-engine.txt`: direct Engine dependencies
- `defense-environment.txt`: captured Defense environment
- `DEFENSE_README.md`: detailed Defense notes
- `SETUP.md`: installation and first-run guide
- `THIRD_PARTY_NOTICES.md`: third-party software and model notices

## License

Chameleon source code is released under the MIT License. See [LICENSE](LICENSE).

Third-party software, datasets, and pretrained weights remain governed by their own terms. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
