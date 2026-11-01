# Chameleon setup

These instructions cover the Windows launchers supplied with the project.

The setup commands use 64-bit Python 3.12 as the documented baseline. The captured Defense environment did not preserve the Python version used to create it, so Python 3.12 is a setup target for this package rather than a claim about the original machine.

## Before you start

You will need:

- Windows 10 or Windows 11
- Python 3.12, 64-bit, available on the system path
- Internet access for the first package and model downloads
- Enough free disk space for the PyTorch environment and its model cache

For GPU use in Defense, use a current NVIDIA driver. Defense can run on CPU, but the supplied environment capture uses the CUDA 12.8 PyTorch build and processing will be slower without a supported GPU.

## Engine setup

The Engine needs NumPy, Pillow, and OpenCV.

### Option 1: use the launcher

Double-click `run_chameleon.bat`. If the three packages are missing, the launcher offers to install them with pip.

### Option 2: install manually

From the project folder:

```bat
python -m pip install -r requirements-engine.txt
```

Run the Engine:

```bat
python Chameleon_Engine.py input.jpg -o output.jpg --mode balanced --verify
```

Run its audit tests:

```bat
python -m unittest -v test_chameleon_audit.py
```

### YuNet model

The Engine uses the YuNet face detector for verification, adaptive mode, and the geometry stages in Strong and Max modes.

On first use, it searches for an ONNX model in `Chameleon_Toolkit` or beside the script. If none is found, it tries to download `face_detection_yunet_2023mar.onnx` from OpenCV Zoo.

If the machine is offline, place that ONNX file in `Chameleon_Toolkit` before running those features. Basic processing without verification, adaptive mode, or face geometry can continue without it.

## Defense setup

Defense uses the dedicated `.venv-defense` environment expected by `run_defense.bat`.

From the project folder:

```bat
py -3.12 -m venv .venv-defense
.venv-defense\Scripts\python.exe -m pip install --upgrade pip
.venv-defense\Scripts\python.exe -m pip install -r defense-environment.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

The extra package index is needed because the environment capture pins the CUDA 12.8 build of PyTorch.

Check the core imports:

```bat
.venv-defense\Scripts\python.exe -c "import torch, torchvision, facenet_pytorch; print(torch.__version__)"
```

Run the regression tests:

```bat
.venv-defense\Scripts\python.exe -m unittest -v test_defense.py
```

Start Defense with the launcher:

```bat
run_defense.bat
```

Or call it directly:

```bat
.venv-defense\Scripts\python.exe Chameleon_Defense.py photo.jpg --mode gentle
```

### Defense model downloads

Defense uses pretrained models through facenet-pytorch and TorchVision. It sets its Torch cache to:

```text
Chameleon_Toolkit\torch
```

When a face-bearing image is processed, required weights may be downloaded on first use. This can include MTCNN, InceptionResnetV1, ResNet18, and MobileNetV3 weights.

For offline use, run one face-bearing test image while online so the required cache is created, then keep the `Chameleon_Toolkit\torch` folder with the working copy. Do not add that cache to a source repository.

## Generated files to keep out of source control

The supplied `.gitignore` excludes:

- `.venv-defense` and other local virtual environments
- `Chameleon_Toolkit` model caches
- Python bytecode and test caches
- Chameleon output images and JSON reports

## Troubleshooting

### `run_defense.bat` says the Python executable does not exist

The `.venv-defense` folder has not been created, or it was created in a different project folder. Repeat the Defense setup from the folder that contains `run_defense.bat`.

### `ModuleNotFoundError: No module named 'cv2'`

Install the Engine requirements into the same Python interpreter used to start `Chameleon_Engine.py`:

```bat
python -m pip install -r requirements-engine.txt
```

### Defense cannot download a model

Check the internet connection and retry while online. The source package intentionally does not bundle third-party model weights.

### Defense is using CPU instead of GPU

Confirm that PyTorch sees the GPU:

```bat
.venv-defense\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
```

If it prints `False`, check the NVIDIA driver and confirm that the CUDA environment installed successfully.
