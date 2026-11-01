# Third-party notices

The MIT License for Chameleon covers the Chameleon source code in this package. It does not replace the licenses or terms that apply to third-party software, pretrained model weights, or training datasets.

Third-party model files are not bundled with this source package. The tools download them from their upstream distributions on first use, or use a copy supplied locally by the operator.

## Direct software dependencies

| Component | Used by | License or terms |
|---|---|---|
| NumPy | Engine and Defense | BSD 3-Clause License |
| Pillow | Engine and Defense | HPND License, also described by the project as MIT-CMU style |
| OpenCV | Engine | Apache License 2.0 |
| PyTorch | Defense | BSD 3-Clause License and additional notices supplied with the distribution |
| TorchVision | Defense | BSD 3-Clause License |
| facenet-pytorch | Defense | MIT License |

Transitive packages in `defense-environment.txt` remain governed by the license information supplied with each package.

## Models and weights

### YuNet face detector

`Chameleon_Engine.py` can download `face_detection_yunet_2023mar.onnx` from the OpenCV Zoo model collection. The model is an OpenCV third-party artifact and remains subject to the terms supplied by OpenCV Zoo.

### MTCNN and InceptionResnetV1

Defense uses MTCNN and InceptionResnetV1 through facenet-pytorch. The available pretrained weights include models trained on VGGFace2 and CASIA-WebFace.

Those weights and their source datasets are not distributed in this package. Review the facenet-pytorch, VGGFace2, and CASIA-WebFace terms before redistributing downloaded weights or using them for a purpose outside the scope allowed by their providers.

### TorchVision weights

Defense uses ResNet18 weights for scene features and MobileNetV3 Small weights for a whole-photo diagnostic. The weights are obtained through TorchVision and remain subject to the TorchVision and model-specific terms supplied with that distribution.

## Redistribution note

Before publishing a mirror, installer, container, or archive that bundles any downloaded ONNX or PyTorch weights, review the current upstream license and model terms for each artifact. A source-only release avoids bundling those files.
