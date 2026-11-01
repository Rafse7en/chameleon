"""Quality-gated, face-local experimental feature disruption. No privacy guarantee."""
import argparse
import io
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('TORCH_HOME', str(ROOT / 'Chameleon_Toolkit' / 'torch'))
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from facenet_pytorch import MTCNN, InceptionResnetV1

# Bounds are deliberately much lower than experimental-0.1 (8/16/24). A
# checkpoint is usable only when its measured face-region quality also passes.
PRESETS = {
    'gentle': {
        'epsilon_255': 3, 'steps': 80, 'learning_rate': .08,
        'max_warp_px': .75,
        'min_face_psnr_db': 42.5, 'min_face_ssim': .995,
        'max_hf_rms_255': 1.20, 'max_periodicity': .48,
    },
    'balanced': {
        'epsilon_255': 5, 'steps': 110, 'learning_rate': .10,
        'max_warp_px': 1.50,
        'min_face_psnr_db': 37.5, 'min_face_ssim': .990,
        'max_hf_rms_255': 2.20, 'max_periodicity': .55,
    },
    'strong': {
        'epsilon_255': 7, 'steps': 150, 'learning_rate': .12,
        'max_warp_px': 2.25,
        'min_face_psnr_db': 32.0, 'min_face_ssim': .944,
        'max_hf_rms_255': 3.20, 'max_periodicity': .62,
    },
}
TRANSFORMS = ('original', 'jpeg85', 'resize50', 'blur')


def tensor(im, device):
    return torch.from_numpy(np.array(im.convert('RGB'), copy=True)).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255


def picture(x, index=0):
    pixels = (x.detach().clamp(0, 1)[index].permute(1, 2, 0).cpu().numpy() * 255).round().astype('uint8')
    return Image.fromarray(pixels)


def quantize(x):
    return (x.detach().clamp(0, 1) * 255).round() / 255


def jpeg(x, quality=85, straight_through=False):
    images = []
    for index in range(x.shape[0]):
        buf = io.BytesIO()
        picture(x, index).save(buf, format='JPEG', quality=quality)
        buf.seek(0)
        with Image.open(buf) as im:
            images.append(tensor(im, x.device))
    y = torch.cat(images)
    return x + (y - x).detach() if straight_through else y


def transform(x, kind, ste=False):
    if kind == 'jpeg85':
        return jpeg(x, 85, ste)
    if kind == 'jpeg70':
        return jpeg(x, 70, ste)
    if kind == 'resize50':
        small = F.interpolate(x, scale_factor=.5, mode='bilinear', align_corners=False, antialias=True)
        return F.interpolate(small, size=x.shape[-2:], mode='bilinear', align_corners=False)
    if kind == 'blur':
        return F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode='reflect'), 3, stride=1)
    return x


class Features:
    def __init__(self, device, faces):
        self.device, self.faces = device, faces
        self.scene = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.scene.fc = nn.Identity()
        self.face = InceptionResnetV1(pretrained='vggface2') if faces else None
        self.face_alt = InceptionResnetV1(pretrained='casia-webface') if faces else None
        for model in (self.scene, self.face, self.face_alt):
            if model is not None:
                model.eval().to(device).requires_grad_(False)
        self.mean = torch.tensor([.485, .456, .406], device=device)[None, :, None, None]
        self.std = torch.tensor([.229, .224, .225], device=device)[None, :, None, None]

    def scene_input(self, x):
        return (F.interpolate(x, (224, 224), mode='bilinear', align_corners=False, antialias=True) - self.mean) / self.std

    def face_input_region(self, x, offset=(0, 0)):
        ox, oy = offset
        crops = [x[:, :, y0 - oy:y1 - oy, x0 - ox:x1 - ox] for x0, y0, x1, y1 in self.faces]
        return torch.cat([F.interpolate(crop, (160, 160), mode='bilinear', align_corners=False, antialias=True) for crop in crops])

    def face_input(self, x):
        return self.face_input_region(x)

    def face_embeddings(self, x, kind='original', ste=False):
        crops = transform(self.face_input(x), kind, ste)
        normalized = crops * 255 / 128 - 127.5 / 128
        return F.normalize(self.face(normalized), dim=1)

    def optimization_embeddings(self, x, kind='original', ste=False):
        return self.optimization_embeddings_region(x, (0, 0), kind, ste)

    def optimization_embeddings_region(self, x, offset, kind='original', ste=False):
        crops = transform(self.face_input_region(x, offset), kind, ste)
        normalized = crops * 255 / 128 - 127.5 / 128
        face_values = F.normalize(self.face(normalized), dim=1)
        alternate_values = F.normalize(self.face_alt(normalized), dim=1)
        return [face_values, alternate_values]

    def __call__(self, x):
        values = [F.normalize(self.scene(self.scene_input(x)), dim=1)]
        if self.face is not None:
            values.append(self.face_embeddings(x))
        return values


def distance(values, targets):
    return torch.stack([(1 - (a * b).sum(1)).mean() for a, b in zip(values, targets)]).mean()


def detect_faces(im, detector, detection_max_side=1600):
    """Detect on a bounded copy, then return boxes/landmarks in source pixels."""
    probe = im.copy()
    probe.thumbnail((detection_max_side, detection_max_side), Image.Resampling.LANCZOS)
    scale_x, scale_y = im.width / probe.width, im.height / probe.height
    try:
        detected = detector.detect(probe, landmarks=True)
    except TypeError:
        detected = detector.detect(probe)
    boxes, probs = detected[:2]
    points = detected[2] if len(detected) > 2 else None
    faces, landmarks = [], []
    if boxes is None:
        return faces, landmarks
    for index, (box, prob) in enumerate(zip(boxes, probs)):
        if prob is None or prob < .9:
            continue
        x0, y0, x1, y1 = box
        x0, x1, y0, y1 = x0 * scale_x, x1 * scale_x, y0 * scale_y, y1 * scale_y
        mx, my = (x1 - x0) * .08, (y1 - y0) * .08
        rect = (max(0, int(x0 - mx)), max(0, int(y0 - my)),
                min(im.width, int(math.ceil(x1 + mx))), min(im.height, int(math.ceil(y1 + my))))
        if rect[2] - rect[0] < 16 or rect[3] - rect[1] < 16:
            continue
        faces.append(rect)
        if points is not None:
            landmarks.append([(float(px * scale_x), float(py * scale_y)) for px, py in points[index]])
        else:
            landmarks.append([])
    return faces, landmarks


def make_face_mask(size, faces, landmarks, device):
    """Soft face oval, strengthened around identity-bearing landmarks."""
    width, height = size
    mask = np.zeros((height, width), dtype=np.float32)
    for index, (x0, y0, x1, y1) in enumerate(faces):
        fw, fh = max(1, x1 - x0), max(1, y1 - y0)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        cx, cy = (x0 + x1) / 2, y0 + .50 * fh
        radial = np.sqrt(((xx - cx) / (.49 * fw)) ** 2 + ((yy - cy) / (.55 * fh)) ** 2)
        oval = np.clip((1.0 - radial) / .22, 0, 1)
        local = .62 * oval
        if index < len(landmarks) and landmarks[index]:
            emphasis = np.zeros_like(local)
            sigma = max(4.0, .115 * min(fw, fh))
            for px, py in landmarks[index]:
                emphasis = np.maximum(emphasis, np.exp(-((xx - px) ** 2 + (yy - py) ** 2) / (2 * sigma ** 2)))
            local = np.maximum(local, oval * (.55 + .45 * emphasis))
        mask[y0:y1, x0:x1] = np.maximum(mask[y0:y1, x0:x1], local)
    return torch.from_numpy(mask.astype(np.float32, copy=False))[None, None].to(device)


def _ssim_map(a, b):
    window = 7 if min(a.shape[-2:]) >= 7 else 3
    pad = window // 2
    a = F.pad(a, (pad, pad, pad, pad), mode='reflect')
    b = F.pad(b, (pad, pad, pad, pad), mode='reflect')
    mu_a, mu_b = F.avg_pool2d(a, window, 1), F.avg_pool2d(b, window, 1)
    var_a = F.avg_pool2d(a * a, window, 1) - mu_a * mu_a
    var_b = F.avg_pool2d(b * b, window, 1) - mu_b * mu_b
    cov = F.avg_pool2d(a * b, window, 1) - mu_a * mu_b
    c1, c2 = .01 ** 2, .03 ** 2
    return ((2 * mu_a * mu_b + c1) * (2 * cov + c2) /
            ((mu_a * mu_a + mu_b * mu_b + c1) * (var_a + var_b + c2))).mean(1, keepdim=True)


def _mask_bounds(mask):
    active = torch.nonzero(mask[0, 0] > .01, as_tuple=False)
    if not len(active):
        return 0, 0, mask.shape[-1], mask.shape[-2]
    y0, x0 = active.min(0).values.tolist()
    y1, x1 = (active.max(0).values + 1).tolist()
    return x0, y0, x1, y1


def _weighted_correlation(a, b, weight):
    denom = weight.sum().clamp_min(1e-8)
    ma, mb = (a * weight).sum() / denom, (b * weight).sum() / denom
    aa, bb = (a - ma) * weight.sqrt(), (b - mb) * weight.sqrt()
    energy = (aa.square().sum() * bb.square().sum()).sqrt().clamp_min(1e-10)
    return float((aa * bb).sum().abs() / energy)


def _periodicity(highpass, weight):
    gray = highpass.mean(1, keepdim=True)
    scores = []
    for shift in (2, 3, 4, 6, 8):
        if gray.shape[-1] > shift + 8:
            scores.append(_weighted_correlation(gray[..., :-shift], gray[..., shift:],
                                                torch.minimum(weight[..., :-shift], weight[..., shift:])))
        if gray.shape[-2] > shift + 8:
            scores.append(_weighted_correlation(gray[..., :-shift, :], gray[..., shift:, :],
                                                torch.minimum(weight[..., :-shift, :], weight[..., shift:, :])))
    return max(scores, default=0.0)


def quality_metrics(original, candidate, mask):
    candidate = quantize(candidate)
    delta = candidate - original
    global_mse = float(delta.square().mean())
    weight = mask.clamp(0, 1)
    channels = 3 * weight.sum().clamp_min(1e-8)
    face_mse = float((delta.square() * weight).sum() / channels)
    x0, y0, x1, y1 = _mask_bounds(mask)
    a, b, w = original[:, :, y0:y1, x0:x1], candidate[:, :, y0:y1, x0:x1], weight[:, :, y0:y1, x0:x1]
    max_dim = max(a.shape[-2:])
    if max_dim > 768:
        ratio = 768 / max_dim
        target = (max(16, round(a.shape[-2] * ratio)), max(16, round(a.shape[-1] * ratio)))
        a = F.interpolate(a, target, mode='bilinear', align_corners=False, antialias=True)
        b = F.interpolate(b, target, mode='bilinear', align_corners=False, antialias=True)
        w = F.interpolate(w, target, mode='bilinear', align_corners=False)
    ssim = _ssim_map(a, b)
    face_ssim = float((ssim * w).sum() / w.sum().clamp_min(1e-8))
    local_delta = b - a
    smooth = F.avg_pool2d(F.pad(local_delta, (2, 2, 2, 2), mode='reflect'), 5, stride=1)
    highpass = local_delta - smooth
    hf_rms = float(((highpass.square() * w).sum() / (3 * w.sum().clamp_min(1e-8))).sqrt() * 255)
    support = (mask > .01).expand_as(delta)
    outside = delta.masked_select(~support)
    outside_max = float(outside.abs().max() * 255) if outside.numel() else 0.0
    return {
        'psnr_db': float(-10 * np.log10(max(global_mse, 1e-12))),
        'face_psnr_db': float(-10 * np.log10(max(face_mse, 1e-12))),
        'face_ssim': face_ssim,
        'high_frequency_rms_255': hf_rms,
        'residual_periodicity': _periodicity(highpass, w),
        'outside_face_max_change_255': outside_max,
        'changed_pixel_fraction': float((delta.abs().amax(1) > (0.5 / 255)).float().mean()),
    }


def quality_passes(metrics, limits):
    failures = []
    checks = (
        ('face PSNR', metrics['face_psnr_db'], limits['min_face_psnr_db'], '>='),
        ('face SSIM', metrics['face_ssim'], limits['min_face_ssim'], '>='),
        ('high-frequency RMS', metrics['high_frequency_rms_255'], limits['max_hf_rms_255'], '<='),
        ('residual periodicity', metrics['residual_periodicity'], limits['max_periodicity'], '<='),
        ('outside-face change', metrics['outside_face_max_change_255'], 0.0, '<='),
    )
    for label, value, threshold, direction in checks:
        passed = value >= threshold if direction == '>=' else value <= threshold + 1e-7
        if not passed:
            failures.append(f'{label} {value:.4f} must be {direction} {threshold:.4f}')
    return not failures, failures


def _optimization_values(features, image, kind, ste):
    if hasattr(features, 'optimization_embeddings'):
        return features.optimization_embeddings(image, kind, ste)
    if hasattr(features, 'face_embeddings'):
        return [features.face_embeddings(image, kind, ste)]
    return features(transform(image, kind, ste))


def optimize(original, features, eps, steps, seed, mask=None, quality_limits=None,
             learning_rate=.1, max_warp_px=0.0, progress=print, return_details=False):
    """Optimize a smooth field, retaining only quality-gate-passing checkpoints."""
    torch.manual_seed(seed)
    if mask is None:
        mask = torch.ones_like(original[:, :1])
    height, width = original.shape[-2:]
    if hasattr(features, 'faces') and features.faces:
        x0 = min(face[0] for face in features.faces); y0 = min(face[1] for face in features.faces)
        x1 = max(face[2] for face in features.faces); y1 = max(face[3] for face in features.faces)
    else:
        x0, y0, x1, y1 = _mask_bounds(mask)
    patch_height, patch_width = y1 - y0, x1 - x0
    original_patch = original[:, :, y0:y1, x0:x1]
    mask_patch = mask[:, :, y0:y1, x0:x1]
    uses_region_features = hasattr(features, 'optimization_embeddings_region')

    def feature_values(image_or_patch, kind, ste):
        if uses_region_features:
            return features.optimization_embeddings_region(image_or_patch, (x0, y0), kind, ste)
        return _optimization_values(features, image_or_patch, kind, ste)

    with torch.no_grad():
        reference_source = original_patch if uses_region_features else original
        refs = {kind: feature_values(reference_source, kind, False) for kind in TRANSFORMS}
    field_size = (max(12, math.ceil(patch_height / 6)), max(12, math.ceil(patch_width / 6)))
    field = torch.zeros((1, 3, *field_size), device=original.device, requires_grad=True)
    flow_size = (max(6, math.ceil(patch_height / 32)), max(6, math.ceil(patch_width / 32)))
    flow = torch.zeros((1, 2, *flow_size), device=original.device, requires_grad=True)
    optimizer = torch.optim.Adam([
        {'params': [field], 'lr': learning_rate * eps},
        {'params': [flow], 'lr': .025 if max_warp_px else 0.0},
    ])
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, patch_height, device=original.device),
        torch.linspace(-1, 1, patch_width, device=original.device), indexing='ij')
    base_grid = torch.stack((grid_x, grid_y), dim=-1)[None]

    def render_candidate():
        color_raw = F.interpolate(field, (patch_height, patch_width), mode='bicubic', align_corners=False).clamp(-eps, eps)
        luminance = color_raw.mean(1, keepdim=True)
        # Most additive freedom is shared across RGB to avoid colored eye/skin artifacts.
        color = (luminance + .20 * (color_raw - luminance)).clamp(-eps, eps)
        flow_pixels = F.interpolate(flow, (patch_height, patch_width), mode='bicubic', align_corners=False).clamp(-1, 1)
        displacement = torch.stack((
            flow_pixels[:, 0] * (2 * max_warp_px / max(1, patch_width - 1)),
            flow_pixels[:, 1] * (2 * max_warp_px / max(1, patch_height - 1)),
        ), dim=-1)
        warped = F.grid_sample(original_patch, base_grid + displacement, mode='bilinear',
                               padding_mode='border', align_corners=True)
        defended_patch = original_patch + (warped - original_patch + color) * mask_patch
        defended_patch = defended_patch.clamp(0, 1)
        patch_delta = defended_patch - original_patch
        return defended_patch, color * mask_patch, flow_pixels

    def expand_patch(defended_patch):
        patch_delta = defended_patch - original_patch
        return (original + F.pad(patch_delta, (x0, width - x1, y0, height - y1))).clamp(0, 1)
    best, best_score = original.detach().clone(), 0.0
    history = []
    reject_streak = 0
    stop_reason = None
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        candidate_patch, color_delta, flow_pixels = render_candidate()
        delta = candidate_patch - original_patch
        kind = TRANSFORMS[step % len(TRANSFORMS)]
        feature_source = candidate_patch if uses_region_features else expand_patch(candidate_patch)
        identity_distance = distance(feature_values(feature_source, kind, True), refs[kind])
        active = mask_patch.expand_as(delta)
        normalization = (active.sum() * eps * eps).clamp_min(1e-12)
        energy = (delta.square() * active).sum() / normalization
        dx, dy = delta[..., 1:] - delta[..., :-1], delta[..., 1:, :] - delta[..., :-1, :]
        smoothness = (dx.square().mean() + dy.square().mean()) / max(eps * eps, 1e-12)
        blurred = F.avg_pool2d(F.pad(delta, (2, 2, 2, 2), mode='reflect'), 5, stride=1)
        high_frequency = (delta - blurred).square().mean() / max(eps * eps, 1e-12)
        flow_energy = flow_pixels.square().mean()
        flow_dx = flow_pixels[..., 1:] - flow_pixels[..., :-1]
        flow_dy = flow_pixels[..., 1:, :] - flow_pixels[..., :-1, :]
        flow_smoothness = flow_dx.square().mean() + flow_dy.square().mean()
        objective = (identity_distance - .018 * energy - .055 * smoothness - .10 * high_frequency
                     - .020 * flow_energy - .040 * flow_smoothness)
        (-objective).backward()
        if field.grad is None or flow.grad is None or not torch.isfinite(field.grad).all() or not torch.isfinite(flow.grad).all():
            raise RuntimeError('Non-finite optimization gradient')
        torch.nn.utils.clip_grad_norm_([field, flow], 1.0)
        optimizer.step()
        with torch.no_grad():
            field.clamp_(-eps, eps)
            flow.clamp_(-1, 1)
        if (step + 1) % 2 == 0 or step == steps - 1:
            with torch.no_grad():
                raw_candidate = expand_patch(render_candidate()[0])
                candidate = quantize(raw_candidate) if quality_limits is not None else raw_candidate
                score = sum(float(distance(feature_values(
                                candidate[:, :, y0:y1, x0:x1] if uses_region_features else candidate,
                                kind, False), refs[kind]))
                            for kind in TRANSFORMS) / len(TRANSFORMS)
                metrics = quality_metrics(original, candidate, mask) if quality_limits is not None else None
                passed, failures = quality_passes(metrics, quality_limits) if metrics else (True, [])
                if passed and score > best_score:
                    best, best_score = candidate.detach().clone(), score
                    reject_streak = 0
                elif passed:
                    reject_streak = 0
                else:
                    reject_streak += 1
                row = {'step': step + 1, 'feature_distance': score, 'quality_passed': passed}
                if metrics:
                    row['quality'] = metrics
                    row['quality_failures'] = failures
                history.append(row)
            progress(f'  Step {step + 1}/{steps}: face feature distance {score:.4f}; quality ' +
                     ('PASS' if passed else 'REJECT'), flush=True)
            if best_score > 0 and reject_streak >= 10:
                stop_reason = 'stopped after 10 consecutive quality-rejected checkpoints'
                progress('  Stopping early: later checkpoints repeatedly failed the quality gate.', flush=True)
                break
    details = {'selected_feature_distance': best_score, 'checkpoints': history,
               'selected_unchanged_fallback': best_score <= 0.0, 'stop_reason': stop_reason}
    return (best, details) if return_details else best


def _evaluation_view(x, max_side=1024):
    height, width = x.shape[-2:]
    if max(height, width) <= max_side:
        return x
    scale = max_side / max(height, width)
    return F.interpolate(x, (round(height * scale), round(width * scale)), mode='bilinear', align_corners=False, antialias=True)


def evaluate(original, saved, features):
    # MobileNet is a held-out whole-photo diagnostic. Both face models are in-ensemble.
    held_scene = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT).eval().to(original.device)
    held_scene.classifier = nn.Identity()
    rows = []
    scene_a, scene_b = _evaluation_view(original), _evaluation_view(saved)
    face_a, face_b = features.face_input(original), features.face_input(saved)
    with torch.no_grad():
        for kind in ('original', 'jpeg85', 'jpeg70', 'resize50', 'blur'):
            sa, sb = transform(scene_a, kind), transform(scene_b, kind)
            optimized_scene_a = F.normalize(features.scene(features.scene_input(sa)), dim=1)
            optimized_scene_b = F.normalize(features.scene(features.scene_input(sb)), dim=1)
            held_scene_a, held_scene_b = held_scene(features.scene_input(sa)), held_scene(features.scene_input(sb))
            fa, fb = transform(face_a, kind), transform(face_b, kind)
            normalized_a, normalized_b = fa * 255 / 128 - 127.5 / 128, fb * 255 / 128 - 127.5 / 128
            optimized_face_a = F.normalize(features.face(normalized_a), dim=1)
            optimized_face_b = F.normalize(features.face(normalized_b), dim=1)
            alternate_face_a = F.normalize(features.face_alt(normalized_a), dim=1)
            alternate_face_b = F.normalize(features.face_alt(normalized_b), dim=1)
            rows.append({
                'transform': kind,
                'optimized_scene_cosine': float((optimized_scene_a * optimized_scene_b).sum()),
                'heldout_scene_cosine': float(F.cosine_similarity(held_scene_a, held_scene_b).mean()),
                'vggface2_face_cosines': (optimized_face_a * optimized_face_b).sum(1).tolist(),
                'casia_face_cosines': (alternate_face_a * alternate_face_b).sum(1).tolist(),
            })
    return rows


def write_preview(original, defended, destination, faces=()):
    left, right = original.copy(), defended.copy()
    left.thumbnail((800, 800), Image.Resampling.LANCZOS)
    right.thumbnail((800, 800), Image.Resampling.LANCZOS)
    width = left.width + right.width
    full_height = max(left.height, right.height)
    detail_height = 0
    face_pair = None
    if faces:
        face = max(faces, key=lambda rect: (rect[2] - rect[0]) * (rect[3] - rect[1]))
        x0, y0, x1, y1 = face
        fw, fh = x1 - x0, y1 - y0
        margin_x, margin_y = round(fw * .18), round(fh * .14)
        crop_box = (max(0, x0 - margin_x), max(0, y0 - margin_y),
                    min(original.width, x1 + margin_x), min(original.height, y1 + margin_y))
        original_face, defended_face = original.crop(crop_box), defended.crop(crop_box)
        scale = 300 / max(1, original_face.height)
        detail_size = (max(1, round(original_face.width * scale)), 300)
        original_face = original_face.resize(detail_size, Image.Resampling.LANCZOS)
        defended_face = defended_face.resize(detail_size, Image.Resampling.LANCZOS)
        face_pair = original_face, defended_face
        detail_height = 340
    height = full_height + 32 + detail_height
    sheet = Image.new('RGB', (width, height), 'white')
    sheet.paste(left, (0, 32)); sheet.paste(right, (left.width, 32))
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 9), 'ORIGINAL', fill='black')
    draw.text((left.width + 8, 9), 'PROCESSED', fill='black')
    if face_pair:
        original_face, defended_face = face_pair
        detail_y = 32 + full_height + 34
        sheet.paste(original_face, ((left.width - original_face.width) // 2, detail_y))
        sheet.paste(defended_face, (left.width + (right.width - defended_face.width) // 2, detail_y))
        draw.text((8, 32 + full_height + 10), 'ORIGINAL FACE DETAIL', fill='black')
        draw.text((left.width + 8, 32 + full_height + 10), 'PROCESSED FACE DETAIL', fill='black')
    with destination.open('xb') as stream:
        sheet.save(stream, format='PNG')


def choose_output_paths(path, mode, preview=True):
    """Return a free output set without replacing any result from an earlier run."""
    base = path.stem + '_defense_' + mode
    run_number = 1
    while True:
        numbered_base = base if run_number == 1 else f'{base}_{run_number}'
        out = path.with_name(numbered_base + '.png')
        report_path = path.with_name(numbered_base + '.json')
        preview_path = path.with_name(numbered_base + '_preview.png')
        required = (out, report_path, preview_path) if preview else (out, report_path)
        if not any(candidate.exists() for candidate in required):
            return out, report_path, preview_path
        run_number += 1


def process(path, mode='balanced', steps=None, seed=42, max_side=None, preview=True):
    path = Path(path).resolve()
    out, report_path, preview_path = choose_output_paths(path, mode, preview)
    default_out = path.with_name(path.stem + '_defense_' + mode + '.png')
    if out != default_out:
        print(f'Existing result preserved; this run will use: {out.name}', flush=True)
    with Image.open(path) as raw:
        rgba = ImageOps.exif_transpose(raw).convert('RGBA')
        background = Image.new('RGBA', rgba.size, 'white')
        background.alpha_composite(rgba)
        im = background.convert('RGB')
    input_size = im.size
    if max_side is not None:
        im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    if min(im.size) < 32:
        raise ValueError('Image must be at least 32 pixels on both axes')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    print(f'Local device: {device}; output size: {im.size}; resolution ' +
          ('preserved' if im.size == input_size else 'explicitly reduced'), flush=True)
    detector = MTCNN(keep_all=True, device=device)
    faces, landmarks = detect_faces(im, detector)
    del detector
    print(f'{len(faces)} face(s) detected. ' +
          ('Face-local optimization; background pixels remain unchanged.' if faces else
           'No face-local processing will be applied; face protection NOT evaluated.'), flush=True)
    original = tensor(im, device)
    selected = original.clone()
    optimization = {'selected_feature_distance': 0.0, 'checkpoints': [], 'selected_unchanged_fallback': True}
    mask = torch.zeros_like(original[:, :1])
    features = None
    preset = PRESETS[mode]
    run_steps = steps or preset['steps']
    if faces:
        features = Features(device, faces)
        mask = make_face_mask(im.size, faces, landmarks, device)
        selected, optimization = optimize(
            original, features, preset['epsilon_255'] / 255, run_steps, seed, mask=mask,
            quality_limits=preset, learning_rate=preset['learning_rate'],
            max_warp_px=preset['max_warp_px'], return_details=True)
    with out.open('xb') as stream:
        picture(selected).save(stream, format='PNG')
    with Image.open(out) as saved_image:
        saved_pil = saved_image.convert('RGB')
    saved = tensor(saved_pil, device)
    quality = quality_metrics(original, saved, mask) if faces else {
        'psnr_db': 120.0, 'face_psnr_db': 120.0, 'face_ssim': 1.0,
        'high_frequency_rms_255': 0.0, 'residual_periodicity': 0.0,
        'outside_face_max_change_255': 0.0, 'changed_pixel_fraction': 0.0,
    }
    if preview:
        write_preview(im, saved_pil, preview_path, faces)
    report = {
        'version': 'experimental-0.2-quality-gated',
        'status': 'no_face_no_change' if not faces else 'evaluation_pending',
        'mode': mode, 'device': device, 'input_size': input_size, 'output_size': im.size,
        'resolution_preserved': input_size == im.size,
        'max_pixel_change_255': float((saved - original).abs().max() * 255),
        'quality': quality, 'faces': faces, 'landmarks': landmarks,
        'face_mask_pixel_fraction': float((mask > .01).float().mean()),
        'additive_pixel_bound_255': preset['epsilon_255'],
        'subpixel_warp_bound_px': preset['max_warp_px'],
        'steps': run_steps, 'seed': seed,
        'optimization': optimization,
        'optimization_models': ['InceptionResnetV1 VGGFace2', 'InceptionResnetV1 CASIA-WebFace'],
        'independent_face_evaluation_model': None,
        'training_protection': 'not_tested',
        'reference_image_generation_protection': 'not_tested',
        'privacy_scope': 'face-feature disruption only; background is intentionally preserved',
        'evaluation_note': ('Face cosines compare fixed original crops on the same two models used by optimization; there '
                            'is no independent held-out identity model in this build. Scores are not privacy percentages, '
                            'identity thresholds, or proof against training, generation, face swaps, or future models.'),
    }
    report_path.write_text(json.dumps(report, indent=2))
    if faces:
        try:
            report['measurements'] = evaluate(original, saved, features)
            report['status'] = 'quality_gate_fallback' if optimization['selected_unchanged_fallback'] else 'evaluated'
        except Exception as exc:
            report['status'] = 'evaluation_failed'
            report['error'] = str(exc)
            raise
        finally:
            report_path.write_text(json.dumps(report, indent=2))
    print(f"Quality: face PSNR {quality['face_psnr_db']:.1f} dB; face SSIM {quality['face_ssim']:.4f}; "
          f"HF residual {quality['high_frequency_rms_255']:.2f}/255.", flush=True)
    for row in report.get('measurements', []):
        scores = row.get('casia_face_cosines')
        if scores:
            print(f"  {row['transform']}: CASIA-trained ensemble-model face similarity " +
                  ', '.join(f'{value:.3f}' for value in scores), flush=True)
    print('Lower similarity means feature disruption on these optimization models, NOT a protection percentage.', flush=True)
    print(f'Saved: {out}\nReport: {report_path}' + (f'\nPreview: {preview_path}' if preview else '') +
          '\nTraining and likeness-generation protection remain untested.', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('images', nargs='*')
    parser.add_argument('--mode', choices=PRESETS, default='balanced')
    parser.add_argument('--steps', type=int)
    parser.add_argument('--max-side', type=int, help='Opt-in output downscaling; omitted preserves source resolution')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-preview', action='store_true', help='Do not create an original/processed comparison sheet')
    args = parser.parse_args()
    if args.steps is not None and args.steps < 1:
        parser.error('steps must be positive')
    if args.max_side is not None and not 32 <= args.max_side <= 8192:
        parser.error('max-side must be 32..8192')
    if not args.images:
        import tkinter as tk
        from tkinter.filedialog import askopenfilenames
        root = tk.Tk(); root.withdraw()
        args.images = askopenfilenames(title='Select photos for experimental privacy processing',
                                       filetypes=[('Images', '*.jpg *.jpeg *.png *.webp *.bmp')])
        root.destroy()
    failures = 0
    for path in args.images:
        try:
            process(path, args.mode, args.steps, args.seed, args.max_side, not args.no_preview)
        except Exception as exc:
            print(f'FAILED: {path}: {exc}', file=sys.stderr)
            failures += 1
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
