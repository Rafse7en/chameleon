import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
import torch
from torch.nn import functional as F

import Chameleon_Defense as d


class Tests(unittest.TestCase):
    def test_pixel_roundtrip(self):
        im = Image.new('RGB', (32, 32), (17, 80, 201))
        self.assertEqual(d.picture(d.tensor(im, 'cpu')).tobytes(), im.tobytes())

    def test_jpeg_ste_has_real_forward_and_gradient_for_batch(self):
        x = torch.rand(2, 3, 32, 32, requires_grad=True)
        y = d.jpeg(x, 85, True)
        self.assertTrue(torch.allclose(y, d.jpeg(x, 85), atol=1e-7))
        y.sum().backward()
        self.assertTrue(torch.equal(x.grad, torch.ones_like(x)))

    def test_optimization_changes_features_within_bound(self):
        x = torch.rand(1, 3, 32, 32) * .8 + .1

        def features(z):
            return [F.normalize(z.flatten(1), dim=1)]

        y = d.optimize(x, features, 8 / 255, 12, 42, progress=lambda *args, **kwargs: None)
        self.assertLessEqual(float((y - x).abs().max()), 8 / 255 + 1e-6)
        self.assertGreater(float(d.distance(features(y), features(x))), 0)
        self.assertTrue(bool(((y >= 0) & (y <= 1)).all()))

    def test_face_mask_keeps_background_exact(self):
        x = torch.rand(1, 3, 48, 48) * .8 + .1
        mask = torch.zeros(1, 1, 48, 48)
        mask[:, :, 12:36, 14:34] = 1

        def features(z):
            return [F.normalize(z.flatten(1), dim=1)]

        y = d.optimize(x, features, 5 / 255, 8, 7, mask=mask, progress=lambda *args, **kwargs: None)
        outside = (~mask.bool()).expand_as(x)
        self.assertTrue(torch.equal(x[outside], y[outside]))
        self.assertGreater(float((x - y).abs().max()), 0)

    def test_quality_gate_rejects_patterned_residual(self):
        original = torch.full((1, 3, 64, 64), .5)
        pattern = ((torch.arange(64)[:, None] + torch.arange(64)[None, :]) % 2) * 2 - 1
        candidate = original + pattern[None, None].repeat(1, 3, 1, 1) * (3 / 255)
        mask = torch.ones(1, 1, 64, 64)
        metrics = d.quality_metrics(original, candidate, mask)
        passed, failures = d.quality_passes(metrics, d.PRESETS['gentle'])
        self.assertFalse(passed)
        self.assertTrue(any('high-frequency' in failure for failure in failures))

    def test_existing_outputs_are_preserved_and_numbered(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'test.png'
            Image.new('RGB', (32, 32)).save(path)
            output = path.with_name('test_defense_balanced.png')
            output.write_bytes(b'keep')
            out, report, preview = d.choose_output_paths(path, 'balanced')
            self.assertEqual(output.read_bytes(), b'keep')
            self.assertEqual(out.name, 'test_defense_balanced_2.png')
            self.assertEqual(report.name, 'test_defense_balanced_2.json')
            self.assertEqual(preview.name, 'test_defense_balanced_2_preview.png')

    def test_no_face_is_resolution_preserving_clean_noop(self):
        class Detector:
            def detect(self, im, landmarks=False):
                return (None, None, None) if landmarks else (None, None)

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'gray.png'
            Image.new('RGB', (123, 77), (128, 64, 32)).save(path)
            with patch.object(d, 'MTCNN', return_value=Detector()), \
                 patch.object(d, 'Features') as features, patch.object(d, 'evaluate') as evaluate:
                report = d.process(path, steps=1)
                second_report = d.process(path, steps=1)
            output = path.with_name('gray_defense_balanced.png')
            with Image.open(output) as im:
                self.assertEqual(im.size, (123, 77))
                self.assertEqual(im.tobytes(), Image.open(path).convert('RGB').tobytes())
            self.assertEqual(report['status'], 'no_face_no_change')
            self.assertEqual(report['faces'], [])
            self.assertTrue(report['resolution_preserved'])
            self.assertEqual(report['training_protection'], 'not_tested')
            features.assert_not_called()
            evaluate.assert_not_called()
            saved = json.loads(path.with_name('gray_defense_balanced.json').read_text())
            self.assertEqual(saved['status'], 'no_face_no_change')
            self.assertTrue(path.with_name('gray_defense_balanced_preview.png').exists())
            self.assertEqual(second_report['status'], 'no_face_no_change')
            self.assertTrue(path.with_name('gray_defense_balanced_2.png').exists())
            self.assertTrue(path.with_name('gray_defense_balanced_2.json').exists())
            self.assertTrue(path.with_name('gray_defense_balanced_2_preview.png').exists())


if __name__ == '__main__':
    unittest.main()
