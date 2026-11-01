import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('engine', ROOT / 'Chameleon_Engine.py')
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)

class AuditTests(unittest.TestCase):
    def setUp(self):
        self.img = np.full((64, 64, 3), 128, np.uint8)

    def test_missing_and_corrupt_models_are_unavailable(self):
        self.assertFalse(e.verify_faces(self.img, self.img, None)['available'])
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'broken.onnx'
            p.write_bytes(b'not a model')
            self.assertFalse(e.verify_faces(self.img, self.img, str(p))['available'])

    def test_failed_after_check_cannot_claim_reduction(self):
        with patch.object(e, 'detect_faces_yunet', side_effect=[[(0, 0, 20, 20, .9)], RuntimeError('failed')]):
            result = e.verify_faces(self.img, self.img, 'model')
        self.assertFalse(result['available'])
        self.assertNotIn('detection_reduced', result)

    def test_no_baseline_is_not_reduction(self):
        with patch.object(e, 'detect_faces_yunet', return_value=[]):
            result = e.verify_faces(self.img, self.img, 'model')
        self.assertFalse(result['baseline_detected'])
        self.assertIs(result['detection_reduced'], False)
        self.assertEqual(result['training_protection'], 'not_tested')

    def test_detected_reduction_is_only_detection(self):
        with patch.object(e, 'detect_faces_yunet', side_effect=[[(0, 0, 20, 20, .9)], []]):
            result = e.verify_faces(self.img, self.img, 'model')
        self.assertTrue(result['detection_reduced'])
        self.assertEqual(result['scope'], 'face_detection_only')
        self.assertNotIn('neutralized', result)

    def test_original_preserved_and_saved_output_read(self):
        with tempfile.TemporaryDirectory() as td:
            src, dst = Path(td) / 'input.png', Path(td) / 'output.jpg'
            im = Image.fromarray(self.img)
            exif = Image.Exif(); exif[315] = 'synthetic test'
            im.save(src, exif=exif)
            original = src.read_bytes()
            with self.assertRaises(ValueError):
                e.process_image(str(src), str(src))
            with patch.object(e, 'ensure_yunet_model', side_effect=AssertionError('unnecessary download')):
                result = e.process_image(str(src), str(dst), seed=123)
            self.assertTrue(result['success'])
            self.assertEqual(result['training_protection'], 'unverified')
            self.assertEqual(src.read_bytes(), original)
            with Image.open(dst) as out:
                self.assertFalse(out.getexif())
            real_loader = e.load_image
            with patch.object(e, 'load_image', side_effect=[real_loader(str(src)), RuntimeError('unreadable')]):
                result = e.process_image(str(src), str(dst), seed=123)
            self.assertFalse(result['success'])

    def test_adaptive_keeps_prior_image_when_candidates_worse(self):
        with tempfile.TemporaryDirectory() as td:
            src, dst = Path(td) / 'input.png', Path(td) / 'output.png'
            Image.fromarray(self.img).save(src)
            model = ROOT / 'Chameleon_Toolkit' / 'face_detection_yunet_2023mar.onnx'
            with patch.object(e, 'ensure_yunet_model', return_value=str(model)), \
                 patch.object(e, 'estimate_face_boxes_full', return_value=([(0,0,64,64)], 'test', [])), \
                 patch.object(e, 'detect_faces_yunet', side_effect=[[(0,0,20,20,.8)], [(0,0,20,20,.9)], [(0,0,20,20,.95)]]):
                result = e.process_image(str(src), str(dst), seed=123, adaptive=True)
            self.assertEqual(result['adaptive_rounds'], 0)

if __name__ == '__main__':
    unittest.main()
