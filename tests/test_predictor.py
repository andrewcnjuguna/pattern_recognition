"""Tests for ONNXSiamesePredictor and preprocessing helpers in predictor.py."""

import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

# Ensure the package root is on the path when running tests directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from predictor import (
    ONNXSiamesePredictor,
    _is_image_file,
    preprocess_image,
)


# ---------------------------------------------------------------------------
# _is_image_file
# ---------------------------------------------------------------------------

class TestIsImageFile(unittest.TestCase):

    def test_png_returns_true(self):
        self.assertTrue(_is_image_file("photo.png"))

    def test_jpg_returns_true(self):
        self.assertTrue(_is_image_file("photo.jpg"))

    def test_jpeg_returns_true(self):
        self.assertTrue(_is_image_file("photo.jpeg"))

    def test_uppercase_extension(self):
        self.assertTrue(_is_image_file("PHOTO.PNG"))

    def test_mixed_case_extension(self):
        self.assertTrue(_is_image_file("Photo.Jpg"))

    def test_text_file_returns_false(self):
        self.assertFalse(_is_image_file("README.txt"))

    def test_no_extension_returns_false(self):
        self.assertFalse(_is_image_file("datafile"))

    def test_onnx_returns_false(self):
        self.assertFalse(_is_image_file("model.onnx"))


# ---------------------------------------------------------------------------
# preprocess_image
# ---------------------------------------------------------------------------

class TestPreprocessImage(unittest.TestCase):

    def _rgb_image(self, size=(100, 80)):
        return Image.new("RGB", size, color=(128, 64, 200))

    def _gray_image(self, size=(100, 80), value=128):
        return Image.new("L", size, color=value)

    def test_default_output_shape(self):
        arr = preprocess_image(self._rgb_image())
        self.assertEqual(arr.shape, (1, 1, 64, 64))

    def test_output_dtype_is_float32(self):
        arr = preprocess_image(self._gray_image())
        self.assertEqual(arr.dtype, np.float32)

    def test_normalisation_range(self):
        arr = preprocess_image(self._gray_image())
        self.assertGreaterEqual(float(arr.min()), -1.0 - 1e-5)
        self.assertLessEqual(float(arr.max()), 1.0 + 1e-5)

    def test_custom_size(self):
        arr = preprocess_image(self._rgb_image(), size=(32, 32))
        self.assertEqual(arr.shape, (1, 1, 32, 32))

    def test_rgb_converted_to_single_channel(self):
        arr = preprocess_image(self._rgb_image())
        self.assertEqual(arr.shape[1], 1)

    def test_black_pixel_maps_to_minus_one(self):
        arr = preprocess_image(self._gray_image(value=0))
        self.assertAlmostEqual(float(arr.min()), -1.0, places=4)

    def test_white_pixel_maps_near_plus_one(self):
        arr = preprocess_image(self._gray_image(value=255))
        self.assertAlmostEqual(float(arr.max()), 1.0, places=2)

    def test_grayscale_image_unchanged_mode(self):
        """Grayscale input should be accepted without error."""
        arr = preprocess_image(self._gray_image())
        self.assertIsNotNone(arr)


# ---------------------------------------------------------------------------
# ONNXSiamesePredictor — construction
# ---------------------------------------------------------------------------

class TestONNXSiamesePredictorInit(unittest.TestCase):

    def test_missing_model_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            ONNXSiamesePredictor("/nonexistent/model.onnx")

    def test_valid_path_loads_session(self):
        with tempfile.NamedTemporaryFile(suffix='.onnx') as f:
            with patch('predictor.ort.InferenceSession') as mock_cls:
                mock_cls.return_value = MagicMock()
                mock_cls.return_value.get_inputs.return_value = []
                predictor = ONNXSiamesePredictor(f.name)
                self.assertIsNotNone(predictor.session)


# ---------------------------------------------------------------------------
# ONNXSiamesePredictor — predict_pattern & get_all_distances
# ---------------------------------------------------------------------------

class TestONNXSiamesePredictorPredict(unittest.TestCase):
    """Uses a mocked ONNX session so no real model file is needed."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.ref_dir = tempfile.mkdtemp()

        # Dummy query image
        self.query_path = os.path.join(self.tmp_dir, "query.png")
        Image.new("L", (64, 64), 128).save(self.query_path)

        # Two reference images
        for name in ["pattern_0.png", "pattern_1.png"]:
            Image.new("L", (64, 64), 64).save(os.path.join(self.ref_dir, name))

        # Empty file to act as the ONNX path (FileNotFoundError check needs a file)
        self.fake_onnx = os.path.join(self.tmp_dir, "model.onnx")
        open(self.fake_onnx, 'w').close()

    def _make_predictor(self, embeddings=None):
        """Return a predictor whose session.run returns *embeddings* in order."""
        call_count = [0]

        def fake_run(_output_names, _feed):
            val = float(call_count[0] + 1)
            call_count[0] += 1
            if embeddings and call_count[0] <= len(embeddings):
                return [np.array([embeddings[call_count[0] - 1]], dtype=np.float32)]
            return [np.array([[val, val, val]], dtype=np.float32)]

        mock_session = MagicMock()
        mock_inp1 = MagicMock(); mock_inp1.name = "input1"
        mock_inp2 = MagicMock(); mock_inp2.name = "input2"
        mock_session.get_inputs.return_value = [mock_inp1, mock_inp2]
        mock_session.run.side_effect = fake_run

        with patch('predictor.ort.InferenceSession', return_value=mock_session):
            predictor = ONNXSiamesePredictor(self.fake_onnx)
        predictor.session = mock_session
        return predictor

    def test_predict_returns_valid_index(self):
        predictor = self._make_predictor()
        idx, _ = predictor.predict_pattern(self.query_path, self.ref_dir)
        self.assertIn(idx, [0, 1])

    def test_predict_distance_is_non_negative(self):
        predictor = self._make_predictor()
        _, dist = predictor.predict_pattern(self.query_path, self.ref_dir)
        self.assertGreaterEqual(dist, 0.0)

    def test_predict_selects_closest_embedding(self):
        # Query embedding = [1, 1, 1], ref0 = [1, 1, 1], ref1 = [10, 10, 10]
        # Distance to ref0 should be 0.0 (or very small), so idx must be 0.
        embeddings = [
            [1.0, 1.0, 1.0],   # query
            [1.0, 1.0, 1.0],   # pattern_0 (closest)
            [10.0, 10.0, 10.0],  # pattern_1
        ]
        predictor = self._make_predictor(embeddings=embeddings)
        idx, dist = predictor.predict_pattern(self.query_path, self.ref_dir)
        self.assertEqual(idx, 0)
        self.assertAlmostEqual(dist, 0.0, places=4)

    def test_get_all_distances_sorted(self):
        predictor = self._make_predictor()
        results = predictor.get_all_distances(self.query_path, self.ref_dir)
        distances = [d for _, d in results]
        self.assertEqual(distances, sorted(distances))

    def test_get_all_distances_covers_all_images(self):
        predictor = self._make_predictor()
        results = predictor.get_all_distances(self.query_path, self.ref_dir)
        self.assertEqual(len(results), 2)

    def test_get_all_distances_returns_filenames(self):
        predictor = self._make_predictor()
        results = predictor.get_all_distances(self.query_path, self.ref_dir)
        filenames = [f for f, _ in results]
        self.assertIn("pattern_0.png", filenames)
        self.assertIn("pattern_1.png", filenames)


if __name__ == '__main__':
    unittest.main()
