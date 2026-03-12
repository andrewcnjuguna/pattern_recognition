"""ONNX-based Siamese Network predictor for pattern recognition.

This module provides the :class:`ONNXSiamesePredictor` class, which loads a
Siamese network exported to ONNX format and uses it to match an input image
against a folder of reference patterns by comparing embedding distances.

It also exposes the :func:`preprocess_image` helper so callers can reuse the
exact same preprocessing pipeline outside of the predictor class.

Typical usage::

    from predictor import ONNXSiamesePredictor

    predictor = ONNXSiamesePredictor("siamese_network.onnx")
    idx, dist = predictor.predict_pattern("query.png", "reference_patterns/")
    print(f"Best match: pattern #{idx}  (distance {dist:.4f})")
"""

import os

import numpy as np
import onnxruntime as ort
from PIL import Image

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg')
DEFAULT_IMAGE_SIZE = (64, 64)


class ONNXSiamesePredictor:
    """Siamese network predictor backed by an ONNX Runtime session.

    The predictor converts each image to an embedding vector by running one
    forward pass through the exported network, then compares embeddings with
    Euclidean distance to find the closest reference pattern.

    Args:
        onnx_path: Path to the exported ``.onnx`` model file.
        image_size: ``(width, height)`` to resize images before inference.
            Defaults to ``(64, 64)`` to match training transforms.

    Raises:
        FileNotFoundError: If *onnx_path* does not point to an existing file.
    """

    def __init__(self, onnx_path: str, image_size: tuple = DEFAULT_IMAGE_SIZE):
        if not os.path.isfile(onnx_path):
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
        self.session = ort.InferenceSession(onnx_path)
        self.image_size = image_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_pattern(self, new_image_path: str, reference_patterns_path: str):
        """Find the reference pattern most similar to *new_image_path*.

        Args:
            new_image_path: Path to the query image.
            reference_patterns_path: Directory containing reference pattern
                images (``*.png``, ``*.jpg``, ``*.jpeg``).

        Returns:
            tuple[int | None, float]: A ``(pattern_index, distance)`` pair
            where *pattern_index* is the 0-based index of the best-matching
            reference file (sorted alphabetically) and *distance* is the
            Euclidean distance between embeddings.  Returns ``(None, inf)``
            if no valid reference images are found.
        """
        new_embedding = self._get_embedding(new_image_path)

        min_distance = float('inf')
        predicted_pattern = None

        for pattern_number, pattern_file in enumerate(
            sorted(os.listdir(reference_patterns_path))
        ):
            if not _is_image_file(pattern_file):
                continue

            pattern_path = os.path.join(reference_patterns_path, pattern_file)
            pattern_embedding = self._get_embedding(pattern_path)

            distance = float(np.linalg.norm(new_embedding - pattern_embedding))
            if distance < min_distance:
                min_distance = distance
                predicted_pattern = pattern_number

        return predicted_pattern, min_distance

    def get_all_distances(self, new_image_path: str, reference_patterns_path: str):
        """Return distances from *new_image_path* to every reference pattern.

        Args:
            new_image_path: Path to the query image.
            reference_patterns_path: Directory of reference pattern images.

        Returns:
            list[tuple[str, float]]: Sorted list of ``(filename, distance)``
            pairs, closest match first.
        """
        new_embedding = self._get_embedding(new_image_path)
        results = []

        for pattern_file in sorted(os.listdir(reference_patterns_path)):
            if not _is_image_file(pattern_file):
                continue
            pattern_path = os.path.join(reference_patterns_path, pattern_file)
            pattern_embedding = self._get_embedding(pattern_path)
            distance = float(np.linalg.norm(new_embedding - pattern_embedding))
            results.append((pattern_file, distance))

        results.sort(key=lambda x: x[1])
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_embedding(self, image_path: str) -> np.ndarray:
        """Return the embedding vector for a single image file."""
        img = Image.open(image_path)
        arr = preprocess_image(img, size=self.image_size)
        return self._run_inference(arr)

    def _run_inference(self, arr: np.ndarray) -> np.ndarray:
        """Run a single-image forward pass and return the embedding.

        The same image is passed as both Siamese inputs; only the first
        output (embedding of the first branch) is returned.
        """
        input_names = [inp.name for inp in self.session.get_inputs()]
        feed = {name: arr for name in input_names}
        outputs = self.session.run(None, feed)
        return outputs[0][0]


# ---------------------------------------------------------------------------
# Module-level helpers (importable and testable independently)
# ---------------------------------------------------------------------------

def preprocess_image(img: "Image.Image", size: tuple = DEFAULT_IMAGE_SIZE) -> np.ndarray:
    """Preprocess a PIL image into a model-ready NumPy array.

    Applies the same pipeline used during training:

    1. Resize to *size* with bilinear interpolation.
    2. Convert to grayscale (``"L"`` mode).
    3. Normalise pixel values from ``[0, 255]`` to ``[-1, 1]``.
    4. Add batch and channel dimensions → shape ``(1, 1, H, W)``.

    Args:
        img: A PIL ``Image`` object (any mode).
        size: ``(width, height)`` target size.  Defaults to ``(64, 64)``.

    Returns:
        NumPy ``float32`` array of shape ``(1, 1, H, W)``.
    """
    img = img.resize(size, Image.BILINEAR)
    if img.mode != 'L':
        img = img.convert('L')
    arr = np.array(img, dtype=np.float32)
    arr = (arr / 127.5) - 1.0           # normalise to [-1, 1]
    arr = arr[np.newaxis, np.newaxis]   # (H, W) -> (1, 1, H, W)
    return arr


def _is_image_file(filename: str) -> bool:
    """Return ``True`` if *filename* has a recognised image extension."""
    return filename.lower().endswith(IMAGE_EXTENSIONS)
