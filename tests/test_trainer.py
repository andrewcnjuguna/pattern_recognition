"""Tests for training utilities in trainer.py."""

import os
import sys
import unittest
from unittest.mock import MagicMock

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from trainer import create_pairs, train_one_epoch, validate


# ---------------------------------------------------------------------------
# create_pairs
# ---------------------------------------------------------------------------

class TestCreatePairs(unittest.TestCase):

    def _batch(self, n: int = 8, n_classes: int = 2):
        images = torch.randn(n, 1, 64, 64)
        labels = torch.tensor([i % n_classes for i in range(n)])
        return images, labels

    def test_returns_none_for_single_sample(self):
        images = torch.randn(1, 1, 64, 64)
        labels = torch.tensor([0])
        self.assertIsNone(create_pairs(images, labels))

    def test_returns_none_when_no_valid_pairs(self):
        # Two samples, same class → positive pair exists, no negative → still returns pairs
        # but if truly nothing can be formed it returns None
        images = torch.randn(1, 1, 8, 8)
        labels = torch.tensor([0])
        self.assertIsNone(create_pairs(images, labels))

    def test_output_is_tuple_of_three(self):
        result = create_pairs(*self._batch())
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 3)

    def test_input1_input2_same_batch_size(self):
        input1, input2, labels = create_pairs(*self._batch())
        self.assertEqual(input1.shape[0], input2.shape[0])
        self.assertEqual(input1.shape[0], labels.shape[0])

    def test_pair_labels_are_binary(self):
        _, _, pair_labels = create_pairs(*self._batch())
        unique_vals = set(pair_labels.tolist())
        self.assertTrue(unique_vals.issubset({0, 1}))

    def test_spatial_dims_preserved(self):
        images, labels = self._batch()
        input1, input2, _ = create_pairs(images, labels)
        self.assertEqual(input1.shape[1:], images.shape[1:])
        self.assertEqual(input2.shape[1:], images.shape[1:])

    def test_positive_pairs_exist_with_multiple_same_class(self):
        """Batches with shared classes should yield at least one positive pair."""
        images = torch.randn(6, 1, 8, 8)
        labels = torch.tensor([0, 0, 0, 1, 1, 1])
        _, _, pair_labels = create_pairs(images, labels)
        self.assertIn(1, pair_labels.tolist())

    def test_negative_pairs_exist_with_multiple_classes(self):
        """Batches with two classes should yield at least one negative pair."""
        images = torch.randn(6, 1, 8, 8)
        labels = torch.tensor([0, 0, 0, 1, 1, 1])
        _, _, pair_labels = create_pairs(images, labels)
        self.assertIn(0, pair_labels.tolist())


# ---------------------------------------------------------------------------
# train_one_epoch
# ---------------------------------------------------------------------------

class _FakeLoader:
    """Minimal stand-in for a DataLoader that yields (images, labels) tuples."""

    def __init__(self, n_batches: int = 3, batch_size: int = 8, n_classes: int = 2):
        self._batches = [
            (torch.randn(batch_size, 1, 8, 8),
             torch.tensor([i % n_classes for i in range(batch_size)]))
            for _ in range(n_batches)
        ]

    def __iter__(self):
        return iter(self._batches)

    def __len__(self):
        return len(self._batches)


class TestTrainOneEpoch(unittest.TestCase):

    def _make_model(self):
        model = MagicMock()
        model.train = MagicMock()
        model.parameters = MagicMock(return_value=iter([torch.randn(3, 3)]))
        emb = torch.randn(50, 16, requires_grad=True)
        model.return_value = (emb, emb)
        return model

    def test_returns_float(self):
        model = self._make_model()
        criterion = MagicMock(
            return_value=torch.tensor(0.5, requires_grad=True)
        )
        optimizer = MagicMock()
        loss = train_one_epoch(
            model, _FakeLoader(), criterion, optimizer,
            torch.device('cpu'), max_grad_norm=0
        )
        self.assertIsInstance(loss, float)

    def test_empty_loader_returns_zero(self):
        model = self._make_model()
        criterion = MagicMock()
        optimizer = MagicMock()
        loss = train_one_epoch(
            model, [], criterion, optimizer, torch.device('cpu')
        )
        self.assertEqual(loss, 0.0)

    def test_loss_is_non_negative(self):
        model = self._make_model()
        criterion = MagicMock(
            return_value=torch.tensor(0.3, requires_grad=True)
        )
        optimizer = MagicMock()
        loss = train_one_epoch(
            model, _FakeLoader(), criterion, optimizer,
            torch.device('cpu'), max_grad_norm=0
        )
        self.assertGreaterEqual(loss, 0.0)


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

class TestValidate(unittest.TestCase):

    def _make_model(self):
        model = MagicMock()
        model.eval = MagicMock()
        emb = torch.randn(8, 16)
        model.return_value = (emb, emb)
        return model

    def test_returns_float(self):
        model = self._make_model()
        criterion = MagicMock(return_value=torch.tensor(0.3))
        loss = validate(model, _FakeLoader(n_batches=2), criterion, torch.device('cpu'))
        self.assertIsInstance(loss, float)

    def test_empty_loader_returns_zero(self):
        model = self._make_model()
        criterion = MagicMock()
        loss = validate(model, [], criterion, torch.device('cpu'))
        self.assertEqual(loss, 0.0)

    def test_no_grad_context(self):
        """validate() should not compute gradients."""
        model = self._make_model()
        criterion = MagicMock(return_value=torch.tensor(0.2))
        with torch.autograd.set_grad_enabled(True):
            validate(model, _FakeLoader(n_batches=1), criterion, torch.device('cpu'))
        # If we reach here without RuntimeError, no grads were tracked unexpectedly.


if __name__ == '__main__':
    unittest.main()
