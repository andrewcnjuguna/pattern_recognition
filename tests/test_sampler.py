"""Tests for BalancedBatchSampler in sampler.py."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sampler import BalancedBatchSampler


def make_labels(n_classes: int = 4, n_per_class: int = 20):
    return [cls for cls in range(n_classes) for _ in range(n_per_class)]


class TestBalancedBatchSamplerBatchSize(unittest.TestCase):

    def test_every_batch_has_correct_size(self):
        labels = make_labels(n_classes=4, n_per_class=20)
        sampler = BalancedBatchSampler(labels, n_classes=2, n_samples=4)
        for batch in sampler:
            self.assertEqual(len(batch), 2 * 4)

    def test_single_class_per_batch(self):
        labels = make_labels(n_classes=4, n_per_class=20)
        sampler = BalancedBatchSampler(labels, n_classes=1, n_samples=5)
        for batch in sampler:
            self.assertEqual(len(batch), 5)


class TestBalancedBatchSamplerIndices(unittest.TestCase):

    def test_all_indices_in_valid_range(self):
        labels = make_labels()
        sampler = BalancedBatchSampler(labels, n_classes=2, n_samples=4)
        for batch in sampler:
            for idx in batch:
                self.assertGreaterEqual(idx, 0)
                self.assertLess(idx, len(labels))

    def test_each_batch_has_correct_number_of_classes(self):
        labels = make_labels(n_classes=6, n_per_class=10)
        sampler = BalancedBatchSampler(labels, n_classes=3, n_samples=4)
        for batch in sampler:
            batch_classes = {labels[i] for i in batch}
            self.assertEqual(len(batch_classes), 3)


class TestBalancedBatchSamplerLen(unittest.TestCase):

    def test_len_matches_expected(self):
        # 100 labels, batch_size = 2 * 5 = 10 → 10 batches
        labels = list(range(100))
        sampler = BalancedBatchSampler(labels, n_classes=2, n_samples=5)
        self.assertEqual(len(sampler), 10)

    def test_len_is_positive(self):
        labels = make_labels(n_classes=4, n_per_class=20)
        sampler = BalancedBatchSampler(labels, n_classes=2, n_samples=4)
        self.assertGreater(len(sampler), 0)

    def test_iteration_count_matches_len(self):
        labels = make_labels(n_classes=4, n_per_class=20)
        sampler = BalancedBatchSampler(labels, n_classes=2, n_samples=4)
        batches = list(sampler)
        self.assertEqual(len(batches), len(sampler))


class TestBalancedBatchSamplerEdgeCases(unittest.TestCase):

    def test_small_class_does_not_crash(self):
        """A class with fewer than n_samples examples should be padded without error."""
        labels = [0] * 3 + [1] * 20   # class 0 has only 3 samples
        sampler = BalancedBatchSampler(labels, n_classes=2, n_samples=8)
        batches = list(sampler)
        self.assertGreater(len(batches), 0)
        for batch in batches:
            self.assertEqual(len(batch), 16)

    def test_single_sample_class_padded_correctly(self):
        """A class with exactly 1 sample should fill the entire class slot via replacement."""
        labels = [0] * 1 + [1] * 30
        sampler = BalancedBatchSampler(labels, n_classes=2, n_samples=5)
        for batch in sampler:
            self.assertEqual(len(batch), 10)
            break  # one batch is enough

    def test_produces_no_duplicate_classes_beyond_n_classes(self):
        """Each batch should draw from exactly n_classes distinct classes."""
        labels = make_labels(n_classes=5, n_per_class=12)
        sampler = BalancedBatchSampler(labels, n_classes=2, n_samples=4)
        for batch in sampler:
            batch_classes = {labels[i] for i in batch}
            self.assertEqual(len(batch_classes), 2)


if __name__ == '__main__':
    unittest.main()
