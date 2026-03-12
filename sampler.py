"""Balanced batch sampler for Siamese / metric-learning training.

Guarantees that each mini-batch contains exactly *n_samples* examples from
each of *n_classes* randomly selected classes.  This is important for
contrastive and triplet losses where pair/triplet quality depends on having
multiple samples from the same class in every batch.

Typical usage::

    from sampler import BalancedBatchSampler
    from torch.utils.data import DataLoader

    sampler = BalancedBatchSampler(dataset.labels, n_classes=4, n_samples=8)
    loader = DataLoader(dataset, batch_sampler=sampler)
"""

import random

from torch.utils.data import BatchSampler


class BalancedBatchSampler(BatchSampler):
    """Yield balanced batches for metric learning.

    Each batch contains ``n_classes × n_samples`` indices such that every
    class in the batch is represented by exactly *n_samples* examples.
    When a class has fewer than *n_samples* examples available, the deficit
    is filled by sampling with replacement from that class's indices so the
    batch size stays constant.

    Args:
        labels: Sequence of integer class labels, one per dataset sample.
        n_classes: Number of distinct classes per batch.
        n_samples: Number of samples per class per batch.

    Example::

        labels = [dataset.labels[i] for i in train_subset.indices]
        sampler = BalancedBatchSampler(labels, n_classes=2, n_samples=8)
        loader = DataLoader(train_subset, batch_sampler=sampler)
    """

    def __init__(self, labels, n_classes: int = 2, n_samples: int = 16):
        self.labels = list(labels)
        self.labels_set = list(set(self.labels))
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.batch_size = n_classes * n_samples

        # Pre-index labels once to avoid O(N) scans inside __iter__
        self._label_to_indices = {
            label: [i for i, l in enumerate(self.labels) if l == label]
            for label in self.labels_set
        }

    # ------------------------------------------------------------------
    # BatchSampler interface
    # ------------------------------------------------------------------

    def __iter__(self):
        for _ in range(len(self)):
            classes = random.sample(self.labels_set, self.n_classes)
            indices = []
            for cls in classes:
                cls_indices = self._label_to_indices[cls]
                available = min(len(cls_indices), self.n_samples)
                selected = random.sample(cls_indices, available)
                if len(selected) < self.n_samples:
                    # Fill deficit with replacement sampling
                    selected += random.choices(
                        cls_indices, k=self.n_samples - len(selected)
                    )
                indices.extend(selected)
            random.shuffle(indices)
            yield indices

    def __len__(self) -> int:
        return len(self.labels) // self.batch_size
