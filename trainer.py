"""Training utilities for the Siamese network.

Provides :func:`train_one_epoch`, :func:`validate`, and the high-level
:func:`train` function that wraps them with early stopping and model
checkpointing.  Separating these helpers from the entry-point script makes
it easy to write custom loops or integrate with experiment-tracking
frameworks (e.g. MLflow, W&B).

Typical usage::

    from trainer import train

    history = train(
        model, train_loader, val_loader,
        criterion, optimizer, scheduler,
        device, epochs=20, patience=5,
        save_path="best_model.pth",
    )
"""

import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Pair construction
# ---------------------------------------------------------------------------

def create_pairs(images: torch.Tensor, labels: torch.Tensor):
    """Build positive and negative image pairs from a labeled batch.

    For every sample *i* in the batch:

    * A **positive pair** ``(i, j)`` is formed with a randomly chosen index
      *j* that shares the same class label (skipped if none exists).
    * A **negative pair** ``(i, k)`` is formed with a randomly chosen index
      *k* from a different class (skipped if none exists).

    Args:
        images: Float tensor of shape ``(B, C, H, W)``.
        labels: Integer tensor of shape ``(B,)``.

    Returns:
        tuple[Tensor, Tensor, Tensor] | None:
        ``(input1, input2, pair_labels)`` where *pair_labels* is ``1`` for
        positive pairs and ``0`` for negative pairs.  Returns ``None`` when
        no valid pairs can be formed (e.g. batch size of 1).
    """
    pairs = []
    pair_labels = []

    for i in range(len(images)):
        same_class = torch.where(labels == labels[i])[0]
        if len(same_class) > 1:
            j = random.choice(same_class[same_class != i].tolist())
            pairs.append((images[i], images[j]))
            pair_labels.append(1)

        diff_class = torch.where(labels != labels[i])[0]
        if len(diff_class) > 0:
            j = random.choice(diff_class.tolist())
            pairs.append((images[i], images[j]))
            pair_labels.append(0)

    if not pairs:
        return None

    input1 = torch.stack([p[0] for p in pairs])
    input2 = torch.stack([p[1] for p in pairs])
    pair_labels_tensor = torch.tensor(pair_labels, device=labels.device)
    return input1, input2, pair_labels_tensor


# ---------------------------------------------------------------------------
# Single-epoch helpers
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_grad_norm: float = 1.0,
) -> float:
    """Run one full training epoch.

    Args:
        model: The Siamese network.
        loader: Training DataLoader that yields ``(images, labels)`` batches.
        criterion: Loss function, e.g. ``ContrastiveLoss()``.
        optimizer: PyTorch optimiser.
        device: Target compute device.
        max_grad_norm: Gradient-clipping max norm.  Pass ``0`` to disable.

    Returns:
        float: Average training loss over all processed batches.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.float().to(device)
        labels = labels.to(device)

        result = create_pairs(images, labels)
        if result is None:
            continue

        input1, input2, pair_labels = result
        optimizer.zero_grad()
        output1, output2 = model(input1, input2)
        loss = criterion(output1, output2, pair_labels)
        loss.backward()

        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        optimizer.step()
        total_loss += loss.item()
        num_batches += 1

        print(f"  Batch {batch_idx + 1}/{len(loader)}", end='\r')

    return total_loss / num_batches if num_batches > 0 else 0.0


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    device: torch.device,
) -> float:
    """Evaluate the model on a validation set.

    Args:
        model: The Siamese network.
        loader: Validation DataLoader that yields ``(images, labels)`` batches.
        criterion: Loss function.
        device: Target compute device.

    Returns:
        float: Average validation loss over all batches.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.float().to(device)
            labels = labels.to(device)

            rand_indices = torch.randperm(len(images))
            input1 = images
            input2 = images[rand_indices]
            pair_labels = (labels == labels[rand_indices]).float()

            output1, output2 = model(input1, input2)
            loss = criterion(output1, output2, pair_labels)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


# ---------------------------------------------------------------------------
# Full training loop
# ---------------------------------------------------------------------------

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    epochs: int = 10,
    patience: int = 5,
    save_path: str = 'best_model.pth',
) -> dict:
    """Full training loop with validation, early stopping, and checkpointing.

    Args:
        model: The Siamese network.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        criterion: Loss function.
        optimizer: PyTorch optimiser.
        scheduler: Learning-rate scheduler; ``scheduler.step()`` is called
            once per epoch after validation.
        device: Compute device.
        epochs: Maximum number of training epochs.
        patience: Early-stopping patience — number of epochs allowed without
            validation-loss improvement before training is halted.
        save_path: File path where the best model checkpoint is saved.

    Returns:
        dict: Training history with keys ``"train_loss"`` and ``"val_loss"``,
        each mapping to a list of per-epoch averages.
    """
    history: dict = {"train_loss": [], "val_loss": []}
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        avg_train = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        avg_val = validate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)

        print(
            f"\nEpoch {epoch + 1}/{epochs} | "
            f"Train: {avg_train:.4f} | Val: {avg_val:.4f}"
        )

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"  Saved best model → {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    return history
