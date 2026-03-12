"""Training entry point for the Siamese pattern recognition network.

All hyperparameters are configurable via command-line arguments so no source
edits are needed between runs.

Usage::

    python main.py --data /path/to/dataset
    python main.py --data /path/to/dataset --epochs 20 --lr 0.0001 --save best.pth
"""

import argparse

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from scripts.model import SiameseNetwork
from scripts.loss import ContrastiveLoss
from scripts.utils import SiameseDataset, calculate_mean_std
from sampler import BalancedBatchSampler
from trainer import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Siamese pattern recognition network"
    )
    parser.add_argument('--data', required=True,
                        help='Path to the dataset directory')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Maximum training epochs (default: 10)')
    parser.add_argument('--batch-size', type=int, default=30,
                        help='Batch size for validation loader (default: 30)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate (default: 0.0001)')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Adam weight decay (default: 1e-5)')
    parser.add_argument('--n-classes', type=int, default=2,
                        help='Classes per batch for balanced sampler (default: 2)')
    parser.add_argument('--n-samples', type=int, default=8,
                        help='Samples per class per batch (default: 8)')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early-stopping patience in epochs (default: 5)')
    parser.add_argument('--save', default='best_model.pth',
                        help='Path to save the best model checkpoint')
    return parser.parse_args()


def build_dataloaders(
    dataset_path: str,
    batch_size: int,
    n_classes: int,
    n_samples: int,
):
    """Construct train and validation DataLoaders from *dataset_path*.

    Args:
        dataset_path: Root directory of the dataset.
        batch_size: Batch size used for the validation loader.
        n_classes: Classes per batch for :class:`~sampler.BalancedBatchSampler`.
        n_samples: Samples per class per batch for the balanced sampler.

    Returns:
        tuple[DataLoader, DataLoader]: ``(train_loader, val_loader)``.
    """
    dataset_mean, dataset_std = calculate_mean_std(dataset_path)

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize(mean=[dataset_mean], std=[dataset_std]),
    ])

    dataset = SiameseDataset(dataset_path, transform=transform)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_labels = [dataset.labels[i] for i in train_dataset.indices]
    train_sampler = BalancedBatchSampler(
        train_labels, n_classes=n_classes, n_samples=n_samples
    )

    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader = build_dataloaders(
        args.data, args.batch_size, args.n_classes, args.n_samples
    )

    model = SiameseNetwork().to(device)
    criterion = ContrastiveLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs // 2, eta_min=1e-6
    )

    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=args.epochs,
        patience=args.patience,
        save_path=args.save,
    )


if __name__ == '__main__':
    main()
