# pattern_recognition

Siamese network for image pattern recognition.  The model learns an embedding
space where images of the same pattern cluster together, enabling distance-based
matching without re-training when new patterns are added.

## Project layout

```
pattern_recognition/
├── main.py                  # CLI training entry point
├── trainer.py               # train(), train_one_epoch(), validate(), create_pairs()
├── sampler.py               # BalancedBatchSampler
├── predictor.py             # ONNXSiamesePredictor + preprocess_image()
├── scripts/
│   ├── model.py             # SiameseNetwork definition
│   ├── loss.py              # ContrastiveLoss
│   └── utils.py             # SiameseDataset, calculate_mean_std, visualize_embeddings
└── tests/
    ├── test_predictor.py
    ├── test_sampler.py
    └── test_trainer.py
```

## Installation

```bash
pip install torch torchvision pillow numpy onnxruntime scikit-learn matplotlib
```

## Training

```bash
python main.py --data /path/to/dataset
```

All hyperparameters have sensible defaults and can be overridden:

| Flag | Default | Description |
|------|---------|-------------|
| `--data` | *(required)* | Path to the dataset directory |
| `--epochs` | 10 | Maximum training epochs |
| `--batch-size` | 30 | Batch size for the validation loader |
| `--lr` | 0.0001 | Adam learning rate |
| `--weight-decay` | 1e-5 | Adam weight decay |
| `--n-classes` | 2 | Classes per batch (balanced sampler) |
| `--n-samples` | 8 | Samples per class per batch (balanced sampler) |
| `--patience` | 5 | Early-stopping patience in epochs |
| `--save` | best_model.pth | Path for the best-model checkpoint |

Example with custom settings:

```bash
python main.py \
  --data ./data/dataset \
  --epochs 50 \
  --lr 5e-5 \
  --n-classes 4 \
  --n-samples 16 \
  --save models/run1.pth
```

## Inference (ONNX)

Export your PyTorch model to ONNX (see `scripts/`) then use
`ONNXSiamesePredictor` for dependency-light deployment:

```python
from predictor import ONNXSiamesePredictor

predictor = ONNXSiamesePredictor("siamese_network.onnx")

# Find the closest reference pattern
idx, distance = predictor.predict_pattern(
    "query_image.png",
    "reference_patterns/",
)
print(f"Best match: pattern #{idx}  (distance {distance:.4f})")

# Inspect all distances, sorted closest-first
for filename, dist in predictor.get_all_distances("query_image.png", "reference_patterns/"):
    print(f"  {filename}: {dist:.4f}")
```

### Preprocessing contract

`preprocess_image` is exposed as a module-level function so it can be reused
or tested independently:

```python
from predictor import preprocess_image
from PIL import Image

arr = preprocess_image(Image.open("my_image.png"))
# arr.shape == (1, 1, 64, 64), dtype float32, values in [-1, 1]
```

## Reusable training utilities

`trainer.py` exposes each phase of the loop separately:

```python
from trainer import train_one_epoch, validate, train

# Custom loop
for epoch in range(20):
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss   = validate(model, val_loader, criterion, device)
    scheduler.step()

# Or use the batteries-included wrapper
history = train(
    model, train_loader, val_loader,
    criterion, optimizer, scheduler, device,
    epochs=20, patience=5, save_path="best.pth",
)
```

## Balanced batch sampler

`BalancedBatchSampler` ensures every mini-batch contains `n_classes × n_samples`
items with equal class representation — important for contrastive losses:

```python
from sampler import BalancedBatchSampler
from torch.utils.data import DataLoader

train_labels = [dataset.labels[i] for i in train_subset.indices]
sampler = BalancedBatchSampler(train_labels, n_classes=4, n_samples=8)
loader  = DataLoader(train_subset, batch_sampler=sampler)
```

Classes with fewer samples than `n_samples` are padded via replacement
sampling so the batch size stays constant.

## Running tests

```bash
python -m unittest discover -s tests -v
```

All tests use the standard library `unittest` module with `unittest.mock` for
ONNX sessions and PyTorch models, so no GPU or trained weights are required.
