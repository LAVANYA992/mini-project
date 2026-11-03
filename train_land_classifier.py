"""
train_land_classifier.py

Layer 2 (Land Image Analysis) — MobileNetV3 fine-tuned on the **EuroSAT RGB**
dataset (10 land-use classes, Sentinel-2 imagery, 64x64 RGB JPEGs).

Uses torchvision's built-in `torchvision.datasets.EuroSAT` loader, which
downloads and extracts the official EuroSAT RGB dataset automatically
(no manual zip handling needed) — this is the standard, citable EuroSAT
RGB release referenced by the project brief:
    Helber et al., "EuroSAT: A Novel Dataset and Deep Learning Benchmark
    for Land Use and Land Cover Classification", 2019.
    https://github.com/phelber/EuroSAT

Output of this stage feeds Layer 3 (Unified Agricultural Feature Set) via
two derived signals per config.LAND_CONDITION_MAP:
    - land_condition:            dry / wet / moist / vegetated / not_suitable
    - suitable_for_agriculture:  bool

NOTE: This script requires torch + torchvision and ~2GB of disk for the
downloaded dataset + a GPU is strongly recommended. It was written and
reviewed for correctness but NOT executed in the assistant's sandbox
(no GPU, no disk headroom, and the dataset host is outside the sandbox's
network allowlist) — run it locally or on Colab/Kaggle.

Run:
    python -m models.train_land_classifier --epochs 10 --batch-size 32
    python -m models.train_land_classifier --epochs 3 --freeze-backbone   # fast CPU run
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import copy
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from torchvision.datasets import EuroSAT

import config


def build_dataloaders(data_dir: str, batch_size: int, val_split: float = 0.2, img_size: int = 224):
    # MobileNetV3 (ImageNet-pretrained) normalization stats
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),        # EuroSAT tiles are 64x64 -> upsample for MobileNetV3
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),                  # satellite imagery has no canonical "up"
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        normalize,
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ])

    # torchvision downloads + extracts the official EuroSAT RGB split automatically.
    full_dataset = EuroSAT(root=data_dir, download=True, transform=train_transform)
    class_names = full_dataset.classes  # matches config.EUROSAT_CLASSES

    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(config.RANDOM_STATE),
    )
    # Validation subset should use eval-time transforms, not train-time augmentation.
    eval_dataset = EuroSAT(root=data_dir, download=False, transform=eval_transform)
    val_ds.dataset = eval_dataset

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, class_names


def build_model(num_classes: int) -> nn.Module:
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, running_correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        running_correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return running_loss / total, running_correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, running_correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        running_correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return running_loss / total, running_correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=config.EUROSAT_DIR,
                         help="Where torchvision should download/extract EuroSAT RGB.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--freeze-backbone", action="store_true",
                         help="Only train the classifier head (fast, good for CPU/small GPUs)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.data_dir, exist_ok=True)

    train_loader, val_loader, class_names = build_dataloaders(args.data_dir, args.batch_size)
    print(f"EuroSAT RGB — {len(train_loader.dataset)} train / {len(val_loader.dataset)} val images")
    print(f"Classes ({len(class_names)}): {class_names}")

    model = build_model(num_classes=len(class_names)).to(device)

    if args.freeze_backbone:
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
        print("Backbone frozen — training classifier head only.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_acc, best_state = 0.0, None
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | {elapsed:.1f}s")

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "best_val_acc": best_acc,
    }, config.LAND_CLASSIFIER_MODEL_PATH)
    print(f"\nBest val accuracy: {best_acc:.4f}")
    print(f"Saved model -> {config.LAND_CLASSIFIER_MODEL_PATH}")


if __name__ == "__main__":
    main()
