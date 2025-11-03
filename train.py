import argparse
import os
from pathlib import Path

import torch
from torch import nn

from data.tiny_imagenet import build_dataloaders
from models.custom_net import CustomNet


def train_one_epoch(epoch: int, model: nn.Module, dataloader, criterion, optimizer, device: str) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss = running_loss / len(dataloader)
    train_acc = 100.0 * correct / total
    print(f"Train Epoch: {epoch} Loss: {train_loss:.6f} Acc: {train_acc:.2f}%")
    return train_loss, train_acc


@torch.no_grad()
def validate(model: nn.Module, dataloader, criterion, device: str) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    val_loss = running_loss / len(dataloader)
    val_acc = 100.0 * correct / total
    print(f"Validation Loss: {val_loss:.6f} Acc: {val_acc:.2f}%")
    return val_loss, val_acc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CustomNet on Tiny-ImageNet")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to tiny-imagenet-200 root directory")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    return parser.parse_args()


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    torch.backends.cudnn.benchmark = device == "cuda"

    train_loader, val_loader = build_dataloaders(
        args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers, use_cuda=device == "cuda"
    )

    model = CustomNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    best_acc = 0.0
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best_model.pth"

    for epoch in range(1, args.epochs + 1):
        train_one_epoch(epoch, model, train_loader, criterion, optimizer, device)
        _, val_acc = validate(model, val_loader, criterion, device)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path.as_posix())
            print("Saved new best model!", best_path)

    print(f"Best validation accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()

