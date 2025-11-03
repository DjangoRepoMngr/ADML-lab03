import argparse
from pathlib import Path

import torch
from torch import nn

from data.tiny_imagenet import build_dataloaders
from models.custom_net import CustomNet


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
    parser = argparse.ArgumentParser(description="Evaluate CustomNet on Tiny-ImageNet")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    _, val_loader = build_dataloaders(
        args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers, use_cuda=device == "cuda"
    )

    model = CustomNet().to(device)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path.as_posix(), map_location=device))

    criterion = nn.CrossEntropyLoss()
    validate(model, val_loader, criterion, device)


if __name__ == "__main__":
    main()

