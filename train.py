import os
import argparse
import torch
from torch import nn
from torch.optim import SGD

from data.datasets import get_datasets
from data.loaders import get_loaders
from models.customnet import CustomNet
from utils.wandb_logger import WB


def train_one_epoch(epoch, model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in train_loader:
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

    train_loss = running_loss / len(train_loader)
    train_acc = 100.0 * correct / total
    print(f"Train Epoch: {epoch} Loss: {train_loss:.6f} Acc: {train_acc:.2f}%")
    return train_loss, train_acc


def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss /= len(val_loader)
    val_acc = 100.0 * correct / total
    print(f"Validation Loss: {val_loss:.6f} Acc: {val_acc:.2f}%")
    return val_loss, val_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', type=str, default='dataset/tiny-imagenet-200')
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--momentum', type=float, default=0.9)
    ap.add_argument('--num_workers', type=int, default=2)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--use_wandb', action='store_true')
    ap.add_argument('--wandb_project', type=str, default='adml-lab3')
    ap.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    args = ap.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    device = args.device
    print('Using device:', device)

    # Data
    train_ds, val_ds = get_datasets(args.data_dir)
    train_loader, val_loader = get_loaders(
        train_ds, val_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device == 'cuda')
    )

    # Model, loss, optim
    model = CustomNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # W&B (optional, like slides)
    wb = WB(
        use_wandb=args.use_wandb,
        project=args.wandb_project,
        config={
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'lr': args.lr,
            'momentum': args.momentum,
            'model': 'CustomNet',
            'dataset': 'Tiny-ImageNet-200'
        }
    )

    best_acc = 0.0
    best_path = os.path.join(args.checkpoint_dir, 'best_model.pth')

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(epoch, model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        if wb.enabled:
            wb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'best_acc': max(best_acc, val_acc)
            })

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)
            print('Saved new best model!')

    print(f'Best validation accuracy: {best_acc:.2f}%')
    wb.finish()


if __name__ == '__main__':
    main()