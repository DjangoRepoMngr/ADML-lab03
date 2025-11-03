import argparse
import torch
from torch import nn

from data.datasets import get_datasets
from data.loaders import get_loaders
from models.customnet import CustomNet
from train import validate  # reuse the same function


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', type=str, default='dataset/tiny-imagenet-200')
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--num_workers', type=int, default=2)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth')
    args = ap.parse_args()

    device = args.device
    print('Using device:', device)

    # Data
    _, val_ds = get_datasets(args.data_dir)
    _, val_loader = get_loaders(None, val_ds, batch_size=args.batch_size, num_workers=args.num_workers)

    # Model
    model = CustomNet().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    criterion = nn.CrossEntropyLoss()
    _, val_acc = validate(model, val_loader, criterion, device)
    print(f'Validation accuracy (checkpoint): {val_acc:.2f}%')


if __name__ == '__main__':
    main()