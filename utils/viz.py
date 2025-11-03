import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

from data.datasets import get_datasets
from torch.utils.data import DataLoader


def _denormalize(image: torch.Tensor):
    image = image.cpu().numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    image = image * std + mean
    image = np.clip(image, 0, 1)
    return image


def show_ten_classes(data_dir: str, batch_size: int = 28, num_workers: int = 0):
    train_ds, _ = get_datasets(data_dir)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, pin_memory=False)

    picked = {}
    for inputs, labels in loader:
        for x, y in zip(inputs, labels):
            y = int(y.item())
            if y not in picked:
                picked[y] = x
                if len(picked) == 10:
                    break
        if len(picked) == 10:
            break

    idx_to_class = {i: c for i, c in enumerate(train_ds.classes)}

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    for ax, (cls, img) in zip(axes, picked.items()):
        ax.imshow(_denormalize(img))
        ax.set_title(idx_to_class[cls])
        ax.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, default='dataset/tiny-imagenet-200')
    p.add_argument('--batch_size', type=int, default=28)
    p.add_argument('--num_workers', type=int, default=0)  # Windows-safe default
    args = p.parse_args()

    show_ten_classes(args.data_dir, args.batch_size, args.num_workers)