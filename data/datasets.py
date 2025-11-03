import os
from torchvision.datasets import ImageFolder
from .transforms import get_transforms


def get_datasets(data_dir: str):
    """
    Expects Tiny-ImageNet-200 in the usual layout with class subfolders under
    train/ and val/ (if your val is in the original single-folder layout, rearrange first).
    """
    train_dir = os.path.join(data_dir, "train")
    val_dir   = os.path.join(data_dir, "val")

    tfm = get_transforms()
    train_ds = ImageFolder(train_dir, transform=tfm)
    val_ds   = ImageFolder(val_dir,   transform=tfm)
    return train_ds, val_ds