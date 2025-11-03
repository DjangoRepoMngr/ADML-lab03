from torch.utils.data import DataLoader
import torch


def get_loaders(train_ds, val_ds, batch_size=64, num_workers=2, pin_memory=None):
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    return train_loader, val_loader