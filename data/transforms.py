from torchvision import transforms

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_transforms():
    """Lab1/Lab2 identical transforms: ToTensor + Normalize.
    No resize/augmentation.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])