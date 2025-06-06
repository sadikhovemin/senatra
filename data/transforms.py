import torchvision.transforms as T

def imagenet_train_transforms(img_size: int):
    """
    Returns torchvision transforms for ImageNet training.
    """
    normalize = T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    return T.Compose([
        T.RandomResizedCrop(img_size),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize,
    ])

def imagenet_val_transforms(img_size: int):
    """
    Returns torchvision transforms for ImageNet validation.
    """
    normalize = T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(img_size),
        T.ToTensor(),
        normalize,
    ])