import torch
from lightning.pytorch.cli import LightningCLI
from src.senatra.tasks.classification import SeNaTraClassificationModule
from src.senatra.datamodules.imagenet import ImageNetDataModule


def main():
    torch.autograd.set_detect_anomaly(True)

    LightningCLI(
        SeNaTraClassificationModule,
        ImageNetDataModule,
        save_config_kwargs={"overwrite": True},
    )

    """
    For classification ImageNet1K:
    - epochs: 300
    - optimizer: AdamW
    cosine decay learning rate schedulear and 20 epochs of linear warmup
    batch size: 1024
    initial learning rate: 0.001
    - weight decay: 0.05

    augmentation:
    - rand augment
    - mixup (prob = 0.8)
    - cutmix (prob = 1.0)
    - random erasing
    """


if __name__ == "__main__":
    main()
