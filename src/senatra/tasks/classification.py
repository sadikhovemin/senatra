import lightning.pytorch as pl
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from src.senatra.networks.backbone import SeNaTra

SENATRA_CONFIGS = {
    "T": {"layers": [3, 4, 18, 5], "embed_dim": 512, "mlp_ratio": 3},
    "B": {"layers": [3, 4, 18, 5], "embed_dim": 1024, "mlp_ratio": 2},
    "L": {"layers": [3, 4, 18, 5], "embed_dim": 1536, "mlp_ratio": 2},
}


class SeNaTraClassificationModule(pl.LightningModule):
    def __init__(
        self,
        img_size=224,
        num_classes=1000,
        lr=1e-3,
        weight_decay=1e-4,
        variant="T",
        warmup_epochs=5,
        min_lr=1e-6,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = SeNaTra(variant="T")
        self.token_dim = SENATRA_CONFIGS[variant]["embed_dim"]
        self.cls_head = nn.Linear(self.token_dim, num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        pred = self.model(x)
        pooled = pred.mean(dim=1)  # Global average pooling
        logits = self.cls_head(pooled)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
        )
        self.log(
            "train_acc", acc, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True
        )

        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("learning_rate", lr, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True
        )
        self.log(
            "val_acc", acc, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True
        )
        return {"val_loss": loss, "val_acc": acc}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            betas=(0.9, 0.999),
            weight_decay=self.hparams.weight_decay,
        )

        max_epochs = self.trainer.max_epochs
        warmup_epochs = self.hparams.warmup_epochs

        active_schedulers = []
        milestones_for_seq = []

        if warmup_epochs > 0:
            start_lr_val = self.hparams.min_lr
            start_factor_val = start_lr_val / self.hparams.lr

            linear_scheduler = LinearLR(
                optimizer,
                start_factor=start_factor_val,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            active_schedulers.append(linear_scheduler)
            milestones_for_seq.append(warmup_epochs)

        cosine_t_max_epochs = max_epochs - warmup_epochs

        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cosine_t_max_epochs,
            eta_min=self.hparams.min_lr,
        )
        active_schedulers.append(cosine_scheduler)

        final_scheduler = SequentialLR(
            optimizer, schedulers=active_schedulers, milestones=milestones_for_seq
        )
        print(
            f"Info: Using SequentialLR (epoch interval) with milestones at epoch {milestones_for_seq}"
        )

        print("self.hparams:", self.hparams)
        print("self.trainer.max_epochs:", self.trainer.max_epochs)
        print("self.trainer.max_steps:", self.trainer.max_steps)
        print("len(active_schedulers):", len(active_schedulers))
        print("warmup_epochs_for_scheduler:", warmup_epochs)
        print("effective_max_epochs_for_scheduler:", max_epochs)
        print("cosine_t_max_epochs:", cosine_t_max_epochs)
        print("active_schedulers:", active_schedulers)
        print("milestones_for_seq:", milestones_for_seq)
        print("Final Scheduler:", final_scheduler)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": final_scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
