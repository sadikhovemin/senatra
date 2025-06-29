import math
import torch
import torch.nn.functional as F
import lightning as L
import wandb
from torchvision.utils import make_grid


def up_grid_to_matrix(A_up_grid, q_idx):
    # Convert sparse [B, 4, H, W, 9] attention to dense [B, N_in, N_out]
    B, _, H, W, _ = A_up_grid.shape
    A_flat = A_up_grid.reshape(B, 4 * H * W, 9)  # [B, N_in, 9]
    idx = q_idx.expand(B, 4, -1, -1, -1)  # [B,4,H,W,9]
    idx = idx.reshape(B, 4 * H * W, 9)  # [B, N_in, 9]

    N_out = H * W
    dense = torch.zeros(
        B, 4 * H * W, N_out, device=A_up_grid.device, dtype=A_up_grid.dtype
    )
    dense.scatter_add_(2, idx.long(), A_flat)  # accumulate 9 neighbours
    return dense  # [B, N_in, N_out]


class AttnMapLogger(L.Callback):
    def __init__(
        self, train_log_every_n_epochs=5, val_log_every_n_epochs=10, max_imgs=4
    ):
        super().__init__()
        self.train_epoch_freq = train_log_every_n_epochs
        self.val_epoch_freq = val_log_every_n_epochs
        self.max_imgs = max_imgs

        # self.freq, self.max_imgs = log_every_n_batches, max_imgs
        self.palette = torch.rand(1, 1024, 3)  # fixed colour set

    def _colourise(self, idx_map):
        pal = self.palette.to(idx_map.device)  # [1,1024,3]
        flat = idx_map.flatten().long()  # (H·W)
        color = pal.squeeze(0)[flat]  # (H·W,3)
        return color.view(*idx_map.shape, 3).permute(2, 0, 1)  # [3,H,W]

    def _log_attn_maps(self, trainer, pl_module, batch, batch_idx, phase="val"):
        if trainer.global_rank != 0:
            return

        imgs, _ = batch
        imgs = imgs[: self.max_imgs].to(pl_module.device)  # (B,3,224,224)

        _, maps = pl_module.model(imgs, return_attn=True)
        A1, A2, A3 = maps[0][1], maps[1][1], maps[2][1]  # s1-s3

        # convert sparse -> dense
        q1 = pl_module.model.s1.group.q_idx
        q2 = pl_module.model.s2.group.q_idx
        M1 = up_grid_to_matrix(A1, q1)  # 3136 -> 784
        M2 = up_grid_to_matrix(A2, q2)  #  784 -> 196
        M3 = A3  #  196→49

        A2_map = M1
        A3_map = torch.bmm(M1, M2)
        A4_map = torch.bmm(A3_map, M3)

        seg2 = A2_map.argmax(-1).view(-1, 56, 56)
        seg3 = A3_map.argmax(-1).view(-1, 56, 56)
        seg4 = A4_map.argmax(-1).view(-1, 56, 56)

        H_out = imgs.shape[2]  # 224

        viz_stack = []
        for i in range(imgs.size(0)):
            col2 = self._colourise(seg2[i])
            col3 = self._colourise(seg3[i])
            col4 = self._colourise(seg4[i])

            # upsample colour maps to 224×224
            col2 = F.interpolate(col2.unsqueeze(0), size=H_out, mode="nearest").squeeze(
                0
            )
            col3 = F.interpolate(col3.unsqueeze(0), size=H_out, mode="nearest").squeeze(
                0
            )
            col4 = F.interpolate(col4.unsqueeze(0), size=H_out, mode="nearest").squeeze(
                0
            )

            viz_stack.append(
                torch.stack([imgs[i].cpu(), col2.cpu(), col3.cpu(), col4.cpu()])
            )

        grid = make_grid(
            torch.cat(viz_stack, 0), nrow=4, normalize=True, scale_each=True
        )
        trainer.logger.experiment.log(
            {
                f"{phase}/segmentation_hierarchy/epoch_{trainer.current_epoch}_batch_{batch_idx}": wandb.Image(
                    grid
                )
            }
        )

    @torch.no_grad()
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.current_epoch % self.val_epoch_freq != 0:
            return
        self._log_attn_maps(trainer, pl_module, batch, batch_idx, phase="val")

    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.current_epoch % self.train_epoch_freq != 0:
            return
        if batch_idx > 0:
            return
        self._log_attn_maps(trainer, pl_module, batch, batch_idx, phase="train")
