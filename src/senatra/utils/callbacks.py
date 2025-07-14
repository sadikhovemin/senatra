import math
import torch
import torch.nn.functional as F
import lightning as L
import wandb
from torchvision.utils import make_grid


class AttnMapLogger(L.Callback):
    def __init__(
        self, train_log_every_n_epochs=5, val_log_every_n_epochs=10, max_imgs=4
    ):
        super().__init__()
        self.train_epoch_freq = train_log_every_n_epochs
        self.val_epoch_freq = val_log_every_n_epochs
        self.max_imgs = max_imgs
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
        imgs = imgs[: self.max_imgs].to(pl_module.device)  # (B, 3, 224, 224)
        B, _, H_in, W_in = imgs.shape  # (B, 3, 224, 224)

        _, maps = pl_module.model(imgs, return_attn=True)
        # Assuming maps are (A_ups, A_down) tuples from each stage
        A1_ups, A2_ups, A3_ups = (
            maps[0][1],
            maps[1][1],
            maps[2][1],
        )  # A1 - (B, 4, 28, 28, 9), A2 - (B, 4, 14, 14, 9), A3 - (B, 4, 196 49)

        # --- 1. Convert sparse attention grids to dense matrices ---
        M1 = A1_ups
        M2 = A2_ups
        M3 = A3_ups

        # --- 2. Compose the matrices to get S1_token -> Final_group mappings ---
        S1_to_S2_map = M1
        S1_to_S3_map = torch.bmm(S1_to_S2_map, M2)  # (B, 3136, 196)
        S1_to_S4_map = torch.bmm(S1_to_S3_map, M3)  # (B, 3136, 49)

        # Get the final group assignment for each S1 token
        s2_groups_for_s1_tokens = S1_to_S2_map.argmax(-1)  # (B, 3136)
        s3_groups_for_s1_tokens = S1_to_S3_map.argmax(-1)  # (B, 3136)
        s4_groups_for_s1_tokens = S1_to_S4_map.argmax(-1)  # (B, 3136)

        # --- 3. Create the initial, pixel-to-S1_token mapping ---
        patch_size = 4
        px, py = torch.meshgrid(
            torch.arange(H_in, device=pl_module.device),
            torch.arange(W_in, device=pl_module.device),
            indexing="ij",
        )
        token_x = px // patch_size
        token_y = py // patch_size
        num_tokens_w = W_in // patch_size
        initial_pixel_to_s1_token_map = (
            token_x * num_tokens_w + token_y
        ).flatten()  # (50176,)

        # Expand for batch and long type for gather
        initial_map_b = initial_pixel_to_s1_token_map.long().expand(B, -1)  # (B, 50176)

        # --- 4. Apply token-level maps to the pixel-level map using gather ---
        # For each pixel, find its S1 token ID, then find the final group for that S1 token.
        seg2_map_flat = torch.gather(s2_groups_for_s1_tokens, 1, initial_map_b)
        seg3_map_flat = torch.gather(s3_groups_for_s1_tokens, 1, initial_map_b)
        seg4_map_flat = torch.gather(s4_groups_for_s1_tokens, 1, initial_map_b)

        # Reshape to 2D image segmentation maps
        seg2 = seg2_map_flat.view(B, H_in, W_in)
        seg3 = seg3_map_flat.view(B, H_in, W_in)
        seg4 = seg4_map_flat.view(B, H_in, W_in)

        viz_stack = []
        for i in range(imgs.size(0)):
            col2 = self._colourise(seg2[i])
            col3 = self._colourise(seg3[i])
            col4 = self._colourise(seg4[i])

            viz_stack.append(
                torch.stack([imgs[i].cpu(), col2.cpu(), col3.cpu(), col4.cpu()])
            )

        grid = make_grid(
            torch.cat(viz_stack, 0), nrow=4, normalize=True, scale_each=True
        )
        trainer.logger.experiment.log(
            {f"{phase}/segmentation_hierarchy": wandb.Image(grid)}
        )

    @torch.no_grad()
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.current_epoch % self.val_epoch_freq != 0:
            return
        if batch_idx > 2:
            return
        self._log_attn_maps(trainer, pl_module, batch, batch_idx, phase="val")

    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.current_epoch % self.train_epoch_freq != 0:
            return
        if batch_idx > 0:
            return
        self._log_attn_maps(trainer, pl_module, batch, batch_idx, phase="train")
