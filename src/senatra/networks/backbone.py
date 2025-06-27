import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import natten.functional as natten_F
from attn_gym.paged_attention.model import apply_rotary_emb, precompute_freqs_cis


def _get_rope(freq_cache, seq_len, dim, device, batch_size):
    key = (seq_len, dim, device, batch_size)
    if key not in freq_cache:
        freqs = precompute_freqs_cis(seq_len=seq_len, n_elem=dim)
        freqs = freqs.to(device)  # [S, D//2, 2]
        freqs = freqs.unsqueeze(0).expand(batch_size, -1, -1, -1)  # -> [B, S, D//2, 2]
        freq_cache[key] = freqs
    return freq_cache[key]


# -----------------------------------------------------------------------------
# Stochastic Drop
# -----------------------------------------------------------------------------
class DropPath(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        keep = torch.rand(x.shape[0], 1, 1, device=x.device) >= self.p
        return x * keep / (1.0 - self.p)


# -----------------------------------------------------------------------------
# Self Attention Blocks
# -----------------------------------------------------------------------------
class LocalSelfAttention(nn.Module):
    """Neighborhood Attention (window=7) + RoPE positional encoding"""

    def __init__(self, dim, head_dim=32, window=7):
        super().__init__()
        assert dim % head_dim == 0
        self.head_dim = head_dim
        self.heads = dim // head_dim
        self.scale = self.head_dim**-0.5
        self.window = window
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.freq_cache = {}

    def forward(self, x, H, W):
        B, N, C = x.shape
        qkv = (
            self.qkv(x).view(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        freqs = _get_rope(self.freq_cache, N, self.head_dim, x.device, B)
        q = apply_rotary_emb(q.permute(0, 2, 1, 3), freqs).permute(0, 2, 1, 3)
        k = apply_rotary_emb(k.permute(0, 2, 1, 3), freqs).permute(0, 2, 1, 3)

        q = (
            q.transpose(2, 3)
            .reshape(B, self.heads, self.head_dim, H, W)
            .permute(0, 1, 3, 4, 2)
        )
        k = (
            k.transpose(2, 3)
            .reshape(B, self.heads, self.head_dim, H, W)
            .permute(0, 1, 3, 4, 2)
        )
        v = (
            v.transpose(2, 3)
            .reshape(B, self.heads, self.head_dim, H, W)
            .permute(0, 1, 3, 4, 2)
        )

        attn = natten_F.na2d_qk(q, k, self.window) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = natten_F.na2d_av(attn, v, self.window)
        out = out.permute(0, 1, 4, 2, 3).reshape(B, C, N).transpose(1, 2)
        return self.proj(out)


class GlobalSelfAttention(nn.Module):
    """Multi Head Attention with RoPE (used in final stage)"""

    def __init__(self, dim, head_dim=32):
        super().__init__()
        assert dim % head_dim == 0, "dim must be divisible by head_dim"
        num_heads = dim // head_dim
        self.mha = nn.MultiheadAttention(dim, num_heads, batch_first=True, bias=False)
        self.freq_cache = {}

    def forward(self, x, *_):
        B, N, C = x.shape
        freqs = _get_rope(self.freq_cache, N, C // self.mha.num_heads, x.device, B)
        qk = apply_rotary_emb(
            x.view(B, N, self.mha.num_heads, C // self.mha.num_heads), freqs
        ).view(B, N, C)
        out, _ = self.mha(qk, qk, x, need_weights=False)
        return out


class SeNaTraBlock(nn.Module):
    def __init__(
        self,
        H: int,
        W: int,
        dim: int,
        mlp_ratio: float = 3.0,
        local: bool = True,
        drop_path=0.3,
    ):
        super().__init__()
        self.H, self.W = H, W
        self.ln1 = nn.LayerNorm(dim)
        if local:
            self.attn = LocalSelfAttention(dim, head_dim=32, window=7)
        else:
            self.attn = GlobalSelfAttention(dim=dim, head_dim=32)
        self.ln2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        # Layer norm, attention, and skip connection
        x_norm = self.ln1(x)
        x_attn = self.attn(x_norm, self.H, self.W)
        x_drop = self.drop_path(x_attn)
        x = x + x_drop

        # Layer norm, MLP, and skip connection
        x_norm = self.ln2(x)
        x_mlp = self.mlp(x_norm)
        x_drop = self.drop_path(x_mlp)
        x = x + x_drop

        return x


# -----------------------------------------------------------------------------
# Grouping Layer
# -----------------------------------------------------------------------------
class _WindowRelPosBias(nn.Module):
    def __init__(self, heads: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(heads, 9))
        nn.init.trunc_normal_(self.bias, std=0.02)

    def forward(self):
        return self.bias.view(1, -1, 1, 1, 9)


def compute_q_idx(Hi, Wi, kernel_size=3, B=1, G=1):

    with torch.no_grad():
        dummy_q = torch.arange(Hi * Wi, dtype=torch.float32).reshape(1, 1, Hi, Wi, 1)
        dummy_k = torch.ones_like(dummy_q)

        q_idx = natten_F.na2d_qk(query=dummy_k, key=dummy_q, kernel_size=kernel_size)

    return q_idx


class SparseGroupingLayer(nn.Module):
    def __init__(
        self,
        in_h: int,
        in_w: int,
        dim: int,
        window=3,
        local: bool = True,
        num_iters: int = 3,
    ):
        super().__init__()
        self.Hi, self.Wi = (
            in_h,
            in_w,
        )  # Dimensions of the input tokens - X_in in algorithm 1 in the paper
        self.Ho, self.Wo = (
            in_h // 2,
            in_w // 2,
        )  # Dimensions of the output tokens - X_out in algorithm 1 in the paper
        self.N_in, self.N_out = in_h * in_w, (in_h // 2) * (
            in_w // 2
        )  # Number of input and output tokens
        self.dim = dim  # Dimension of the input tokens
        self.local = local  # Whether to use local self attention
        self.num_iters = num_iters  # Number of iterations in the sparse grouping layer
        self.window = window  # Window size for the local self attention

        # Initial convolution layer for downsampling and create the initial slots - line 1 in algorithm 1 in the paper
        self.seed_conv = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        nn.init.kaiming_normal_(
            self.seed_conv.weight, nonlinearity="relu"
        )  # Initialize the weights of the convolution layer

        # Project X_in and X_out into query = q_proj(X_out), key = k_proj(X_in), and value = v_proj(X_in) vectors.
        self.q_proj = nn.Linear(in_features=dim, out_features=dim, bias=False)
        self.k_proj = nn.Linear(in_features=dim, out_features=dim, bias=False)
        self.v_proj = nn.Linear(in_features=dim, out_features=dim, bias=False)

        # MLP layer for updating the slots - line 10 in algorithm 1 in the paper
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim)
        )

        self.ln_in, self.ln_out = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.tau = nn.Parameter(torch.tensor(math.log(dim**-0.5)))

        if local:
            self.rpb = _WindowRelPosBias(4)
            q_idx = compute_q_idx(self.Ho, self.Wo, kernel_size=self.window)
            self.register_buffer("q_idx", q_idx, persistent=False)

    # ----- dense -------------------------------------------------------
    def _dense_forward(self, x):
        B, N_in, D = x.shape
        eps = 1e-6
        x_out = self.ln_out(
            self.seed_conv(x.transpose(1, 2).reshape(B, D, self.Hi, self.Wi))
            .flatten(2)
            .transpose(1, 2)
        )
        k = self.k_proj(self.ln_in(x))  # Shape: (B, N_in, D)
        v = self.v_proj(x)  # Shape: (B, N_in, D)

        for _ in range(self.num_iters):
            q = self.q_proj(self.ln_out(x_out))  # Shape: (B, N_out, D)
            logits = torch.bmm(k, q.transpose(1, 2)) * torch.exp(self.tau)
            a_ups = logits.softmax(dim=-1) + eps
            col_sums = a_ups.sum(dim=1, keepdim=True)  # Shape: (B, 1, N_out)
            a_down = a_ups / col_sums  # Shape: (B, N_in, N_out)

            updates = torch.bmm(a_down.transpose(1, 2), v)
            x_out = x_out + self.ln_out(updates)

            x_out = x_out + self.ln_out(self.mlp(x_out))

        return x_out, a_ups, a_down

    # ----- sparse ------------------------------------------------------
    def _sparse_forward(self, x):
        B, _, D = x.shape
        eps = 1e-6

        A_row_list, A_col_list = [], []

        # ---- K, V patches ------------------------------------------------
        # Project and reshape k
        k_proj = self.k_proj(self.ln_in(x))  # (B, N, self.dim)
        k_trans = k_proj.transpose(1, 2)  # (B, self.dim, N)
        k_map = k_trans.reshape(
            B, D, self.Hi, self.Wi
        )  # (B, self.dim, self.Hi, self.Wi)
        k_unfold = F.unfold(
            input=k_map, kernel_size=2, stride=2
        )  # (B, self.dim*4, self.Ho*self.Wo) where Ho = Hi/2 and Wo = Wi/2
        k_view = k_unfold.view(
            B, D, 4, self.Ho, self.Wo
        )  # (B, self.dim, 4, self.Ho, self.Wo)
        k = k_view.permute(
            0, 2, 3, 4, 1
        ).contiguous()  # (B, 4, self.Ho, self.Wo, self.dim)

        # Project and reshape v
        v_proj = self.v_proj(x)  # (B, N, self.dim)
        v_trans = v_proj.transpose(1, 2)  # (B, self.dim, N)
        v_map = v_trans.reshape(
            B, D, self.Hi, self.Wi
        )  # (B, self.dim, self.Hi, self.Wi)
        v_unfold = F.unfold(
            input=v_map, kernel_size=2, stride=2
        )  # (B, self.dim*4, self.Ho*self.Wo) where Ho = Hi/2 and Wo = Wi/2
        v_view = v_unfold.view(
            B, D, 4, self.Ho, self.Wo
        )  # (B, self.dim, 4, self.Ho, self.Wo)
        v = v_view.permute(
            0, 2, 3, 4, 1
        ).contiguous()  # (B, 4, self.Ho, self.Wo, self.dim)

        # ---- seed slots ---------------------------------------------------
        x_reshaped = x.transpose(1, 2).reshape(
            B, D, self.Hi, self.Wi
        )  # (B, D, self.Hi, self.Wi)
        x_conv = self.seed_conv(x_reshaped)  # (B, D, self.Ho, self.Wo)
        x_flat = x_conv.flatten(2)  # (B, D, self.Ho*self.Wo)
        x_trans = x_flat.transpose(1, 2)  # (B, self.Ho*self.Wo, D)
        x_out = self.ln_out(x_trans)  # (B, self.Ho*self.Wo, D)

        for _ in range(self.num_iters):
            q = self.q_proj(self.ln_out(x_out))  # (B, self.Ho*self.Wo, D)
            q = q.view(B, self.Ho, self.Wo, D)  # (B, self.Ho, self.Wo, D)
            q = q.unsqueeze(1)  # (B, 1, self.Ho, self.Wo, D)
            q = q.repeat(1, 4, 1, 1, 1)  # (B, 4, self.Ho, self.Wo, D)
            q = q.contiguous()  # (B, 4, self.Ho, self.Wo, D)

            # ---- attention -----------------------------------------------
            attn = natten_F.na2d_qk(
                query=k, key=q, kernel_size=3
            )  # (B, 4, self.Ho, self.Wo, 9) - the last dimension is the size of the local region that query attends to in the key
            attn = attn + self.rpb()  # (B, 4, self.Ho, self.Wo, 9)
            attn = (
                F.softmax(attn * torch.exp(self.tau), dim=-1) + eps
            )  # (B, 4, self.Ho, self.Wo, 9) - Each token in X_in (self.Hi*self.Wi) attends to 9 tokens in X_out (self.Ho*self.Wo) where Ho = Hi/2 and Wo = Wi/2

            # Flatten attn over spatial dimensions (Ho*Wo)
            attn_flat = attn.flatten(2, 3)  # (B, 4, Ho*Wo, 9)
            q_idx_flat = (
                self.q_idx.expand(B, 4, -1, -1, -1).flatten(2, 3).long()
            )  # (B, 1, Ho*Wo, 9)

            num_input_tokens = 4 * self.Ho * self.Wo

            # We create a tensor to hold the sum of each column (input token)
            col_sums_sparse = torch.zeros(
                B, num_input_tokens, dtype=torch.float32, device=q.device
            )

            # Flatten the attention scores and indices for efficient processing
            attn_flat_1d = attn_flat.flatten(1, -1)  # (B, 4*784*9) = (256, 28224)
            q_idx_flat_1d = q_idx_flat.flatten(1, -1)  # (B, 4*784*9) = (256, 28224)

            # Create batch offsets to make indices unique across batches
            batch_offsets = (
                torch.arange(B, device=q.device).unsqueeze(1) * num_input_tokens
            )
            q_idx_with_batch = q_idx_flat_1d + batch_offsets

            # Flatten everything for a single scatter_add_ operation
            # This sums all attention scores that belong to the same input token (column)
            col_sums_flat = col_sums_sparse.flatten()  # (B * 3136) = (1024)
            q_idx_flat_all = q_idx_with_batch.flatten()  # (B * 576) = (9216)
            attn_flat_all = attn_flat_1d.flatten()  # (B * 576) = (9216)

            # Single scatter_add_ operation for all batches
            # This sums all attention scores that belong to the same input token (column)
            col_sums_flat.scatter_add_(
                dim=0, index=q_idx_flat_all.to(torch.int64), src=attn_flat_all
            )

            # Reshape back to (B, num_input_tokens)
            col_sums_sparse = col_sums_flat.view(B, num_input_tokens)

            # 2. Gather the corresponding column sum for each attention score
            # We need to reshape col_sums_sparse to match the dimensions of q_idx_flat
            # col_sums_sparse: (B, num_input_tokens) -> (B, 1, num_input_tokens) -> (B, 4, Ho*Wo, num_input_tokens)
            col_sums_expanded = (
                col_sums_sparse.unsqueeze(1)
                .unsqueeze(2)
                .expand(B, 4, self.Ho * self.Wo, -1)
            )

            # Gather the appropriate column sum for each attention score
            # q_idx_flat: (B, 1, Ho*Wo, 9) contains indices into the num_input_tokens
            col_sums_for_each_attn = torch.gather(
                col_sums_expanded, dim=3, index=q_idx_flat.to(torch.int64)
            )

            # 3. Perform column normalization
            A_col = attn_flat / (col_sums_for_each_attn + 1e-8)  # (B, 4, 784, 9)
            A_col = A_col.view(B, 4, self.Ho, self.Wo, 9)

            updates = natten_F.na2d_av(A_col, v, 3)
            updates = updates.sum(dim=1).permute(0, 3, 1, 2).flatten(2).transpose(1, 2)
            x_out = x_out + updates
            x_out = x_out + self.ln_out(self.mlp(x_out))

            A_row_list.append(attn)
            A_col_list.append(A_col)

        return x_out, A_row_list[-1], A_col_list[-1]

    def forward(self, x: torch.Tensor):
        # x: (B, N, D)
        return self._sparse_forward(x) if self.local else self._dense_forward(x)


class SeNaTraStage(nn.Module):
    """
    SeNaTraStage is a stage of the SeNaTra backbone shown in Figure 1a of the paper.
    Its main building blocks are:
    - Sparse Grouping Layer:
        - use_group is False on the first stage, and True on the other stages.
        - Grouping layer is applied locally in the second and third stages.
        - Dense grouping is applied in the fourth stage.
    - SeNaTra Blocks:
        - This block consists of a self-attention layer, layer normalization, and an MLP layer.
        - The self-attention layer is applied locally in the first three stages.
        - Global self-attention is applied in the fourth stage.
        - The output of the stage is the output of the last SeNaTra block.
    """

    def __init__(
        self,
        in_h: int,
        in_w: int,
        dim_in: int,
        dim_out: int,
        depth: int,
        mlp_ratio: float,
        use_group: bool,
        dense_group: bool,
        local_blocks: bool,
    ):
        super().__init__()
        self.use_group = use_group
        if use_group:
            self.group = SparseGroupingLayer(
                in_h=in_h,
                in_w=in_w,
                dim=dim_in,
                window=3 if not dense_group else None,
                local=not dense_group,
                num_iters=3,
            )
            self.proj = nn.Linear(
                dim_in, dim_out
            )  # Should we do the projection here or inside the sparse grouping layer?
            self.norm = nn.LayerNorm(dim_out)
            post_h, post_w = in_h // 2, in_w // 2
            post_dim = dim_out
        else:
            self.group = None
            post_h, post_w, post_dim = in_h, in_w, dim_in

        self.blocks = nn.ModuleList(
            SeNaTraBlock(post_h, post_w, post_dim, mlp_ratio, local=local_blocks)
            for _ in range(depth)
        )

    def forward(self, x):
        # x: (B, 3136, 64 (T), 128 (B), 192 (L))
        if self.use_group:
            x, A_up, A_down = self.group(x)
            x = self.proj(x)
            x = self.norm(x)
        else:
            A_up = None
            A_down = None

        for blk in self.blocks:
            x = blk(x)

        return x, A_up, A_down


class PatchEmbed(nn.Module):
    """
    Split each image into non-overlapping patch_size x patch_size patches and linealy projects them to dimension embed_dim
    Input: x of shape (B, 3, H, W)
    Output: patches of shape (B, N, C) where N = (H/patch_size) * (W/patch_size)
    """

    def __init__(self, img_size=224, patch_size=4, in_channels=3, embed_dim=96):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.H = self.W = img_size // patch_size

    def forward(self, x):
        # x: (B, 3, 224, 224)

        x = self.proj(x)  # (B, 96, 56, 56)
        x = x.flatten(2)  # (B, 96, 3136)
        x = x.transpose(1, 2)  # (B, 3136, 96)

        x = self.norm(x)  # (B, 3136, 96)

        return x


class SeNaTra(nn.Module):
    """Backbone variants T / B / L"""

    cfgs = {
        "T": dict(depths=[3, 4, 18, 5], dims=[64, 128, 256, 512], mlp=3.0),
        "B": dict(depths=[3, 4, 18, 5], dims=[128, 256, 512, 1024], mlp=2.0),
        "L": dict(depths=[3, 4, 18, 5], dims=[192, 384, 768, 1536], mlp=2.0),
    }

    def __init__(self, img_size=224, variant="T", in_channels=3, embed_dim=96):
        super().__init__()

        cfg = self.cfgs[variant]
        self.patch = PatchEmbed(
            img_size=img_size,
            patch_size=4,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        self.proj0 = nn.Linear(in_features=embed_dim, out_features=cfg["dims"][0])

        self.norm0 = nn.LayerNorm(cfg["dims"][0])

        H = W = (
            img_size // 4
        )  # image size after patch embedding layer: img_size / patch_size

        # Stage 0
        # No grouping, no downsample,
        # Only applies local self attension in SeNaTra blocks
        self.s0 = SeNaTraStage(
            in_h=H,
            in_w=W,
            dim_in=cfg["dims"][0],
            dim_out=cfg["dims"][0],
            depth=cfg["depths"][0],
            mlp_ratio=cfg["mlp"],
            use_group=False,
            dense_group=False,
            local_blocks=True,
        )

        # Stage 1
        # Local grouping, local self attention in SeNaTra blocks
        self.s1 = SeNaTraStage(
            in_h=H,
            in_w=W,
            dim_in=cfg["dims"][0],
            dim_out=cfg["dims"][1],
            depth=cfg["depths"][1],
            mlp_ratio=cfg["mlp"],
            use_group=True,
            dense_group=False,
            local_blocks=True,
        )
        # Dimensions are halved after the grouping layer
        H //= 2
        W //= 2

        # Stage 2
        # Local grouping, local self attention in SeNaTra blocks
        self.s2 = SeNaTraStage(
            in_h=H,
            in_w=W,
            dim_in=cfg["dims"][1],
            dim_out=cfg["dims"][2],
            depth=cfg["depths"][2],
            mlp_ratio=cfg["mlp"],
            use_group=True,
            dense_group=False,
            local_blocks=True,
        )
        # Dimensions are halved after the grouping layer
        H //= 2
        W //= 2

        # Stage 3
        # Dense grouping, global self attention in SeNaTra blocks
        self.s3 = SeNaTraStage(
            in_h=H,
            in_w=W,
            dim_in=cfg["dims"][2],
            dim_out=cfg["dims"][3],
            depth=cfg["depths"][3],
            mlp_ratio=cfg["mlp"],
            use_group=True,
            dense_group=True,
            local_blocks=False,
        )

    def forward(self, x, return_attn=False):
        # Patch Embedding
        # x: (B, 3, 224, 224)
        x = self.patch(x)  # (B, 3136, 96)
        x = self.proj0(x)  # (B, 3136, 64 (T), 128 (B), 192 (L))
        x = self.norm0(x)  # (B, 3136, 64 (T), 128 (B), 192 (L))

        maps = []

        # Stage-0  (no grouping)
        x, *_ = self.s0(x)

        # Stage-1 … 3  (with grouping)
        for name, stage in (("s1", self.s1), ("s2", self.s2), ("s3", self.s3)):
            x, A_up, A_down = stage(x)
            maps.append((name, A_up, A_down))

        return (x, maps) if return_attn else x
