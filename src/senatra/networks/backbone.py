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


def build_M_loc(Hi: int, Wi: int, window: int = 3, dtype=torch.float32):
    """
    Builds a local attention mask of shape (N_in, N_out), where each (i, j) entry is:
    - 0 if output slot j is within the 3x3 window centered around input token i,
    - -1e4 otherwise.
    """
    Ho, Wo = Hi // 2, Wi // 2
    N_in = Hi * Wi
    N_out = Ho * Wo
    large_neg = -1e4

    # initialise everything to "forbidden"
    mask = torch.full((N_in, N_out), large_neg, dtype=dtype)  # (N_in, N_out)

    # 1. (row, col) coordinate of every input token
    in_rows, in_cols = torch.meshgrid(
        torch.arange(Hi), torch.arange(Wi), indexing="ij"
    )  # each (Hi, Wi)
    in_rows = in_rows.flatten()  # (Nin,)
    in_cols = in_cols.flatten()

    # 2. central output slot each input maps to (integer down-sampling by 2)
    out_r0 = (in_rows // 2).clamp(0, Ho - 1)
    out_c0 = (in_cols // 2).clamp(0, Wo - 1)

    # 3. for every shift inside the 3×3 window, mark the position "allowed"
    half_w = window // 2  # 1 for a 3×3 window
    in_idx = torch.arange(N_in)  # vectorised over all inputs

    for dr in range(-half_w, half_w + 1):
        for dc in range(-half_w, half_w + 1):
            out_r = (out_r0 + dr).clamp(0, Ho - 1)
            out_c = (out_c0 + dc).clamp(0, Wo - 1)
            out_idx = out_r * Wo + out_c  # flatten to 1-D index
            mask[in_idx, out_idx] = 0.0  # allow this connection

    return mask


def build_qidx_dense(Hi: int, Wi: int, window: int = 3):
    """
    Returns a dense index matrix of shape (N_in, N_out), where each (i, j) entry
    indicates the 3x3 relative position of input token i to output slot j:
    - Values in {0-8} map to 3x3 window positions (e.g., 0 = top-left, 4 = center),
    - -1 indicates (i, j) is outside the local window.
    """
    Ho, Wo = Hi // 2, Wi // 2
    N_in, N_out = Hi * Wi, Ho * Wo
    idx = torch.full((N_in, N_out), -1, dtype=torch.long)  # (N_in, N_out)

    in_r, in_c = torch.meshgrid(
        torch.arange(Hi),
        torch.arange(Wi),
        indexing="ij",
    )  # each (Hi, Wi)
    in_r, in_c = in_r.flatten(), in_c.flatten()  # each (N_in,)
    out_r0, out_c0 = in_r // 2, in_c // 2  # each (N_in,)

    off2i = {
        (dr, dc): k
        for k, (dr, dc) in enumerate(
            [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]
        )
    }
    in_idx = torch.arange(N_in)

    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            out_r = (out_r0 + dr).clamp(0, Ho - 1)
            out_c = (out_c0 + dc).clamp(0, Wo - 1)
            out_idx = out_r * Wo + out_c
            idx[in_idx, out_idx] = off2i[(dr, dc)]

    return idx  # (N_in , N_out)


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
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.freq_cache = {}

    def forward(self, x, *_):
        B, N, C = x.shape

        # 1. Project to Q, K, V
        qkv = (
            self.qkv(x)
            .view(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # Shape: [B, num_heads, N, head_dim]
        freqs = _get_rope(self.freq_cache, N, self.head_dim, x.device, B)
        q = apply_rotary_emb(q.permute(0, 2, 1, 3), freqs).permute(0, 2, 1, 3)
        k = apply_rotary_emb(k.permute(0, 2, 1, 3), freqs).permute(0, 2, 1, 3)

        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


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
class SparseGroupingLayer(nn.Module):
    def __init__(
        self,
        in_h: int,
        in_w: int,
        dim_in: int,
        dim_out: int,
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
        self.dim_in = dim_in  # Dimension of the input tokens
        self.dim_out = dim_out  # Dimension of the output tokens
        self.local = local  # Whether to use local self attention
        self.num_iters = num_iters  # Number of iterations in the sparse grouping layer
        self.window = window  # Window size for the local self attention

        # Initial convolution layer for downsampling and create the initial slots - line 1 in algorithm 1 in the paper
        self.seed_conv = nn.Conv2d(
            in_channels=dim_in,
            out_channels=dim_out,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        nn.init.kaiming_normal_(
            self.seed_conv.weight, nonlinearity="relu"
        )  # Initialize the weights of the convolution layer

        # Project X_in and X_out into query = q_proj(X_out), key = k_proj(X_in), and value = v_proj(X_in) vectors.
        self.q_proj = nn.Linear(in_features=dim_out, out_features=dim_in, bias=False)
        self.k_proj = nn.Linear(in_features=dim_in, out_features=dim_in, bias=False)
        self.v_proj = nn.Linear(in_features=dim_in, out_features=dim_out, bias=False)

        # MLP layer for updating the slots - line 10 in algorithm 1 in the paper
        hidden_dim = int(dim_out * 2)
        self.mlp = nn.Sequential(
            nn.Linear(dim_out, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim_out)
        )

        self.ln_in, self.ln_out = nn.LayerNorm(dim_in), nn.LayerNorm(dim_out)
        self.ln_attn, self.ln_mlp = nn.LayerNorm(dim_out), nn.LayerNorm(dim_out)
        self.tau = nn.Parameter(torch.tensor(math.log(dim_in**-0.5)))

        if local:
            self.register_buffer(
                "M_loc", build_M_loc(in_h, in_w, window=3), persistent=False
            )
            self.rpb = nn.Parameter(torch.zeros(9))
            nn.init.trunc_normal_(self.rpb, std=0.02)

            self.register_buffer(
                "qidx_dense", build_qidx_dense(in_h, in_w, window=3), persistent=False
            )

    # ----- dense -------------------------------------------------------
    def _dense_forward(self, x):
        B, N_in, _ = x.shape
        eps = 1e-6
        x_out = self.ln_out(
            self.seed_conv(x.transpose(1, 2).reshape(B, self.dim_in, self.Hi, self.Wi))
            .flatten(2)
            .transpose(1, 2)
        )  # (B, Ho*Wo, self.dim_out)
        k = self.k_proj(self.ln_in(x))  # Shape: (B, N_in, self.dim_in)
        v = self.v_proj(self.ln_in(x))  # Shape: (B, N_in, self.dim_out)

        for _ in range(self.num_iters):
            q = self.q_proj(self.ln_out(x_out))  # Shape: (B, N_out, self.dim_out)
            logits = torch.bmm(k, q.transpose(1, 2)) * torch.exp(
                self.tau
            )  # (B, N_in, N_out)
            a_ups = logits.softmax(dim=-1) + eps
            col_sums = a_ups.sum(dim=1, keepdim=True)  # Shape: (B, 1, N_out)
            a_down = a_ups / col_sums  # Shape: (B, N_in, N_out)

            updates = torch.bmm(a_down.transpose(1, 2), v)

            x_out = x_out + self.ln_attn(updates)
            x_out = x_out + self.ln_mlp(self.mlp(x_out))

        return x_out, a_ups, a_down

    # # ----- sparse ------------------------------------------------------
    def _sparse_forward(self, x):
        B, N_in, _ = x.shape

        # 1. Seed slots ------------------------------------------------------
        x_out = self.ln_out(
            self.seed_conv(x.transpose(1, 2).reshape(B, self.dim_in, self.Hi, self.Wi))
            .flatten(2)
            .transpose(1, 2)
        )

        k = self.k_proj(self.ln_in(x))  # (B, Nin, d)
        v = self.v_proj(self.ln_in(x))  # (B, Nin, Dout)

        for _ in range(self.num_iters):
            q = self.q_proj(self.ln_out(x_out))  # (B, Nout, d)

            # 3. logits with learnable τ and rel-pos bias
            logits = (k @ q.transpose(1, 2)) * torch.exp(self.tau)  # τ·kqᵀ
            bias = self.rpb[self.qidx_dense]  # (Nin, Nout)
            logits = logits + bias.unsqueeze(0)  # (B, Nin, Nout)
            logits = logits + self.M_loc  # + M_loc (0 / −1e4)

            # 5. row soft-max
            a_ups = logits.softmax(dim=-1)

            # 6. column renormalisation
            col_sums = a_ups.sum(dim=1, keepdim=True)  # (B,1,Nout)
            a_down = a_ups / (col_sums + 1e-8)  # columns sum to 1

            # 9. slot update
            updates = torch.bmm(a_down.transpose(1, 2), v)  # (B,Nout,Dout)
            x_out = x_out + self.ln_attn(updates)
            x_out = x_out + self.ln_mlp(self.mlp(x_out))

        return x_out, a_ups, a_down

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
                dim_in=dim_in,
                dim_out=dim_out,
                window=3 if not dense_group else None,
                local=not dense_group,
                num_iters=3,
            )
            # self.proj = nn.Linear(
            #     dim_in, dim_out
            # )  # Should we do the projection here or inside the sparse grouping layer?
            # self.norm = nn.LayerNorm(dim_out)
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
        A_up, A_down = None, None

        if self.use_group:
            x, A_up, A_down = self.group(x)

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

    def __init__(self, img_size=224, variant="T", in_channels=3, embed_dim=64):
        super().__init__()

        cfg = self.cfgs[variant]
        self.patch = PatchEmbed(
            img_size=img_size,
            patch_size=4,
            in_channels=in_channels,
            embed_dim=cfg["dims"][0],
        )

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
        x = self.patch(x)  # (B, 3136, 64 (T), 128 (B), 192 (L))

        maps = []

        # Stage-0  (no grouping)
        x, *_ = self.s0(x)

        # Stage-1 … 3  (with grouping)
        for name, stage in (("s1", self.s1), ("s2", self.s2), ("s3", self.s3)):
            x, A_up, A_down = stage(x)
            maps.append((name, A_up, A_down))

        return (x, maps) if return_attn else x
