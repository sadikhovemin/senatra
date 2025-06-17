import math
import natten.functional as natten_F
import torch
import torch.nn as nn
import torch.nn.functional as F
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
# 1. Self‑Attention blocks
# -----------------------------------------------------------------------------
class LocalSelfAttention(nn.Module):
    """Neighborhood Attention (window=7) + RoPE positional encoding"""

    def __init__(self, dim, num_heads=8, window=7):
        super().__init__()
        assert dim % num_heads == 0
        self.heads = num_heads
        self.head_dim = dim // num_heads
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

    def __init__(self, dim, num_heads=8):
        super().__init__()
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


# -----------------------------------------------------------------------------
# 2. Grouping Layer
# -----------------------------------------------------------------------------
class _WindowRelPosBias(nn.Module):
    def __init__(self, heads: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(heads, 9))
        nn.init.trunc_normal_(self.bias, std=0.02)

    def forward(self):
        return self.bias.view(1, -1, 1, 1, 9)


def _build_q_and_in_idx(Ho: int, Wo: int):
    """Return (q_idx→36, in_idx→N_in, col_idx→N_out)."""
    rels = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
    q_list, in_list, col_list = [], [], []
    for h in range(4):
        g_h, i_h, c_h = [], [], []
        for oy in range(Ho):
            for ox in range(Wo):
                base = (oy * Wo + ox) * 9
                g_tmp, i_tmp, c_tmp = [], [], []
                for dy, dx in rels:
                    ny, nx = oy + dy, ox + dx
                    if 0 <= ny < Ho and 0 <= nx < Wo:
                        coarse = ny * Wo + nx
                        g_tmp.append(base + (dy + 1) * 3 + (dx + 1))
                        i_tmp.append(coarse * 4 + h)
                        c_tmp.append(coarse)
                    else:
                        g_tmp.append(0)
                        i_tmp.append(0)
                        c_tmp.append(0)
                g_h.append(torch.tensor(g_tmp * 4))  # → 36
                i_h.append(torch.tensor(i_tmp * 4))
                c_h.append(torch.tensor(c_tmp * 4))
        q_list.append(torch.stack(g_h))
        in_list.append(torch.stack(i_h))
        col_list.append(torch.stack(c_h))
    return (
        torch.stack(q_list).long(),  # (4,N_out,36)
        torch.stack(in_list).long(),  # (4,N_out,36) indices into N_in
        torch.stack(col_list).long(),  # (4,N_out,36) indices into N_out
    )


class SparseGroupingLayer(nn.Module):
    def __init__(
        self,
        h_in: int,
        w_in: int,
        dim: int,
        window=3,
        loc: bool = True,
        num_iters: int = 3,
    ):
        super().__init__()
        self.Hi, self.Wi = h_in, w_in
        self.Ho, self.Wo = h_in // 2, w_in // 2
        self.N_in, self.N_out = h_in * w_in, (h_in // 2) * (w_in // 2)
        self.dim, self.loc, self.num_iters = dim, loc, num_iters
        self.window = window

        self.seed_conv = nn.Conv2d(dim, dim, 3, 2, 1, bias=False)
        nn.init.kaiming_normal_(self.seed_conv.weight, nonlinearity="relu")
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim)
        )
        self.ln_in, self.ln_out = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.tau = nn.Parameter(torch.tensor(math.log(dim**-0.5)))

        if loc:
            self.rpb = _WindowRelPosBias(4)
            q_idx, in_idx, col_idx = _build_q_and_in_idx(self.Ho, self.Wo)
            self.register_buffer("q_idx", q_idx, persistent=False)
            self.register_buffer("in_idx", in_idx, persistent=False)
            self.register_buffer("col_idx", col_idx, persistent=False)

    # ----- dense -------------------------------------------------------
    def _dense_forward(self, x):
        B, _, D = x.shape
        eps = 1e-6
        x_out = self.ln_out(
            self.seed_conv(x.transpose(1, 2).reshape(B, D, self.Hi, self.Wi))
            .flatten(2)
            .transpose(1, 2)
        )
        k = self.k_proj(self.ln_in(x))
        v = self.v_proj(x)
        for _ in range(self.num_iters):
            q = self.q_proj(self.ln_out(x_out))
            logits = torch.bmm(q, k.transpose(1, 2)) * torch.exp(self.tau)
            a_row = logits.softmax(dim=-1) + eps
            denom = a_row.sum(dim=1, keepdim=True)
            a_col = a_row / denom
            x_out = x_out + torch.bmm(a_col, v)
            x_out = x_out + self.ln_out(self.mlp(x_out))
        return x_out, a_row.transpose(1, 2), a_col

    # ----- sparse ------------------------------------------------------
    def _sparse_forward(self, x):
        B, _, D = x.shape
        eps = 1e-6
        # ---- K, V patches ------------------------------------------------
        k_map = (
            self.k_proj(self.ln_in(x)).transpose(1, 2).reshape(B, D, self.Hi, self.Wi)
        )
        v_map = self.v_proj(x).transpose(1, 2).reshape(B, D, self.Hi, self.Wi)
        k = (
            F.unfold(k_map, 2, stride=2)
            .view(B, D, 4, self.Ho, self.Wo)
            .permute(0, 2, 3, 4, 1)
            .contiguous()
        )
        v = (
            F.unfold(v_map, 2, stride=2)
            .view(B, D, 4, self.Ho, self.Wo)
            .permute(0, 2, 3, 4, 1)
            .contiguous()
        )

        # ---- seed slots ---------------------------------------------------
        x_out = self.ln_out(
            self.seed_conv(x.transpose(1, 2).reshape(B, D, self.Hi, self.Wi))
            .flatten(2)
            .transpose(1, 2)
        )

        for _ in range(self.num_iters):
            q = (
                self.q_proj(self.ln_out(x_out))
                .view(B, self.Ho, self.Wo, D)
                .unsqueeze(1)
                .repeat(1, 4, 1, 1, 1)
            ).contiguous()  # (B,4,Ho,Wo,D)
            attn = natten_F.na2d_qk(q, k, 3) + self.rpb()
            attn = F.softmax(attn * torch.exp(self.tau), dim=-1) + eps  # (B,4,Ho,Wo,9)

            # ---- expand to 36 coeffs ------------------------------------
            flat = attn.reshape(B, 4, -1)  # (B,4,N_out*9)
            attn36 = flat.gather(-1, self.q_idx.view(1, 4, -1).expand(B, -1, -1)).view(
                B, 4, self.N_out, 36
            )
            denom = attn36.sum(dim=(1, 3), keepdim=True)
            a_col_h = attn36 / (denom + eps)

            # ---- build dense A_col --------------------------------------
            A_col = torch.zeros(B, self.N_in, self.N_out, device=x.device)
            idx_in = self.in_idx.view(1, 4, self.N_out, 36).expand(B, -1, -1, -1)
            A_col.scatter_add_(1, idx_in.reshape(B, -1, 36), a_col_h.reshape(B, -1, 36))

            # ---- updates -------------------------------------------------
            attn9 = a_col_h[..., :9].contiguous().view(B, 4, self.Ho, self.Wo, 9)
            updates = natten_F.na2d_av(attn9, v, 3)
            updates = updates.sum(dim=1).permute(0, 3, 1, 2).flatten(2).transpose(1, 2)
            x_out = x_out + updates
            x_out = x_out + self.ln_out(self.mlp(x_out))

        # ---- A_row via col_idx -------------------------------------------
        A_row = torch.zeros_like(A_col)
        idx_out = self.col_idx.view(1, 4, self.N_out, 36).expand(B, -1, -1, -1)
        flat_r = attn36.view(B, 4 * self.N_out, 36)
        A_row.scatter_add_(2, idx_out.reshape(B, -1, 36), flat_r)
        return x_out, A_row, A_col

    def forward(self, x: torch.Tensor):
        return self._sparse_forward(x) if self.loc else self._dense_forward(x)


# -----------------------------------------------------------------------------
# 3. SeNaTra Backbone
# -----------------------------------------------------------------------------
class SeNaTraBlock(nn.Module):
    def __init__(
        self, H: int, W: int, dim: int, mlp_ratio: float = 3.0, local: bool = True
    ):
        super().__init__()
        self.H, self.W = H, W
        self.ln1 = nn.LayerNorm(dim)
        if local:
            self.attn = LocalSelfAttention(dim, num_heads=8, window=7)
        else:
            self.attn = GlobalSelfAttention(dim, num_heads=8)
        self.ln2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x), self.H, self.W)
        x = x + self.mlp(self.ln2(x))
        return x


class PatchEmbed(nn.Module):
    """4x4 patch embed"""

    def __init__(self, img_size=224, patch_size=4, in_channels=3, embed_dim=96):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, patch_size, patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        self.H = self.W = img_size // patch_size

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return self.norm(x)


class SeNaTraStage(nn.Module):
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
                in_h,
                in_w,
                dim_in,
                loc=not dense_group,
                window=3 if not dense_group else None,
                num_iters=3,
            )
            self.proj = nn.Linear(dim_in, dim_out)
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
        if self.use_group:
            x, *_ = self.group(x)
            x = self.norm(self.proj(x))
        for blk in self.blocks:
            x = blk(x)
        return x


class SeNaTra(nn.Module):
    """Backbone variants T / B / L"""

    cfgs = {
        "T": dict(depths=[3, 4, 18, 5], dims=[64, 128, 256, 512], mlp=3.0),
        "B": dict(depths=[3, 4, 18, 5], dims=[128, 256, 512, 1024], mlp=2.0),
        "L": dict(depths=[3, 4, 18, 5], dims=[192, 384, 768, 1536], mlp=2.0),
    }

    def __init__(self, img_size=224, variant="T", in_chans=3, embed_dim=96):
        super().__init__()
        cfg = self.cfgs[variant]
        self.patch = PatchEmbed(img_size, 4, in_chans, embed_dim)
        self.proj0 = nn.Linear(embed_dim, cfg["dims"][0])
        self.norm0 = nn.LayerNorm(cfg["dims"][0])

        H = W = img_size // 4
        dim = cfg["dims"][0]

        # Stage 0 (no downsample, local)
        self.s0 = SeNaTraStage(
            H, W, dim, dim, cfg["depths"][0], cfg["mlp"], False, False, True
        )

        # Stage 1 (local grouping)
        self.s1 = SeNaTraStage(
            H, W, dim, cfg["dims"][1], cfg["depths"][1], cfg["mlp"], True, False, True
        )
        H //= 2
        W //= 2
        dim = cfg["dims"][1]

        # Stage 2 (local grouping)
        self.s2 = SeNaTraStage(
            H, W, dim, cfg["dims"][2], cfg["depths"][2], cfg["mlp"], True, False, True
        )
        H //= 2
        W //= 2
        dim = cfg["dims"][2]

        # Stage 3 (dense grouping, global attn)
        self.s3 = SeNaTraStage(
            H, W, dim, cfg["dims"][3], cfg["depths"][3], cfg["mlp"], True, True, False
        )

    def forward(self, x):
        x = self.norm0(self.proj0(self.patch(x)))
        x = self.s0(x)
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        return x  # final tokens 7×7
