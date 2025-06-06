import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    """
    Split each image into non-overlapping 4x4 patches and linealy projects them to dimension C
    Input: x of shape (B, 3, H, W)
    Output: patches of shape (B, N, C) where N = (H/4) * (W/4)
    """

    def __init__(self, img_size=224, patch_size=4, in_channels=3, embed_dim=96):
        super().__init__()
        self.img_height = img_size
        self.img_width = img_size
        self.patch_size = patch_size
        self.grid_height = img_size // patch_size
        self.grid_width = img_size // patch_size
        self.num_patches = self.grid_height * self.grid_width

        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        x: Tensor of shape (B, 3, H, W)
        returns: patches of shape (B, N, C) where N = (H/4) * (W/4)
        """
        B, C_in, H, W = x.shape
        assert (
            H == self.img_height and W == self.img_width
        ), f"Input image size must be {self.img_height}x{self.img_width}, but got {H}x{W}"

        # (B, embed_dim, H / 4, W / 4)
        x = self.proj(x)

        # Flatten the spatial dimensions (B, embed_dim, (H / 4) * (W / 4))
        x = x.flatten(2)

        # Transpose to get (B, N, embed_dim)
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x


class GroupingLayer(nn.Module):
    """
    Spatial Grouping Layer
    - Input: X_in: (B, N_in, d) -> e.g., N_in = (H * W) at this stage
    - Output: X_out: (B, N_out, d), plus final A_down (N_out * N_in) and A_ups (N_out * N_in)
    - N_out = N_in // 4 (we assume H and W are multiples of 2, so grouping is by 2x2)
    - We do L=3 iterations of:
        * Compute A = Q(X_in) @ K(X_out)^T + B + M_loc_mask
        * Row-softmax -> A_ups
        * Column-normalize -> A_down
        * X_out = X_out + LN((A_down)^T @ V(X_in))
        * X_out = X_out + LN(MLP(X_out))
    - In early stages: we build M_loc_mask so that each X_out token only attends to a 3x3 patch of X_in
    - In final stages: M_loc_mask = 0 (no mask -> dense grouping)
    """

    def __init__(
        self,
        in_h,
        in_w,
        d,
        loc,
        num_iters=3,
        qkv_bias=True,
        mlp_ratio=3.0,
        window_size=3,
    ):
        """
        Args:
          in_h, in_w    : height & width of the input token grid (e.g. 56,56 or 28,28)
          d             : token embedding dimension
          loc           : if True, apply 3x3 local grouping mask; if False → dense grouping
          num_iters     : number of clustering iterations (paper uses L=3)
          qkv_bias      : whether to add bias in Q/K/V projections
          mlp_ratio     : hidden dimension ratio for the feed-forward MLP
          window_size   : local window side length (default=3)
        """
        super().__init__()
        self.in_h = in_h
        self.in_w = in_w
        self.N_in = in_h * in_w
        self.N_out = self.N_in // 4  # grouping into 2x2 -> reduce tokens by 4

        self.d = d
        self.num_iters = num_iters
        self.loc = loc
        self.window_size = window_size

        # 1) Initial linear projection to downsample (like strided conv)
        self.init_proj = nn.Conv2d(
            in_channels=d,
            out_channels=d,
            kernel_size=2,
            stride=2,
            padding=0,
        )
        self.init_norm = nn.LayerNorm(d)

        # 2) Q/K/V projections
        self.q_proj = nn.Linear(d, d, bias=qkv_bias)
        self.k_proj = nn.Linear(d, d, bias=qkv_bias)
        self.v_proj = nn.Linear(d, d, bias=qkv_bias)

        # 3) MLP for each grouped token update (after attention add)
        hidden_dim = int(d * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d),
        )
        self.mlp_norm = nn.LayerNorm(d)

        # 4) LayerNorm modules
        self.ln_out = nn.LayerNorm(d)
        self.ln_init = nn.LayerNorm(d)

        # 5) Relative positional bias table B:  a (2*win-1) x (2*win-1) matrix table, for each head=1
        self.rel_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), 1)
        )
        nn.init.trunc_normal_(self.rel_bias_table, std=0.02)

        # Pre-compute a pairwise indexing map from (i_out, j_out) to (i_in, j_in) offsets
        if self.loc:
            self.register_buffer("loc_mask", self._build_local_mask())
        else:
            # for dense grouping, no mask (se loc_mask to None)
            self.register_buffer("loc_mask", torch.zeros(1, 1, self.N_out, self.N_in))

    def _build_local_mask(self):
        """
        Build a mask of shape (1, 1, N_out, N_in) such that
        loc_mask[0,0,i_out,j_in] = 0 if the (i_out)th output token (2x2 downsample grid)
        and the (j_in)th input token (in_hxin_w) lie within a local window of size winxwin,
        else = -inf.  We tile it for batch & head dims = 1.
        """
        H, W = self.in_h, self.in_w
        Ho, Wo = H // 2, W // 2

        loc_mask = torch.full(
            (Ho * Wo, H * W), float("-inf")
        )  # each row = one out‐token, each col=input
        # Compute (y_out, x_out) coords of each grouped‐token center
        grid_out = []
        for i_o in range(Ho):
            for j_o in range(Wo):
                # Each output “center” is located at input‐grid coordinate (2*i_o, 2*j_o) (top‐left of the 2×2 group).
                grid_out.append((2 * i_o, 2 * j_o))
        # Precompute input token coords
        grid_in = [(i, j) for i in range(H) for j in range(W)]

        for idx_o, (yo, xo) in enumerate(grid_out):
            # Allowed (y_in, x_in) are within a win×win window around (yo, xo) in input
            for idx_i, (yi, xi) in enumerate(grid_in):
                if (
                    abs(yi - yo) < self.window_size // 2 + 1
                    and abs(xi - xo) < self.window_size // 2 + 1
                ):
                    # zero‐out the mask to allow attention
                    loc_mask[idx_o, idx_i] = 0.0

        # Now broadcast to (1, 1, N_out, N_in)
        return loc_mask.unsqueeze(0)

    def forward(self, X_in):
        """
        X_in: (B, N_in, d)   where N_in = in_h * in_w
        Returns:
           X_out: (B, N_out, d)
           A_down: (B, N_out, N_in)   (final column-normalized assignment)
           A_ups:  (B, N_out, N_in)   (final row-softmax assignment)
        """
        B, N_in, d = X_in.shape

        # print("X_in shape:", X_in.shape)
        # print("self.d:", self.d)
        # print("self.N_in:", self.N_in)
        # print("N_in:", N_in)
        # print("d:", d)

        assert N_in == self.N_in and d == self.d

        # 1) Initialize X_out by a strided “conv” -> reduce H×W → (H/2)×(W/2)
        #    Reshape X_in -> (B, d, H, W):
        H, W = self.in_h, self.in_w
        X_in_2d = X_in.transpose(1, 2).reshape(B, d, H, W)  # (B, d, H, W)
        # 2×2 Conv→ (B, d, H/2, W/2)
        X_out_2d = self.init_proj(X_in_2d)
        # Flatten back: (B, d, N_out) -> (B, N_out, d)
        X_out = X_out_2d.flatten(2).transpose(1, 2)  # (B, N_out, d)
        X_out = self.ln_init(X_out)

        # Pre‐compute key/value of X_in just once (we do not update X_in in iterations):
        K_in = self.k_proj(X_in)  # (B, N_in, d)
        V_in = self.v_proj(X_in)  # (B, N_in, d)

        # 2) Iterative clustering (L steps).  We accumulate final A_down / A_ups
        A_down = None
        A_ups = None

        for _ in range(self.num_iters):
            # a) Compute Q_out
            Q_out = self.q_proj(X_out)  # (B, N_out, d)

            # b) Compute raw attention logits: A = Q_out @ K_in^T   => (B, N_out, N_in)
            #    Add relative position bias B.  We need to tile B to (N_out, N_in).
            A = torch.bmm(Q_out, K_in.transpose(1, 2))  # (B, N_out, N_in)

            # Compute relative position indices for each out‐token → in‐token pair
            # (only within win window)
            # Add B_mask and loc_mask
            if self.loc:
                # (1, 1, N_out, N_in)  broadcast across batch
                mask = self.loc_mask  # zeros or -inf
                A = A + mask  # shape broadcast → (B, N_out, N_in)
                B_bias = self.rel_bias_table.mean()
                A = A + B_bias

            else:
                # dense grouping: no mask; add a single global bias
                A = A + self.rel_bias_table.mean()

            # c) Row‐softmax → A_ups  (softmax along dim=2)
            A_ups = F.softmax(A, dim=2)  # (B, N_out, N_in)

            # d) Column‐normalize -> A_down (so that sum over OUT dims = 1 for each IN)
            #    We can do:  A_down[b, i_out, i_in] = A_ups[b, i_out, i_in] / sum_j A_ups[b, j, i_in]
            A_down = A_ups / (
                1e-10 + A_ups.sum(dim=1, keepdim=True)
            )  # broadcast divide
            # shape: (B, N_out, N_in)

            # e) Weighted sum for updating X_out: X_out += LN( (A_down)^T @ V_in )
            #    (A_down)^T: (B, N_in, N_out),  V_in: (B, N_in, d) → => (B, N_out, d)
            # print("A_down shape:", A_down.shape)
            # print("V_in shape:", V_in.shape)
            new_centers = torch.bmm(A_ups, V_in)  # (B, N_out, d)
            X_out = X_out + self.ln_out(new_centers)  # (B, N_out, d)

            # f) MLP update: X_out += LN( MLP(X_out) )
            mlp_out = self.mlp(X_out)  # (B, N_out, d)
            X_out = X_out + self.mlp_norm(mlp_out)

        return X_out, A_down, A_ups


class WindowSelfAttention(nn.Module):
    """
    Single-head self-attention that can operate either
    (a) densely on the whole token set, or
    (b) locally in a (Kw x Kh) window around each query.
    """

    def __init__(self, dim, local: bool, win_size: int = 3):
        super().__init__()
        self.local = local
        self.win = win_size
        self.q = nn.Linear(dim, dim, bias=True)
        self.k = nn.Linear(dim, dim, bias=True)
        self.v = nn.Linear(dim, dim, bias=True)
        self.scale = dim**-0.5

    def forward(self, x, H: int, W: int):
        # x: (B, N, d)
        B, N, d = x.shape
        q, k, v = self.q(x), self.k(x), self.v(x)

        # dense attention --------------------------------------------------
        if not self.local:
            attn = (q @ k.transpose(1, 2)) * self.scale  # (B,N,N)
            prob = attn.softmax(dim=-1)
            return prob @ v  # (B,N,d)

        # local attention --------------------------------------------------
        q = q.reshape(B, H, W, d)
        k = k.reshape(B, H, W, d)
        v = v.reshape(B, H, W, d)

        pad = self.win // 2
        k_pad = F.pad(k, (0, 0, pad, pad, pad, pad))  # (B, H+2p, W+2p, d)
        v_pad = F.pad(v, (0, 0, pad, pad, pad, pad))

        outs = []
        for dy in range(self.win):
            for dx in range(self.win):
                k_slice = k_pad[:, dy : dy + H, dx : dx + W]  # (B,H,W,d)
                v_slice = v_pad[:, dy : dy + H, dx : dx + W]
                scores = (q * k_slice).sum(-1) * self.scale  # (B,H,W)
                outs.append((scores, v_slice))

        # Stack → softmax over window
        scores = torch.stack([sc for sc, _ in outs], dim=-1)  # (B,H,W,W*K)
        prob = scores.softmax(dim=-1)
        v_stacked = torch.stack([v_ for _, v_ in outs], dim=-1)  # (B,H,W,W*K,d)

        # print("self.win:", self.win)
        # print("prob shape:", prob.shape)
        # print("v_stacked shape:", v_stacked.shape)

        out = (prob.unsqueeze(-2) * v_stacked).sum(-1)  # (B,H,W,d)
        return out.reshape(B, N, d)


class SeNaTraBlock(nn.Module):
    def __init__(self, H, W, dim, mlp_ratio=3.0, loc=True):
        super().__init__()
        self.H, self.W = H, W
        self.ln1 = nn.LayerNorm(dim)
        self.attn = WindowSelfAttention(dim, local=loc, win_size=3)
        self.ln2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )

    def forward(self, x):
        h, w = self.H, self.W
        x = x + self.attn(self.ln1(x), h, w)
        x = x + self.mlp(self.ln2(x))
        return x, None, None


class SeNaTraStage(nn.Module):
    """
    • use_group = False - just N self-attention blocks (keeps HxW).
    • use_group = True  - GroupingLayer (+ 2x channels)     then N blocks.
      - dense_group = True  - grouping mask = all zeros (dense).
      - dense_group = False - 3x3 locality mask.
    """

    def __init__(
        self,
        in_h,
        in_w,
        dim_in,
        dim_out,
        num_blocks,
        mlp_ratio=3.0,
        use_group=True,
        dense_group=False,  # True only for final (7×7) stage
        local_blocks=True,  # False only for final stage
    ):
        super().__init__()
        self.use_group = use_group

        if use_group:
            self.group = GroupingLayer(
                in_h,
                in_w,
                dim_in,
                loc=not dense_group,  # False → dense grouping
                num_iters=3,
                mlp_ratio=mlp_ratio,
                window_size=3,
            )
            self.proj = nn.Linear(dim_in, dim_out)
            self.norm = nn.LayerNorm(dim_out)

            post_h, post_w, post_dim = in_h // 2, in_w // 2, dim_out
        else:  # stage-0 (no down-sample)
            self.group = self.proj = self.norm = None
            post_h, post_w, post_dim = in_h, in_w, dim_in

        self.blocks = nn.ModuleList(
            SeNaTraBlock(post_h, post_w, post_dim, mlp_ratio, loc=local_blocks)
            for _ in range(num_blocks)
        )

    def forward(self, x):
        if self.use_group:
            x, A_down, A_up = self.group(x)
            x = self.norm(self.proj(x))
        else:
            A_down = A_up = None

        for block in self.blocks:
            x, _, _ = block(x)
        return x, A_down, A_up


class SeNaTra(nn.Module):
    def __init__(self, img_size=224, in_chans=3, embed_dim=96, variant="T"):
        super().__init__()

        cfg = dict(
            T=dict(depths=[3, 4, 18, 5], dims=[64, 128, 256, 512], mlp_ratio=3.0),
            B=dict(depths=[3, 4, 18, 5], dims=[128, 256, 512, 1024], mlp_ratio=2.0),
            L=dict(depths=[3, 4, 18, 5], dims=[192, 384, 768, 1536], mlp_ratio=2.0),
        )[variant]

        self.patch = PatchEmbed(img_size, 4, in_chans, embed_dim)
        self.proj0 = nn.Linear(embed_dim, cfg["dims"][0])
        self.norm0 = nn.LayerNorm(cfg["dims"][0])

        # Current H, W, Dim of the tokens being processed
        current_h, current_w = img_size // 4, img_size // 4  # 56, 56
        current_dim = cfg["dims"][0]  # 64

        # ---- Stage-0 (56×56, NO grouping, local SA) -----------------
        self.s0 = SeNaTraStage(
            current_h,
            current_w,
            current_dim,
            current_dim,  # 56, 56, 64, 64
            cfg["depths"][0],
            cfg["mlp_ratio"],
            use_group=False,
            local_blocks=True,
        )
        # After s0, H, W, Dim are still current_h, current_w, current_dim

        # ---- Stage-1 (Input: 56x56, 64. Output: 28x28, 128) --------
        next_dim_s1 = cfg["dims"][1]  # 128
        self.s1 = SeNaTraStage(
            current_h,
            current_w,
            current_dim,  # Input to s1.group: 56, 56, 64
            next_dim_s1,  # Target dim for s1.blocks: 128
            cfg["depths"][1],
            cfg["mlp_ratio"],
            use_group=True,
            dense_group=False,
            local_blocks=True,
        )
        current_h, current_w = current_h // 2, current_w // 2  # Now 28, 28
        current_dim = next_dim_s1  # Now 128

        # ---- Stage-2 (Input: 28x28, 128. Output: 14x14, 256) --------
        next_dim_s2 = cfg["dims"][2]  # 256
        self.s2 = SeNaTraStage(
            current_h,
            current_w,
            current_dim,  # Input to s2.group: 28, 28, 128
            next_dim_s2,  # Target dim for s2.blocks: 256
            cfg["depths"][2],
            cfg["mlp_ratio"],
            use_group=True,
            dense_group=False,
            local_blocks=True,
        )
        current_h, current_w = current_h // 2, current_w // 2  # Now 14, 14
        current_dim = next_dim_s2  # Now 256

        # ---- Stage-3 (Input: 14x14, 256. Output: 7x7, 512) ---------------
        next_dim_s3 = cfg["dims"][3]  # 512
        self.s3 = SeNaTraStage(
            current_h,
            current_w,
            current_dim,  # Input to s3.group: 14, 14, 256
            next_dim_s3,  # Target dim for s3.blocks: 512
            cfg["depths"][3],
            cfg["mlp_ratio"],
            use_group=True,
            dense_group=True,
            local_blocks=False,
        )

    def forward(self, x):
        # x = self.patch(x)
        x = self.norm0(self.proj0(self.patch(x)))  # (B, 3136, 64)
        x, *_ = self.s0(x)  # 56×56  → (B, 3136, 64)
        x, *_ = self.s1(x)  # 28×28  → (B,  784, 128)
        x, *_ = self.s2(x)  # 14×14  → (B,  196, 256)
        x, *_ = self.s3(x)  #  7×7   → (B,   49, 512)
        return x  # final 7×7 tokens
