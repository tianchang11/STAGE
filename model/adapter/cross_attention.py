"""Cross-attention layer used by S-GE and LG-DR."""

import torch.nn as nn

from .glu_blocks import GLUFeedForward


class CrossAttentionLayer(nn.Module):
    """
    Cross-attention: query token attends to patches as K/V (heterogeneous dims OK).

    Q lives in out_dim; K/V are projected from in_dim into out_dim.
    Only the query token is updated; patches are unchanged.

    Args:
        in_dim: patch (K/V) feature dim
        out_dim: query dim and attention space
        num_heads: number of heads
        drop: dropout rate
    """

    def __init__(self, in_dim, out_dim, num_heads=4, drop=0.1, use_ffn=True):
        super().__init__()
        if out_dim % num_heads != 0:
            raise ValueError(f"out_dim ({out_dim}) must be divisible by num_heads ({num_heads})")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.use_ffn = bool(use_ffn)
        self.head_dim = out_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Separate LayerNorm for Q and K/V (supports mismatched dims)
        self.norm_q = nn.LayerNorm(out_dim)
        self.norm_kv = nn.LayerNorm(in_dim)

        self.q_proj = nn.Linear(out_dim, out_dim)
        self.kv_proj = nn.Linear(in_dim, out_dim * 2, bias=True)
        self.out_proj = nn.Linear(out_dim, out_dim)
        self.attn_drop = nn.Dropout(drop)
        self.proj_drop = nn.Dropout(drop)

        if self.use_ffn:
            self.norm2 = nn.LayerNorm(out_dim)
            self.ffn = GLUFeedForward(out_dim, hidden_dim=out_dim * 2, drop=drop)

    def forward(self, q, p, return_attn=False):
        """
        Args:
            q: (B, 1, out_dim) learnable query token
            p: (B, N, in_dim) patch features (K/V)
            return_attn: if True, also return pre-dropout softmax attention

        Returns:
            q_out: (B, 1, out_dim), or (q_out, attn) with attn (B, H, 1, N)
        """
        B, N, _ = p.shape
        H, Dh = self.num_heads, self.head_dim

        q_norm = self.norm_q(q)
        p_norm = self.norm_kv(p)

        Q = self.q_proj(q_norm)
        K, V = self.kv_proj(p_norm).chunk(2, dim=-1)

        Q = Q.view(B, 1, H, Dh).transpose(1, 2)
        K = K.view(B, N, H, Dh).transpose(1, 2)
        V = V.view(B, N, H, Dh).transpose(1, 2)

        attn = ((Q @ K.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        attn_for_viz = attn if return_attn else None
        attn = self.attn_drop(attn)

        x = (attn @ V).transpose(1, 2).reshape(B, 1, self.out_dim)
        x = self.proj_drop(self.out_proj(x))

        q = q + x
        if self.use_ffn:
            q = q + self.ffn(self.norm2(q))

        if return_attn:
            return q, attn_for_viz
        return q
