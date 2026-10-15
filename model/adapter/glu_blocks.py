"""GLU building blocks used by S-GE and cross-attention FFN."""

import torch.nn as nn

from .sge_utils import normalize_sge_scales


class GatedMultiScaleConvolution(nn.Module):
    """
    Paper-aligned S-GE units: SiLU(DWConv_k(x)) multiplied by Conv1(x).
    """

    def __init__(self, dim, scales=(1, 3, 5)):
        super().__init__()
        self.dim = dim
        self.scales = normalize_sge_scales(scales)
        self.act = nn.SiLU()

        self.branches = nn.ModuleDict()
        for k in self.scales:
            gate = nn.Conv2d(dim, dim, k, padding=k // 2, groups=dim, bias=False)
            value = nn.Conv2d(dim, dim, 1, bias=False)
            self.branches[str(k)] = nn.ModuleDict({
                "gate": gate,
                "value": value,
            })

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            list[Tensor]: feature maps in self.scales order, each (B, C, H, W)
        """
        outs = []
        for k in self.scales:
            b = self.branches[str(k)]
            outs.append(self.act(b["gate"](x)) * b["value"](x))
        return outs


class GLUFeedForward(nn.Module):
    """
    Gated Linear Unit FFN: SiLU(W1 x) * (W2 x) then W3.

    Args:
        dim: input/output feature dim
        hidden_dim: hidden size (default 2*dim)
        drop: dropout rate
    """

    def __init__(self, dim, hidden_dim=None, drop=0.1):
        super().__init__()
        hidden_dim = hidden_dim or dim * 2
        self.w1 = nn.Linear(dim, hidden_dim)  # gate
        self.w2 = nn.Linear(dim, hidden_dim)  # value
        self.w3 = nn.Linear(hidden_dim, dim)  # output proj
        self.act = nn.SiLU()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """
        Args:
            x: (B, N, dim)
        Returns:
            (B, N, dim)
        """
        return self.drop(self.w3(self.act(self.w1(x)) * self.w2(x)))
