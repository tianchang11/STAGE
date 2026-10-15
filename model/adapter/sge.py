"""Scale-sensitive Granularity Expert (S-GE)."""

import torch
import torch.nn as nn

from .cross_attention import CrossAttentionLayer
from .glu_blocks import GatedMultiScaleConvolution
from .sge_utils import normalize_sge_scales


class ScaleSensitiveGranularityExpert(nn.Module):
    """Extract a granularity representation and its multi-scale routing tokens."""

    def __init__(
        self,
        visual_dim,
        reduced_dim=512,
        num_heads=4,
        drop=0.0,
        scales=(1, 3, 5),
        cross_attention_layers=1,
    ):
        super().__init__()
        self.visual_dim = int(visual_dim)
        self.reduced_dim = int(reduced_dim)
        self.scales = normalize_sge_scales(scales)
        self.cross_attention_layers = max(int(cross_attention_layers), 1)

        if self.reduced_dim % num_heads != 0:
            raise ValueError("reduced_dim must be divisible by num_heads")

        # Equation (9): point-wise channel reduction W_r.
        self.channel_reduction = nn.Conv2d(self.visual_dim, self.reduced_dim, 1, bias=False)
        self.scale_units = GatedMultiScaleConvolution(self.reduced_dim, scales=self.scales)

        self.non_local_tokens = nn.ParameterDict()
        for kernel_size in self.scales:
            token = nn.Parameter(torch.empty(1, 1, self.reduced_dim))
            nn.init.normal_(token, std=0.02)
            self.non_local_tokens[str(kernel_size)] = token

        self.cross_attention = nn.ModuleDict({
            str(kernel_size): nn.ModuleList([
                CrossAttentionLayer(
                    in_dim=self.reduced_dim,
                    out_dim=self.reduced_dim,
                    num_heads=num_heads,
                    drop=drop,
                    use_ffn=False,
                )
                for _ in range(self.cross_attention_layers)
            ])
            for kernel_size in self.scales
        })

        # W_g projects concatenated non-local tokens to d_v.
        self.granularity_projection = nn.Linear(
            len(self.scales) * self.reduced_dim,
            self.visual_dim,
            bias=False,
        )

    def forward(self, feature_map, modality_token):
        """Return the granularity representation and multi-scale routing tokens."""
        raise NotImplementedError(
            "The S-GE forward implementation is omitted from the public release."
        )
