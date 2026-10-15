"""
Average-pooling components for granularity-aware learning.
"""

import torch.nn as nn


class GranularityAveragePooling(nn.Module):
    """Average spatial tokens within a granularity region."""
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):

        """Pool an input tensor with shape ``(B, C, S)`` over ``S``."""
        if x.dim() != 3:
            raise ValueError("GranularityAveragePooling expects input of shape (B, C, S)")
        return self.pool(x).squeeze(-1)
