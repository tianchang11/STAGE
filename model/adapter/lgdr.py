"""Local-to-Global Dual Router (LG-DR)."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .glu_blocks import GLUFeedForward


class LocalToGlobalDualRouter(nn.Module):
    """Select informative local tokens and aggregate them with global guidance."""

    GRANULARITIES = ("bow", "hull", "stern")

    def __init__(self, reduced_dim, visual_dim, retention_ratio=0.75, drop=0.0):
        super().__init__()
        self.retention_ratio = float(retention_ratio)
        if not 0.0 < self.retention_ratio <= 1.0:
            raise ValueError("retention_ratio must be in (0, 1]")

        self.routing_projection = nn.Linear(reduced_dim, visual_dim, bias=False)
        self.query_projection = nn.Linear(visual_dim, visual_dim, bias=False)
        self.key_projection = nn.Linear(visual_dim, visual_dim, bias=False)
        self.value_projection = nn.Linear(visual_dim, visual_dim, bias=False)
        self.scale = visual_dim ** -0.5
        self.attention_dropout = nn.Dropout(drop)
        self.feed_forward = GLUFeedForward(visual_dim, hidden_dim=2 * visual_dim, drop=drop)

    def _select_local_tokens(self, representation, multi_scale_tokens):
        routing_tokens = self.routing_projection(multi_scale_tokens)
        scores = F.cosine_similarity(representation.unsqueeze(1), routing_tokens, dim=-1)
        retained_count = max(1, math.ceil(self.retention_ratio * routing_tokens.shape[1]))
        topk_scores, topk_indices = scores.topk(retained_count, dim=1)
        selected = torch.gather(
            routing_tokens,
            dim=1,
            index=topk_indices.unsqueeze(-1).expand(-1, -1, routing_tokens.shape[-1]),
        )
        return selected, topk_indices, topk_scores

    def forward(self, granularity_representations, multi_scale_tokens, cls_token, return_viz=False):
        """Return the routed global representation and optional visualization data."""
        raise NotImplementedError(
            "The LG-DR forward implementation is omitted from the public release."
        )
