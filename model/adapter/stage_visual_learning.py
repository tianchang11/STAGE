"""Visual learning modules for both stages of STAGE."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lgdr import LocalToGlobalDualRouter
from .sge import ScaleSensitiveGranularityExpert


class STAGEVisualLearning(nn.Module):
    """Extract and align ``cls``, bow, hull, and stern representations."""

    PART_GRANULARITIES = ("bow", "hull", "stern")

    def __init__(
        self,
        visual_dim=768,
        text_dim=512,
        feature_height=16,
        feature_width=8,
        reduced_dim=512,
        retention_ratio=0.75,
        scales=(1, 3, 5),
        cross_attention_layers=1,
        drop=0.0,
    ):
        super().__init__()
        self.visual_dim = int(visual_dim)
        self.text_dim = int(text_dim)
        self.feature_height = int(feature_height)
        self.feature_width = int(feature_width)

        self.alignment_projections = nn.ModuleDict({
            name: nn.Linear(self.visual_dim, self.text_dim, bias=False)
            for name in ("cls", *self.PART_GRANULARITIES)
        })
        self.modality_embeddings = nn.Parameter(
            torch.zeros(2, 1, int(reduced_dim))
        )
        nn.init.normal_(self.modality_embeddings, std=0.02)
        self.granularity_experts = nn.ModuleDict({
            name: ScaleSensitiveGranularityExpert(
                visual_dim=self.visual_dim,
                reduced_dim=reduced_dim,
                scales=scales,
                cross_attention_layers=cross_attention_layers,
                drop=drop,
            )
            for name in self.PART_GRANULARITIES
        })
        self.dual_router = LocalToGlobalDualRouter(
            reduced_dim=reduced_dim,
            visual_dim=self.visual_dim,
            retention_ratio=retention_ratio,
            drop=drop,
        )

    def _tokens_to_feature_map(self, tokens):
        batch_size, token_count, channels = tokens.shape
        expected_tokens = self.feature_height * self.feature_width
        if token_count != expected_tokens:
            raise ValueError(f"Expected {expected_tokens} patch tokens, got {token_count}")
        return tokens.transpose(1, 2).reshape(
            batch_size,
            channels,
            self.feature_height,
            self.feature_width,
        )

    def _average_overlapping_regions(self, selected_layer_tokens):
        """Implement the overlapping regions defined in equations (3) and (9)."""
        feature_maps = torch.stack(
            [self._tokens_to_feature_map(tokens) for tokens in selected_layer_tokens],
            dim=0,
        )
        averaged_map = feature_maps.mean(dim=0)
        quarter = self.feature_height // 4
        half = self.feature_height // 2
        return {
            "bow": averaged_map[:, :, :half, :],
            "hull": averaged_map[:, :, quarter:quarter + half, :],
            "stern": averaged_map[:, :, self.feature_height - half:, :],
        }

    def _align(self, representations):
        return {
            name: F.normalize(self.alignment_projections[name](feature), dim=-1)
            for name, feature in representations.items()
        }

    def forward_stage_one(self, selected_layer_tokens, cls_token):
        regions = self._average_overlapping_regions(selected_layer_tokens)
        representations = {"cls": cls_token}
        representations.update({
            name: region.flatten(2).mean(dim=-1)
            for name, region in regions.items()
        })
        return cls_token, self._align(representations)

    def forward_stage_two(
        self,
        selected_layer_tokens,
        cls_token,
        modality_ids,
        return_viz=False,
    ):
        regions = self._average_overlapping_regions(selected_layer_tokens)
        modality_ids = modality_ids.to(dtype=torch.long)
        modality_tokens = self.modality_embeddings[modality_ids]
        part_representations = {}
        multi_scale_tokens = {}
        for name in self.PART_GRANULARITIES:
            representation, tokens = self.granularity_experts[name](
                regions[name],
                modality_tokens,
            )
            part_representations[name] = representation
            multi_scale_tokens[name] = tokens

        router_output = self.dual_router(
            part_representations,
            multi_scale_tokens,
            cls_token,
            return_viz=return_viz,
        )
        representations = {"cls": router_output["cls"], **part_representations}
        output = self._align(representations)
        if return_viz:
            output.update({key: value for key, value in router_output.items() if key != "cls"})
        return representations["cls"], output
