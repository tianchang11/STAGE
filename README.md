# STAGE: Structured Text Anchors with Granularity Experts

This repository documents the architecture of **STAGE**, a two-stage framework for optical-SAR ship re-identification. STAGE learns modality-shared text anchors for the global ship structure, bow, hull, and stern, then uses those anchors to regularize multi-granularity visual learning.

The documented workflow follows the paper's two training stages:

1. **Stage I: Text-anchored Granularity Alignment** learns identity-specific structured text anchors with Adaptive Prompt Aggregation (APA).
2. **Stage II: Semantic-regularized Expert Fine-tuning** freezes the Stage I anchors and trains Scale-sensitive Granularity Experts (S-GE) together with the Local-to-Global Dual Router (LG-DR).

## Repository Structure

| Path | Purpose |
| --- | --- |
| `train_stage1.py` | Stage I entry point. Builds the frozen CLIP text encoder, structured text-anchor learner, and visual alignment projections. |
| `train_stage2.py` | Stage II entry point. Loads and freezes Stage I anchors and alignment projections, then fine-tunes the visual branch. |
| `test.py` | Evaluation entry point for a trained visual model. |
| `configs/stage1.yml` | Paper-aligned Stage I training configuration. |
| `configs/stage2.yml` | Paper-aligned Stage II training configuration. |
| `config/defaults.py` | Shared configuration schema and default values. |
| `datasets/hoss.py` | HOSS dataset indexing, optical/SAR modality assignment, and paired-sample construction. |
| `datasets/make_dataloader.py` | Training, validation, identity-sampling, and distributed data-loader construction. |
| `model/structured_text_anchors.py` | Granularity-aware prompt learning and APA. |
| `model/backbones/vit_transoss.py` | TransOSS visual encoder, modality-specific patch embeddings, modality embeddings, and selected-layer token extraction. |
| `model/adapter/stage_visual_learning.py` | Shared Stage I/II visual interface, overlapping region extraction, alignment projections, S-GE invocation, and LG-DR invocation. |
| `model/adapter/sge.py` | Scale-sensitive Granularity Expert implementation. |
| `model/adapter/lgdr.py` | Local-to-Global Dual Router implementation. |
| `model/adapter/glu_blocks.py` | Gated depthwise multi-scale convolution and feed-forward building blocks. |
| `model/adapter/cross_attention.py` | Cross-attention used by the scale-specific non-local tokens. |
| `loss/contrastive_loss.py` | Visual-to-text anchor alignment loss and distributed feature gathering. |
| `processor/processor_clip_stage1.py` | Stage I optimization, logging, evaluation, and checkpoint saving. |
| `processor/processor_clip_stage2.py` | Stage II semantic regularization, ReID optimization, evaluation, and best-checkpoint saving. |
| `utils/metrics.py` | All-to-All, Optical-to-SAR, and SAR-to-Optical retrieval metrics. |

## Main Modules

### Structured Text Anchors and APA

`model/structured_text_anchors.py` defines `StructuredTextAnchorLearner`.

For each identity, the module learns prompt tokens for three part-level granularities:

- `bow`
- `hull`
- `stern`

The prompts use the paper templates:

```text
A photo of the bow of a [X_bow] ship.
A photo of the hull of a [X_hull] ship.
A photo of the stern of a [X_stern] ship.
```

APA concatenates the three part-level tokens along the embedding dimension, projects them from `3 * d_p` to `d_p`, and adds the result to an independent global token `X_0`. The resulting `cls` prompt is:

```text
A photo of a [X_cls] ship.
```

The frozen CLIP text encoder converts these prompts into the structured text-anchor dictionary:

```python
{
    "cls": cls_anchor,
    "bow": bow_anchor,
    "hull": hull_anchor,
    "stern": stern_anchor,
}
```

### Granularity Region Extraction

`model/adapter/stage_visual_learning.py` converts selected Transformer-layer patch tokens back into spatial feature maps and averages them across layers. It then extracts the paper's overlapping vertical regions:

- Bow: `[0, 1/2]`
- Hull: `[1/4, 3/4]`
- Stern: `[1/2, 1]`

Each region therefore has height `H_p / 2`. Stage I average-pools these regions before alignment. Stage II sends them to separate S-GE modules.

### Scale-sensitive Granularity Experts

`model/adapter/sge.py` defines one expert for each part-level granularity. Every expert:

1. Reduces channels from `d_v` to `d_r` with a point-wise projection.
2. Applies gated depthwise convolutions with kernel sizes `1`, `3`, and `5`.
3. Adds a shared modality-specific embedding to each scale-specific non-local token.
4. Updates each non-local token through cross-attention over its scale-specific feature map.
5. Concatenates the updated tokens and projects them back to `d_v`.

The expert returns both the granularity representation used for semantic alignment and the multi-scale patch tokens consumed by LG-DR.

### Local-to-Global Dual Router

`model/adapter/lgdr.py` implements two routing steps:

- The **local router** projects the multi-scale tokens to `d_v`, scores them against the corresponding S-GE representation with cosine similarity, and retains the highest-scoring tokens according to the configured retention ratio.
- The **global router** concatenates routed tokens from the bow, hull, and stern. The Transformer class token supplies the query, while the routed tokens supply keys and values. A feed-forward network produces the final `cls` representation.

Only the final visual `cls` representation is required during inference.

## Training Objectives

### Stage I

Stage I aligns the projected visual and textual representations at all four granularities. The implementation computes the visual-to-text identity classification loss for `cls`, bow, hull, and stern, then averages the four losses.

The visual encoder and CLIP text encoder remain frozen. Trainable Stage I parameters are:

- Identity-specific prompt tokens
- APA projection
- Four visual alignment projections `W_a`

### Stage II

Stage II freezes the structured text anchors and all four Stage I alignment projections. Its objective is:

```text
L_II = L_ce + L_tri + alpha * L_reg
```

`L_reg` is the mean visual-to-text anchor loss over `cls`, bow, hull, and stern. The default configuration uses `alpha = 0.3`.


