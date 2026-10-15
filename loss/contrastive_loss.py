import torch.nn as nn
import torch
import torch.nn.functional as F

from utils.dist_utils import gather_labels, gather_tensor, get_world_size


# contrastive loss function, adapted from
# https://sachinruk.github.io/blog/2021-03-07-clip.html
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def multi_positive_contrastive_loss(
    logits: torch.Tensor,
    labels_q: torch.Tensor,
    labels_k: torch.Tensor = None,
) -> torch.Tensor:
    """
    Multi-positive InfoNCE (sum-of-positives / MIL-NCE).

    For each query i:
        L_i = -log( sum_{j: y_j=y_i} exp(s_ij) / sum_k exp(s_ik) )

    Supports rectangular similarity (N, M):
    - labels_q: (N,) query-side IDs
    - labels_k: (M,) key-side IDs; defaults to labels_q (square matrix)

    Why not SupCon-avg (mean log-prob per positive):
    - Text features come from class prompts; same-pid text vectors are identical
    - K text columns for one pid have equal similarity by construction
    - SupCon-avg splits mass across K identical positives; loss floor is log(K), never 0
    - Sum-of-positives logsumexp over positives can reach 0 when perfectly aligned

    With one sample per ID, reduces to standard diagonal InfoNCE / CE.
    """
    if labels_q.ndim != 1:
        labels_q = labels_q.view(-1)
    if labels_k is None:
        labels_k = labels_q
    elif labels_k.ndim != 1:
        labels_k = labels_k.view(-1)

    n_q, n_k = logits.shape
    if labels_q.shape[0] != n_q:
        raise ValueError(
            f"labels_q length ({labels_q.shape[0]}) must match logits rows ({n_q})"
        )
    if labels_k.shape[0] != n_k:
        raise ValueError(
            f"labels_k length ({labels_k.shape[0]}) must match logits cols ({n_k})"
        )

    # Numerical stability
    logits = logits - logits.max(dim=1, keepdim=True)[0].detach()

    pos_mask = torch.eq(labels_q.unsqueeze(1), labels_k.unsqueeze(0))
    # Each query needs at least one positive; guard against log(0)
    if not pos_mask.any(dim=1).all():
        # Square same-order: fall back to diagonal; else skip rows with no positives
        if n_q == n_k and torch.equal(labels_q, labels_k):
            pos_mask = pos_mask | torch.eye(n_q, dtype=torch.bool, device=logits.device)
        else:
            # No-positive rows: treat max logit as positive to avoid NaN (rare)
            empty = ~pos_mask.any(dim=1)
            if empty.any():
                fallback = torch.zeros_like(pos_mask)
                fallback[empty, logits[empty].argmax(dim=1)] = True
                pos_mask = pos_mask | fallback

    # log(sum_pos exp(s)) - log(sum_all exp(s))
    neg_inf = torch.finfo(logits.dtype).min
    logits_pos = logits.masked_fill(~pos_mask, neg_inf)
    log_prob_pos = torch.logsumexp(logits_pos, dim=1)
    log_prob_all = torch.logsumexp(logits, dim=1)
    return -(log_prob_pos - log_prob_all).mean()


def clip_loss(similarity: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
    """
    Bidirectional contrastive loss.

    - labels is None: standard CLIP diagonal InfoNCE (one positive per row on diagonal)
    - labels given: multi-positive InfoNCE (all columns with same ID are positives)
    """
    if labels is None:
        caption_loss = contrastive_loss(similarity)
        image_loss = contrastive_loss(similarity.t())
        return (caption_loss + image_loss) / 2.0

    labels = labels.view(-1)
    # i2t: rows=image, cols=text; t2i: transposed
    loss_i2t = multi_positive_contrastive_loss(similarity, labels)
    loss_t2i = multi_positive_contrastive_loss(similarity.t(), labels)
    return (loss_i2t + loss_t2i) / 2.0


def clip_loss_distributed(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    labels: torch.Tensor = None,
    logit_scale: torch.Tensor = None,
) -> tuple:
    """
    Multi-GPU-safe bidirectional contrastive loss: all-gather features so similarity
    uses the *global batch*, matching single-GPU IMS_PER_BATCH negative pool size.

    Local-loss form:
      sim_i2t: (B_local, B_global) = local_img @ all_txt.T
      sim_t2i: (B_local, B_global) = local_txt @ all_img.T
    Local samples keep gradients; remote features are detached.

    Returns:
        loss: scalar loss
        sim_i2t: (B_local, B_global) for logging accuracy
    """
    if logit_scale is None:
        logit_scale = image_features.new_tensor(1.0)
    elif not torch.is_tensor(logit_scale):
        logit_scale = image_features.new_tensor(float(logit_scale))

    img = F.normalize(image_features, dim=-1)
    txt = F.normalize(text_features, dim=-1)

    world_size = get_world_size()
    if world_size == 1:
        sim = logit_scale * (img @ txt.t())
        if labels is None:
            return clip_loss(sim, labels=None), sim
        return clip_loss(sim, labels=labels.view(-1)), sim

    all_img = gather_tensor(img, keep_local_grad=True)
    all_txt = gather_tensor(txt, keep_local_grad=True)

    # local queries vs global keys
    sim_i2t = logit_scale * (img @ all_txt.t())
    sim_t2i = logit_scale * (txt @ all_img.t())

    if labels is None:
        # Standard CLIP: global diagonal; local rows offset by rank
        rank = torch.distributed.get_rank()
        b = img.shape[0]
        targets = torch.arange(b, device=img.device) + rank * b
        loss_i2t = F.cross_entropy(sim_i2t, targets)
        loss_t2i = F.cross_entropy(sim_t2i, targets)
        return (loss_i2t + loss_t2i) / 2.0, sim_i2t

    labels = labels.view(-1)
    all_labels = gather_labels(labels)
    loss_i2t = multi_positive_contrastive_loss(sim_i2t, labels, all_labels)
    loss_t2i = multi_positive_contrastive_loss(sim_t2i, labels, all_labels)
    return (loss_i2t + loss_t2i) / 2.0, sim_i2t


def text_anchor_alignment_loss_distributed(
    visual_features: torch.Tensor,
    text_anchors: torch.Tensor,
    labels: torch.Tensor,
    logit_scale: torch.Tensor = None,
) -> tuple:
    """Compute the paper's visual-to-text identity alignment objective."""
    if logit_scale is None:
        logit_scale = visual_features.new_tensor(1.0)
    elif not torch.is_tensor(logit_scale):
        logit_scale = visual_features.new_tensor(float(logit_scale))

    visual_features = F.normalize(visual_features, dim=-1)
    text_anchors = F.normalize(text_anchors, dim=-1)
    labels = labels.view(-1)

    if get_world_size() == 1:
        similarity = logit_scale * (visual_features @ text_anchors.t())
        return multi_positive_contrastive_loss(similarity, labels), similarity

    all_anchors = gather_tensor(text_anchors, keep_local_grad=True)
    all_labels = gather_labels(labels)
    similarity = logit_scale * (visual_features @ all_anchors.t())
    loss = multi_positive_contrastive_loss(similarity, labels, all_labels)
    return loss, similarity


def contrastive_accuracy(
    similarity: torch.Tensor,
    labels: torch.Tensor = None,
    labels_k: torch.Tensor = None,
) -> torch.Tensor:
    """
    Contrastive Top-1 accuracy.

    - labels is None: diagonal accuracy (argmax == i); under multi-GPU sim is (B_l, B_g)
    - labels given: correct if top-1 retrieved sample shares query ID
      (Identity Sampler: any same-ID column counts, not only diagonal)
    - labels_k: key-side IDs; default labels or gathered global labels
    """
    pred = similarity.argmax(dim=1)
    if labels is None:
        # Square diagonal; rectangular only valid when key prefix aligns with query
        n = min(similarity.size(0), similarity.size(1))
        target = torch.arange(n, device=similarity.device)
        return (pred[:n] == target).float().mean()

    labels = labels.view(-1)
    if labels.shape[0] != similarity.size(0):
        raise ValueError(
            f"labels length ({labels.shape[0]}) must match similarity rows ({similarity.size(0)})"
        )
    if labels_k is None:
        if similarity.size(0) != similarity.size(1):
            raise ValueError(
                "rectangular similarity requires labels_k (global key labels)"
            )
        labels_k = labels
    else:
        labels_k = labels_k.view(-1)
        if labels_k.shape[0] != similarity.size(1):
            raise ValueError(
                f"labels_k length ({labels_k.shape[0]}) must match similarity cols ({similarity.size(1)})"
            )
    # labels_k[pred[i]] == labels[i]  <=>  top-1 same class
    return (labels_k[pred] == labels).float().mean()
