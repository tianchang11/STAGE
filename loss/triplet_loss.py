import torch
from torch import nn

from utils.dist_utils import gather_labels, gather_tensor, get_world_size


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist = dist - 2 * torch.matmul(x, y.t())
    # dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def cosine_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    x_norm = torch.pow(x, 2).sum(1, keepdim=True).sqrt().expand(m, n)
    y_norm = torch.pow(y, 2).sum(1, keepdim=True).sqrt().expand(n, m).t()
    xy_intersection = torch.mm(x, y.t())
    dist = xy_intersection/(x_norm * y_norm)
    dist = (1. - dist) / 2
    return dist


def hard_example_mining(dist_mat, labels, labels_gallery=None, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.

    Supports rectangular dist_mat [N_anchor, N_gallery] for multi-GPU:
    anchors are local samples, gallery is the global (gathered) batch.

    Args:
      dist_mat: pairwise distance, shape [N, N] or [N, M]
      labels: anchor labels, shape [N]
      labels_gallery: gallery labels, shape [M]; default same as labels
      return_inds: whether to return the indices
    """

    assert len(dist_mat.size()) == 2
    N, M = dist_mat.size()
    if labels_gallery is None:
        if N != M:
            raise ValueError("labels_gallery required when dist_mat is not square")
        labels_gallery = labels

    labels = labels.view(-1)
    labels_gallery = labels_gallery.view(-1)
    assert labels.size(0) == N
    assert labels_gallery.size(0) == M

    # shape [N, M]
    is_pos = labels.unsqueeze(1).eq(labels_gallery.unsqueeze(0))
    is_neg = ~is_pos

    # Ensure each anchor has at least one positive / negative when possible
    # (Identity sampler guarantees K>=2 positives in global batch if K>1)
    dist_ap_list = []
    dist_an_list = []
    p_inds_list = []
    n_inds_list = []

    for i in range(N):
        pos_idx = is_pos[i].nonzero(as_tuple=False).view(-1)
        neg_idx = is_neg[i].nonzero(as_tuple=False).view(-1)
        if pos_idx.numel() == 0:
            # fallback: self-distance 0 if square and same order, else mean dist
            dist_ap_list.append(dist_mat[i].new_zeros(()))
            p_inds_list.append(dist_mat.new_zeros((), dtype=torch.long))
        else:
            dpos = dist_mat[i].index_select(0, pos_idx)
            v, rel = torch.max(dpos, dim=0)
            dist_ap_list.append(v)
            p_inds_list.append(pos_idx[rel])
        if neg_idx.numel() == 0:
            dist_an_list.append(dist_mat[i].new_zeros(()))
            n_inds_list.append(dist_mat.new_zeros((), dtype=torch.long))
        else:
            dneg = dist_mat[i].index_select(0, neg_idx)
            v, rel = torch.min(dneg, dim=0)
            dist_an_list.append(v)
            n_inds_list.append(neg_idx[rel])

    dist_ap = torch.stack(dist_ap_list)
    dist_an = torch.stack(dist_an_list)

    if return_inds:
        return dist_ap, dist_an, torch.stack(p_inds_list), torch.stack(n_inds_list)
    return dist_ap, dist_an


class TripletLoss(object):
    """
    Triplet loss using HARDER example mining,
    modified based on original triplet loss using hard example mining.

    Under DDP, hard mining uses the gathered global batch as gallery so the
    positive/negative pool matches single-GPU IMS_PER_BATCH.
    """

    def __init__(self, margin=None, hard_factor=0.0):
        self.margin = margin
        self.hard_factor = hard_factor
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(
        self,
        global_feat,
        labels,
        normalize_feature=False,
        all_features=None,
        all_labels=None,
        gather_for_ddp=True,
    ):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
            if all_features is not None:
                all_features = normalize(all_features, axis=-1)

        # Auto-gather for multi-GPU so hard mining sees the global batch
        if all_features is None and gather_for_ddp and get_world_size() > 1:
            all_features = gather_tensor(global_feat, keep_local_grad=True)
            all_labels = gather_labels(labels)
        elif all_features is None:
            all_features = global_feat
            all_labels = labels
        elif all_labels is None:
            all_labels = labels

        dist_mat = euclidean_dist(global_feat, all_features)
        dist_ap, dist_an = hard_example_mining(
            dist_mat, labels, labels_gallery=all_labels
        )

        dist_ap *= (1.0 + self.hard_factor)
        dist_an *= (1.0 - self.hard_factor)

        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss, dist_ap, dist_an
