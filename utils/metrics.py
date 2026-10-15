import torch
import numpy as np
import os
from utils.reranking import re_ranking


def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(1, -2, qf, gf.t())
    return dist_mat.cpu().numpy()

def cosine_similarity(qf, gf):
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
    dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
    dist_mat = np.arccos(dist_mat)
    return dist_mat


def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    # distmat g
    #    q    1 3 2 4
    #         4 1 2 3
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    #  0 2 1 3
    #  1 2 3 0
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]  # select one row
        # remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        # keep = np.invert(remove)
        keep = np.ones_like(g_pids, dtype=bool)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        #tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


def eval_cross_modal(distmat, q_pids, g_pids, q_camids, g_camids, q_cam, g_cam, max_rank=50):
    """Evaluate retrieval for a modality pair (camid 0=Optical, 1=SAR)."""
    q_mask = q_camids == q_cam
    g_mask = g_camids == g_cam
    if not np.any(q_mask) or not np.any(g_mask):
        raise ValueError(
            "Empty query/gallery for modality pair q_cam={}, g_cam={} "
            "(num_q={}, num_g={})".format(q_cam, g_cam, int(q_mask.sum()), int(g_mask.sum()))
        )
    sub_dist = distmat[q_mask][:, g_mask]
    return eval_func(
        sub_dist,
        q_pids[q_mask],
        g_pids[g_mask],
        q_camids[q_mask],
        g_camids[g_mask],
        max_rank=max_rank,
    )


def pack_cross_metrics(o2s_cmc, o2s_mAP, s2o_cmc, s2o_mAP):
    """Pack Optical→SAR (O2S) and SAR→Optical (S2O) metrics.

    Keeps both short aliases (O2S*/S2O*) and legacy OpticaltoSAR*/SARtoOptical* keys.
    """
    def _rank(cmc, k):
        idx = min(k - 1, len(cmc) - 1)
        return float(cmc[idx])

    o2s_mAP = float(o2s_mAP)
    s2o_mAP = float(s2o_mAP)
    o2s_r1, o2s_r5, o2s_r10 = _rank(o2s_cmc, 1), _rank(o2s_cmc, 5), _rank(o2s_cmc, 10)
    s2o_r1, s2o_r5, s2o_r10 = _rank(s2o_cmc, 1), _rank(s2o_cmc, 5), _rank(s2o_cmc, 10)
    return {
        # Preferred short names
        "O2SmAP": o2s_mAP,
        "O2SR1": o2s_r1,
        "O2SR5": o2s_r5,
        "O2SR10": o2s_r10,
        "S2OmAP": s2o_mAP,
        "S2OR1": s2o_r1,
        "S2OR5": s2o_r5,
        "S2OR10": s2o_r10,
        # Legacy aliases (kept for scripts / viz)
        "OpticaltoSARmAP": o2s_mAP,
        "OpticaltoSARR1": o2s_r1,
        "OpticaltoSARR5": o2s_r5,
        "OpticaltoSARR10": o2s_r10,
        "SARtoOpticalmAP": s2o_mAP,
        "SARtoOpticalR1": s2o_r1,
        "SARtoOpticalR5": s2o_r5,
        "SARtoOpticalR10": s2o_r10,
    }


def log_cross_metrics(logger, cross_metrics):
    logger.info(
        "O2S (Optical→SAR) mAP: {:.1%}  R1: {:.1%}  R5: {:.1%}  R10: {:.1%}".format(
            cross_metrics["O2SmAP"],
            cross_metrics["O2SR1"],
            cross_metrics["O2SR5"],
            cross_metrics["O2SR10"],
        )
    )
    logger.info(
        "S2O (SAR→Optical) mAP: {:.1%}  R1: {:.1%}  R5: {:.1%}  R10: {:.1%}".format(
            cross_metrics["S2OmAP"],
            cross_metrics["S2OR1"],
            cross_metrics["S2OR5"],
            cross_metrics["S2OR10"],
        )
    )


class R1_mAP_eval():
    def __init__(self, num_query, max_rank=50, feat_norm=True, reranking=False):
        super(R1_mAP_eval, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):  # called once for each batch
        feat, pid, camid = output
        self.feats.append(feat.cpu())
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):  # called after each epoch
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])

        g_camids = np.asarray(self.camids[self.num_query:])
        if self.reranking:
            print('=> Enter reranking')
            # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
            distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)

        else:
            print('=> Computing DistMat with euclidean_distance')
            distmat = euclidean_distance(qf, gf)
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        # Modality index 0 denotes optical imagery and 1 denotes SAR imagery.
        o2s_cmc, o2s_mAP = eval_cross_modal(
            distmat, q_pids, g_pids, q_camids, g_camids, q_cam=0, g_cam=1, max_rank=self.max_rank
        )
        s2o_cmc, s2o_mAP = eval_cross_modal(
            distmat, q_pids, g_pids, q_camids, g_camids, q_cam=1, g_cam=0, max_rank=self.max_rank
        )
        cross_metrics = pack_cross_metrics(o2s_cmc, o2s_mAP, s2o_cmc, s2o_mAP)

        return cmc, mAP, distmat, self.pids, self.camids, qf, gf, cross_metrics


