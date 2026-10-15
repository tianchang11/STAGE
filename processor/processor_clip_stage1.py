import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval, log_cross_metrics
from torch.cuda import amp
import torch.distributed as dist
from loss import contrastive_accuracy, text_anchor_alignment_loss_distributed
from utils.dist_utils import gather_labels, get_world_size


def _is_main_process(cfg):
    if cfg.MODEL.DIST_TRAIN and dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def _unwrap(module):
    return module.module if isinstance(module, nn.parallel.DistributedDataParallel) else module


def _text_anchor_state_dict(text_anchor_learner):
    """Save Stage I structured text-anchor parameters without the frozen CLIP model."""
    state = _unwrap(text_anchor_learner).state_dict()
    return {k: v.detach().cpu() for k, v in state.items() if not k.startswith("clip_model.")}


def do_train_stage1(cfg, model, center_criterion, train_loader, val_loader, optimizer, optimizer_center, scheduler, loss_fn, num_query, local_rank, text_anchor_learner):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info("start training")
    logger.info("Stage I: text-anchored granularity alignment with APA")
    _LOCAL_PROCESS_GROUP = None

    # ---- freeze before DDP so unused params are consistent across ranks ----
    _unwrap(model).train_with_single()
    for name, param in model.named_parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if "stage_visual_learning.alignment_projections" in name:
            param.requires_grad = True

    for param in text_anchor_learner.parameters():
        param.requires_grad = True

    if device:
        model.to(local_rank)
        text_anchor_learner.to(local_rank)
        use_ddp = (
            cfg.MODEL.DIST_TRAIN
            and dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() > 1
        )
        if use_ddp:
            print("Using {} GPUs for training (world_size={})".format(
                torch.cuda.device_count(), dist.get_world_size()
            ))
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True
            )
            # Synchronize trainable structured text-anchor parameters across ranks.
            text_anchor_learner = torch.nn.parallel.DistributedDataParallel(
                text_anchor_learner, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True
            )

    loss_meter = AverageMeter()
    loss_b_meter = AverageMeter()
    loss_h_meter = AverageMeter()
    loss_s_meter = AverageMeter()
    loss_cls_meter = AverageMeter()
    acc_bow_meter = AverageMeter()
    acc_hull_meter = AverageMeter()
    acc_stern_meter = AverageMeter()
    acc_cls_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()

    text_anchor_learner.train()

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        loss_b_meter.reset()
        loss_h_meter.reset()
        loss_s_meter.reset()
        loss_cls_meter.reset()
        acc_bow_meter.reset()
        acc_hull_meter.reset()
        acc_stern_meter.reset()
        acc_cls_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()
        text_anchor_learner.train()

        # DDP sampler: shuffle per epoch
        if hasattr(train_loader, "batch_sampler") and hasattr(train_loader.batch_sampler, "sampler"):
            sampler = train_loader.batch_sampler.sampler
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
        elif hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        for n_iter, (img, vid, target_cam, target_view, img_wh) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device, non_blocking=True)
            target = vid.to(device, non_blocking=True)
            target_cam = target_cam.to(device, non_blocking=True)
            img_wh = img_wh.to(device, non_blocking=True)
            with amp.autocast(enabled=True):
                score, global_feat, cls_proj, granularity_features = model(
                    img, target, cam_label=target_cam, img_wh=img_wh
                )
                text_anchors = text_anchor_learner(target)

                # Get logit scale
                pl = _unwrap(text_anchor_learner)
                if hasattr(pl, "clip_model") and pl.clip_model is not None and hasattr(pl.clip_model, "logit_scale"):
                    logit_scale = pl.clip_model.logit_scale.exp()
                else:
                    logit_scale = torch.tensor(1.0, device=img.device)

                losses = {}
                similarities = {}
                for name in ("cls", "bow", "hull", "stern"):
                    losses[name], similarities[name] = text_anchor_alignment_loss_distributed(
                        granularity_features[name],
                        text_anchors[name],
                        labels=target,
                        logit_scale=logit_scale,
                    )
                loss = sum(losses.values()) / len(losses)
                loss_b, loss_h, loss_s, loss_cls = (
                    losses["bow"], losses["hull"], losses["stern"], losses["cls"]
                )
                sim_b, sim_h, sim_s, sim_cls = (
                    similarities["bow"], similarities["hull"],
                    similarities["stern"], similarities["cls"]
                )

                # Per-part loss logging
                loss_b_meter.update(loss_b.item(), img.shape[0])
                loss_h_meter.update(loss_h.item(), img.shape[0])
                loss_s_meter.update(loss_s.item(), img.shape[0])
                loss_cls_meter.update(loss_cls.item(), img.shape[0])

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if "center" in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= 1.0 / cfg.SOLVER.CENTER_LOSS_WEIGHT
                scaler.step(optimizer_center)
                scaler.update()

            # Accuracy: sim is (B_local, B_global); use global labels for same-class check
            labels_k = gather_labels(target) if get_world_size() > 1 else target
            acc_bow = contrastive_accuracy(sim_b, labels=target, labels_k=labels_k) if sim_b is not None else 0.0
            acc_hull = contrastive_accuracy(sim_h, labels=target, labels_k=labels_k) if sim_h is not None else 0.0
            acc_stern = contrastive_accuracy(sim_s, labels=target, labels_k=labels_k) if sim_s is not None else 0.0
            acc_cls = contrastive_accuracy(sim_cls, labels=target, labels_k=labels_k)

            loss_meter.update(loss.item(), img.shape[0])
            if sim_b is not None:
                acc_bow_meter.update(acc_bow, 1)
                acc_hull_meter.update(acc_hull, 1)
                acc_stern_meter.update(acc_stern, 1)
            acc_cls_meter.update(acc_cls, 1)

            torch.cuda.synchronize()
            if _is_main_process(cfg) and (n_iter + 1) % log_period == 0:
                logger.info(
                    "Epoch[{}] Iteration[{}/{}] Loss: {:.3f} (Bow: {:.3f}, Hull: {:.3f}, Stern: {:.3f}, Cls: {:.3f}), "
                    "Acc-Bow: {:.3f}, Acc-Hull: {:.3f}, Acc-Stern: {:.3f}, Acc-Cls: {:.3f}, Base Lr: {:.2e}".format(
                        epoch, (n_iter + 1), len(train_loader), loss_meter.avg,
                        loss_b_meter.avg, loss_h_meter.avg, loss_s_meter.avg, loss_cls_meter.avg,
                        acc_bow_meter.avg, acc_hull_meter.avg, acc_stern_meter.avg, acc_cls_meter.avg,
                        scheduler._get_lr(epoch)[0]
                    )
                )

        end_time = time.time()
        if _is_main_process(cfg):
            logger.info(
                "Epoch {} summary -> Loss: {:.3f} (Bow: {:.3f}, Hull: {:.3f}, Stern: {:.3f}, Cls: {:.3f}), "
                "Acc-Bow: {:.3f}, Acc-Hull: {:.3f}, Acc-Stern: {:.3f}, Acc-Cls: {:.3f}".format(
                    epoch, loss_meter.avg, loss_b_meter.avg, loss_h_meter.avg, loss_s_meter.avg, loss_cls_meter.avg,
                    acc_bow_meter.avg, acc_hull_meter.avg, acc_stern_meter.avg, acc_cls_meter.avg
                )
            )
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            if not cfg.MODEL.DIST_TRAIN:
                logger.info(
                    "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]".format(
                        epoch, time_per_batch, train_loader.batch_size / time_per_batch
                    )
                )

        if epoch % checkpoint_period == 0:
            if _is_main_process(cfg):
                # Stage II reloads the fixed anchors and alignment projections.
                m = _unwrap(model)
                ckpt = {
                    "epoch": epoch,
                    "structured_text_anchors": _text_anchor_state_dict(text_anchor_learner),
                }
                if hasattr(m, "base") and hasattr(m.base, "stage_visual_learning"):
                    ckpt["alignment_projections"] = {
                        k: v.detach().cpu()
                        for k, v in m.base.stage_visual_learning.alignment_projections.state_dict().items()
                    }
                save_path = os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + "_{}.pth".format(epoch))
                torch.save(ckpt, save_path)
                prompt_mb = sum(
                    t.numel() * t.element_size() for t in ckpt["structured_text_anchors"].values()
                ) / (1024 ** 2)
                proj_mb = sum(
                    t.numel() * t.element_size() for t in ckpt.get("alignment_projections", {}).values()
                ) / (1024 ** 2)
                logger.info(
                    "Saved Stage I checkpoint to {} (anchors {:.1f} MB, projections {:.1f} MB)".format(
                        save_path, prompt_mb, proj_mb
                    )
                )
            if cfg.MODEL.DIST_TRAIN:
                dist.barrier()

        if epoch % eval_period == 0:
            if _is_main_process(cfg):
                model.eval()
                for n_iter, (img, vid, camid, camids, target_view, _, img_wh) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device, non_blocking=True)
                        camids = camids.to(device, non_blocking=True)
                        target_view = target_view.to(device, non_blocking=True)
                        img_wh = img_wh.to(device, non_blocking=True)
                        feat = model(img, cam_label=camids, img_wh=img_wh)
                        # Eval mode returns (feat, cls_proj); use first element
                        if isinstance(feat, tuple):
                            feat = feat[0]
                        evaluator.update((feat, vid, camid))
                cmc, mAP, _, _, _, _, _, cross_metrics = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                log_cross_metrics(logger, cross_metrics)
                torch.cuda.empty_cache()
                model.train()
                text_anchor_learner.train()
            if cfg.MODEL.DIST_TRAIN:
                dist.barrier()
