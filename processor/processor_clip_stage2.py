import logging
import os
import time
import gc
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


def do_train_stage2(cfg, model, center_criterion, train_loader, val_loader, optimizer, optimizer_center, scheduler, loss_fn, num_query, local_rank, text_anchor_learner):
    log_period = cfg.SOLVER.LOG_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info("start training")
    logger.info("Stage II: semantic-regularized expert fine-tuning")
    _LOCAL_PROCESS_GROUP = None

    if device:
        model.to(local_rank)
        if text_anchor_learner is not None:
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
            # Structured text anchors are fixed in Stage II and require no DDP wrapper.

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    reid_loss_meter = AverageMeter()
    semantic_regularization_meter = AverageMeter()
    anchor_accuracy_meters = {
        name: AverageMeter() for name in ("cls", "bow", "hull", "stern")
    }


    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()

    # Keep best checkpoint by composite score:
    # score = overall_mAP + 0.5 * (O2S_mAP + S2O_mAP) / 2
    best_score = 0.0
    best_model_path = None

    # train
    _unwrap(model).train_with_single()

    if text_anchor_learner is not None:
        text_anchor_learner.eval()

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        reid_loss_meter.reset()
        semantic_regularization_meter.reset()
        for meter in anchor_accuracy_meters.values():
            meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()

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
                score, cls_representation, cls_projection, granularity_features = model(
                    img, target, cam_label=target_cam, img_wh=img_wh
                )

                learner = _unwrap(text_anchor_learner) if text_anchor_learner is not None else None
                if learner is not None and learner.clip_model is not None and hasattr(learner.clip_model, "logit_scale"):
                    logit_scale = learner.clip_model.logit_scale.exp()
                else:
                    logit_scale = torch.tensor(1.0, device=img.device)

                text_anchors = None
                if text_anchor_learner is not None:
                    with torch.no_grad():
                        text_anchors = text_anchor_learner(target)

                semantic_losses = {}
                similarities = {}
                for name in ("cls", "bow", "hull", "stern"):
                    semantic_losses[name], similarities[name] = text_anchor_alignment_loss_distributed(
                        granularity_features[name],
                        text_anchors[name],
                        labels=target,
                        logit_scale=logit_scale,
                    )
                semantic_regularization = sum(semantic_losses.values()) / len(semantic_losses)
                reid_loss = loss_fn(score, cls_representation, target, target_cam)
                loss = reid_loss + (
                    cfg.MODEL.SEMANTIC_REGULARIZATION_WEIGHT * semantic_regularization
                )

                reid_loss_meter.update(reid_loss.item(), img.shape[0])
                semantic_regularization_meter.update(
                    semantic_regularization.item(), img.shape[0]
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if "center" in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= 1.0 / cfg.SOLVER.CENTER_LOSS_WEIGHT
                scaler.step(optimizer_center)
                scaler.update()

            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            # Report identity retrieval accuracy against the fixed text anchors.
            labels_k = gather_labels(target) if get_world_size() > 1 else target
            for name, similarity in similarities.items():
                anchor_accuracy = contrastive_accuracy(
                    similarity, labels=target, labels_k=labels_k
                )
                anchor_accuracy_meters[name].update(anchor_accuracy, 1)

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if _is_main_process(cfg) and (n_iter + 1) % log_period == 0:
                logger.info(
                    "Epoch[{}] Iteration[{}/{}] Loss: {:.3f} "
                    "(ReID: {:.3f}, L_reg: {:.3f}), Acc: {:.3f}, "
                    "Anchor-Acc-Cls: {:.3f}, Anchor-Acc-Bow: {:.3f}, "
                    "Anchor-Acc-Hull: {:.3f}, Anchor-Acc-Stern: {:.3f}, "
                    "Base Lr: {:.2e}".format(
                        epoch, (n_iter + 1), len(train_loader), loss_meter.avg,
                        reid_loss_meter.avg, semantic_regularization_meter.avg,
                        acc_meter.avg,
                        anchor_accuracy_meters["cls"].avg,
                        anchor_accuracy_meters["bow"].avg,
                        anchor_accuracy_meters["hull"].avg,
                        anchor_accuracy_meters["stern"].avg,
                        scheduler._get_lr(epoch)[0]
                    )
                )

        end_time = time.time()
        if _is_main_process(cfg):
            logger.info(
                "Epoch {} summary -> Loss: {:.3f} (ReID: {:.3f}, L_reg: {:.3f}), "
                "Acc: {:.3f}, Anchor-Acc-Cls: {:.3f}, Anchor-Acc-Bow: {:.3f}, "
                "Anchor-Acc-Hull: {:.3f}, Anchor-Acc-Stern: {:.3f}".format(
                    epoch, loss_meter.avg,
                    reid_loss_meter.avg, semantic_regularization_meter.avg,
                    acc_meter.avg,
                    anchor_accuracy_meters["cls"].avg,
                    anchor_accuracy_meters["bow"].avg,
                    anchor_accuracy_meters["hull"].avg,
                    anchor_accuracy_meters["stern"].avg,
                )
            )
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            if not cfg.MODEL.DIST_TRAIN:
                logger.info(
                    "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]".format(
                        epoch, time_per_batch, train_loader.batch_size / time_per_batch
                    )
                )

        if epoch % eval_period == 0:
            # DDP note:
            # - Only rank0 runs full validation (metrics / best-ckpt).
            # - Under multi-GPU, optimizer/model already occupy most of 11GB.
            #   Using TEST.IMS_PER_BATCH=256 on rank0 alone OOMs for heavy ablations
            #   (topk>=0.75 / cross-attn>=3), while other ranks idle at barrier and
            #   eventually hit NCCL collective timeout. Micro-batch eval avoids this.
            if _is_main_process(cfg):
                torch.cuda.empty_cache()
                gc.collect()

                model.eval()
                eval_model = _unwrap(model)
                # Cap per-forward eval batch on limited VRAM (2080 Ti ~11GB).
                eval_micro_batch = int(getattr(cfg.TEST, "EVAL_MICRO_BATCH", 0) or 0)
                if eval_micro_batch <= 0:
                    eval_micro_batch = 32 if cfg.MODEL.DIST_TRAIN else int(cfg.TEST.IMS_PER_BATCH)

                with torch.no_grad():
                    for n_iter, (img, vid, camid, camids, target_view, _, img_wh) in enumerate(val_loader):
                        img = img.to(device, non_blocking=True)
                        camids = camids.to(device, non_blocking=True)
                        img_wh = img_wh.to(device, non_blocking=True)
                        batch_size = img.shape[0]
                        for start in range(0, batch_size, eval_micro_batch):
                            end = min(start + eval_micro_batch, batch_size)
                            feat = eval_model(
                                img[start:end],
                                cam_label=camids[start:end],
                                img_wh=img_wh[start:end],
                            )
                            if isinstance(feat, tuple):
                                feat = feat[0]
                            evaluator.update((feat, vid[start:end], camid[start:end]))
                            del feat
                cmc, mAP, _, _, _, _, _, cross_metrics = evaluator.compute()
                evaluator.reset()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                log_cross_metrics(logger, cross_metrics)

                mAP_value = float(mAP)
                o2s_mAP = float(cross_metrics["O2SmAP"])
                s2o_mAP = float(cross_metrics["S2OmAP"])
                score = mAP_value + 0.5 * (o2s_mAP + s2o_mAP) / 2.0
                if score > best_score:
                    if best_model_path is not None and os.path.exists(best_model_path):
                        try:
                            os.remove(best_model_path)
                            logger.info("Removed previous best model: {}".format(best_model_path))
                        except OSError as e:
                            logger.info("Failed to remove previous best model {}: {}".format(best_model_path, e))

                    best_score = score
                    best_model_path = os.path.join(
                        cfg.OUTPUT_DIR,
                        "{}_best_score_{:.2f}.pth".format(cfg.MODEL.NAME, best_score * 100.0),
                    )
                    torch.save(eval_model.state_dict(), best_model_path)
                    logger.info(
                        "New best model saved at epoch {} with score {:.4f} "
                        "(mAP {:.1%}, O2S {:.1%}, S2O {:.1%}): {}".format(
                            epoch, best_score, mAP_value, o2s_mAP, s2o_mAP, best_model_path
                        )
                    )

                # free eval tensors before other ranks leave the barrier
                try:
                    del img, camids, img_wh
                except Exception:
                    pass
                torch.cuda.empty_cache()
                gc.collect()
                model.train()
            if cfg.MODEL.DIST_TRAIN and dist.is_available() and dist.is_initialized():
                dist.barrier()
