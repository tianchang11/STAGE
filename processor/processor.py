import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval, log_cross_metrics
from torch.cuda import amp
import torch.distributed as dist
from loss import clip_loss


def _is_main_process(cfg):
    if cfg.MODEL.DIST_TRAIN and dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def _unwrap(module):
    return module.module if isinstance(module, nn.parallel.DistributedDataParallel) else module


def do_train_pair(cfg, model, train_loader_pair, optimizer, scheduler, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info("start training")
    _LOCAL_PROCESS_GROUP = None

    if device:
        model.to(local_rank)
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

    loss_meter = AverageMeter()
    scaler = amp.GradScaler()

    # train pair
    if cfg.MODEL.PAIR:
        _unwrap(model).train_with_pair()
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            loss_meter.reset()
            scheduler.step(epoch)
            model.train()
            if hasattr(train_loader_pair, "sampler") and hasattr(train_loader_pair.sampler, "set_epoch"):
                train_loader_pair.sampler.set_epoch(epoch)
            for n_iter, (img, vid, target_cam) in enumerate(train_loader_pair):
                optimizer.zero_grad()
                img = img.to(device, non_blocking=True)
                target = vid.to(device, non_blocking=True)
                target_cam = target_cam.to(device, non_blocking=True)
                with amp.autocast(enabled=True):
                    # Paired pretraining returns a scalar CLIP loss in train_pair mode.
                    # The model gathers distributed features to match a global batch.
                    out = model(img, target, cam_label=target_cam)
                    if torch.is_tensor(out) and out.dim() == 0:
                        loss = out
                    elif torch.is_tensor(out) and out.dim() == 2:
                        # The legacy path returns a similarity matrix.
                        loss = clip_loss(out)
                    else:
                        loss = out

                scaler.scale(loss).backward()

                scaler.step(optimizer)
                scaler.update()

                loss_meter.update(loss.item(), img.shape[0])

                torch.cuda.synchronize()
                if _is_main_process(cfg) and (n_iter + 1) % log_period == 0:
                    logger.info(
                        "Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Base Lr: {:.2e}".format(
                            epoch, (n_iter + 1), len(train_loader_pair), loss_meter.avg, scheduler._get_lr(epoch)[0]
                        )
                    )

            end_time = time.time()
            if _is_main_process(cfg):
                logger.info(
                    "Epoch {} summary -> Loss: {:.3f}".format(
                        epoch, loss_meter.avg
                    )
                )
                time_per_batch = (end_time - start_time) / (n_iter + 1)
                if not cfg.MODEL.DIST_TRAIN:
                    logger.info(
                        "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]".format(
                            epoch, time_per_batch, train_loader_pair.batch_size / time_per_batch
                        )
                    )

            if epoch % checkpoint_period == 0:
                if _is_main_process(cfg):
                    torch.save(
                        _unwrap(model).state_dict(),
                        os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + "_{}.pth".format(epoch)),
                    )
                if cfg.MODEL.DIST_TRAIN:
                    dist.barrier()


def do_train(cfg, model, center_criterion, train_loader, val_loader, optimizer, optimizer_center, scheduler, loss_fn, num_query, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info("start training")
    _LOCAL_PROCESS_GROUP = None

    if device:
        model.to(local_rank)
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

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()

    # train
    _unwrap(model).train_with_single()
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()
        # Shuffle the DDP sampler at the start of each epoch.
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
                out = model(img, target, cam_label=target_cam, img_wh=img_wh)
                # Support both the base and STAGE model return signatures.
                if isinstance(out, (tuple, list)) and len(out) >= 2:
                    score, feat = out[0], out[1]
                else:
                    score, feat = out, out
                # TripletLoss gathers the global batch for hard mining under DDP.
                loss = loss_fn(score, feat, target, target_cam)

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

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if _is_main_process(cfg) and (n_iter + 1) % log_period == 0:
                logger.info(
                    "Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}".format(
                        epoch, (n_iter + 1), len(train_loader), loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]
                    )
                )

        end_time = time.time()
        if _is_main_process(cfg):
            logger.info(
                "Epoch {} summary -> Loss: {:.3f}, Acc: {:.3f}".format(
                    epoch, loss_meter.avg, acc_meter.avg
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
                torch.save(
                    _unwrap(model).state_dict(),
                    os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + "_{}.pth".format(epoch)),
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
                        img_wh = img_wh.to(device, non_blocking=True)
                        feat = model(img, cam_label=camids, img_wh=img_wh)
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
            if cfg.MODEL.DIST_TRAIN:
                dist.barrier()


def do_inference(cfg, model, val_loader, num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print("Using {} GPUs for inference".format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath, img_wh) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            img_wh = img_wh.to(device)
            feat = model(img, cam_label=camids, img_wh=img_wh)
            if isinstance(feat, tuple):
                feat = feat[0]
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _, cross_metrics = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    log_cross_metrics(logger, cross_metrics)
    return cmc[0], cmc[4]
