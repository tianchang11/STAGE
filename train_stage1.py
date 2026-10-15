from utils.logger import setup_logger
from datasets import make_dataloader
from model import make_model
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from loss import make_loss
from processor.processor_clip_stage1 import do_train_stage1
import random
import torch
import numpy as np
import os
import argparse
from config import cfg
import clip
from model.structured_text_anchors import StructuredTextAnchorLearner


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_local_rank(args):
    """Resolve local rank from torch.distributed.launch / torchrun."""
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    return args.local_rank


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="STAGE Stage I training")
    parser.add_argument("--config_file", default="configs/stage1.yml", help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    parser.add_argument("--local-rank", "--local_rank", default=0, type=int)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    local_rank = get_local_rank(args)

    # Non-distributed: restrict visible GPUs via config; distributed: use external CUDA_VISIBLE_DEVICES
    if not cfg.MODEL.DIST_TRAIN:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.MODEL.DEVICE_ID

    cfg.freeze()
    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(local_rank)
        # Prefer NCCL on Linux; fall back to gloo elsewhere
        backend = "nccl" if torch.distributed.is_nccl_available() else "gloo"
        torch.distributed.init_process_group(backend=backend, init_method="env://")
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    else:
        rank = 0
        world_size = 1

    output_dir = cfg.OUTPUT_DIR
    if output_dir and rank == 0 and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.barrier()

    logger = setup_logger("transreid", output_dir if rank == 0 else None, if_train=True)
    if rank == 0:
        logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
        logger.info(args)
        logger.info("World size: {}, local_rank: {}".format(world_size, local_rank))
        if args.config_file != "":
            logger.info("Loaded configuration file {}".format(args.config_file))
            with open(args.config_file, "r") as cf:
                config_str = "\n" + cf.read()
                logger.info(config_str)
        logger.info("Running with config:\n{}".format(cfg))
        logger.info("Selected Transformer layers: {}".format(cfg.MODEL.PATCH_LAYER_GROUP))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_device = local_rank if (device == "cuda" and cfg.MODEL.DIST_TRAIN) else device
    try:
        clip_model, _ = clip.load("ViT-B/16", device=clip_device)
        # The CLIP text encoder remains frozen during Stage I.
        clip_model = clip_model.float()
        for p in clip_model.parameters():
            p.requires_grad = False
        clip_model.eval()
        if rank == 0:
            logger.info("CLIP loaded in float32 with a frozen text encoder")
    except ImportError:
        if rank == 0:
            logger.warning("CLIP unavailable; using random initialization")
        clip_model = None

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num)

    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)

    # Keep the center-loss optimizer while optimizing only Stage I parameters.
    _, optimizer_center = make_optimizer(cfg, model, center_criterion)

    anchor_device = f"cuda:{local_rank}" if (device == "cuda" and cfg.MODEL.DIST_TRAIN) else device
    text_anchor_learner = StructuredTextAnchorLearner(
        num_classes=num_classes,
        clip_model=clip_model,
        prompt_dim=512,
        num_prompt_tokens=4,
        device=anchor_device,
    )
    if rank == 0:
        logger.info("Using paper-aligned Adaptive Prompt Aggregation")

    trainable_params = []
    base = None
    if hasattr(model, "base"):
        base = model.base
    elif hasattr(model, "module") and hasattr(model.module, "base"):
        base = model.module.base
    if base is not None and hasattr(base, "stage_visual_learning"):
        trainable_params += list(
            base.stage_visual_learning.alignment_projections.parameters()
        )

    # The frozen CLIP text encoder is not registered as a trainable submodule.
    trainable_params += [p for p in text_anchor_learner.parameters() if p.requires_grad]

    optimizer = torch.optim.Adam(
        trainable_params,
        lr=cfg.SOLVER.BASE_LR,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
    )

    scheduler = create_scheduler(cfg, optimizer)
    do_train_stage1(
        cfg, model, center_criterion, train_loader, val_loader,
        optimizer, optimizer_center, scheduler, loss_func, num_query,
        local_rank, text_anchor_learner
    )
