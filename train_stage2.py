from utils.logger import setup_logger
from datasets import make_dataloader
from model import make_model
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from loss import make_loss
from processor.processor_clip_stage2 import do_train_stage2
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


def strip_module_prefix(state_dict):
    """Strip optional module. prefix from DDP checkpoints."""
    if not state_dict:
        return state_dict
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k[len("module."):] if k.startswith("module.") else k: v for k, v in state_dict.items()}


def load_structured_text_anchor_weights(text_anchor_learner, state_dict):
    """Load Stage I anchors and migrate only semantically compatible legacy keys."""
    state_dict = strip_module_prefix(state_dict)
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("clip_model.")}
    legacy_keys = {
        "bow_ctx": "bow_tokens",
        "hull_ctx": "hull_tokens",
        "stern_ctx": "stern_tokens",
        "x4_ctx": "cls_tokens",
    }
    is_legacy = any(key in state_dict for key in (*legacy_keys, "W"))
    migrated = {}
    for key, value in state_dict.items():
        if key in legacy_keys:
            migrated[legacy_keys[key]] = value
        elif is_legacy and (
            key == "W" or key.endswith(("_prefix", "_suffix", "_eot_index"))
        ):
            continue
        else:
            migrated[key] = value
    missing, unexpected = text_anchor_learner.load_state_dict(migrated, strict=False)
    return missing, unexpected


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="STAGE Stage II training")
    parser.add_argument("--config_file", default="configs/stage2.yml", help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    parser.add_argument("--local-rank", "--local_rank", default=0, type=int)
    parser.add_argument("--stage1-ckpt", default="logs/stage1/transformer_180.pth", type=str, help="path to stage1 checkpoint to load")
    parser.add_argument("--anchor-ckpt", default="", type=str, help="optional Stage I structured text-anchor checkpoint")
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
        logger.info(
            "STAGE settings | retention_ratio={} cross_attention_layers={} "
            "selected_layers={} scales={} alpha={}".format(
                cfg.MODEL.LGDR_RETENTION_RATIO,
                cfg.MODEL.CROSS_ATTENTION_LAYERS,
                cfg.MODEL.PATCH_LAYER_GROUP,
                cfg.MODEL.SGE_SCALES,
                cfg.MODEL.SEMANTIC_REGULARIZATION_WEIGHT,
            )
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_device = local_rank if (device == "cuda" and cfg.MODEL.DIST_TRAIN) else device

    # Load the same frozen CLIP text encoder used in Stage I.
    try:
        clip_model, _ = clip.load("ViT-B/16", device=clip_device)
        clip_model = clip_model.float()
        for p in clip_model.parameters():
            p.requires_grad = False
        clip_model.eval()
        if rank == 0:
            logger.info("CLIP loaded (text encoder frozen)")
    except ImportError:
        if rank == 0:
            logger.warning("CLIP unavailable; using random initialization")
        clip_model = None

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num)
    # make_model already loads ImageNet weights from cfg.MODEL.PRETRAIN_PATH (vit_b512_pre.pth)

    # Reconstruct the Stage I structured text-anchor learner.
    anchor_device = f"cuda:{local_rank}" if (device == "cuda" and cfg.MODEL.DIST_TRAIN) else device
    text_anchor_learner = StructuredTextAnchorLearner(
        num_classes=num_classes,
        clip_model=clip_model,
        prompt_dim=512,
        num_prompt_tokens=4,
        device=anchor_device,
    )
    if rank == 0:
        logger.info("Using fixed structured text anchors with paper-aligned APA")

    # Load Stage I anchors and alignment projections; initialize the visual encoder separately.
    anchors_loaded = False
    if args.stage1_ckpt:
        if os.path.exists(args.stage1_ckpt):
            cp = torch.load(args.stage1_ckpt, map_location="cpu")

            anchor_state = None
            if isinstance(cp, dict):
                anchor_state = cp.get("structured_text_anchors", cp.get("prompt_learner"))
            if anchor_state is not None:
                try:
                    missing, unexpected = load_structured_text_anchor_weights(
                        text_anchor_learner, anchor_state
                    )
                    if rank == 0:
                        print("Loaded structured text anchors from the Stage I checkpoint")
                        if missing:
                            print(f"  missing keys (ignored): {missing[:8]}{'...' if len(missing) > 8 else ''}")
                        if unexpected:
                            print(f"  unexpected keys (ignored): {unexpected[:8]}{'...' if len(unexpected) > 8 else ''}")
                    anchors_loaded = True
                except Exception as e:
                    if rank == 0:
                        print(f"Warning: failed to load Stage I structured text anchors: {e}")
            if isinstance(cp, dict) and "alignment_projections" in cp:
                try:
                    model.base.stage_visual_learning.alignment_projections.load_state_dict(
                        cp["alignment_projections"], strict=True
                    )
                    if rank == 0:
                        print("Loaded Stage I alignment projections")
                except Exception as e:
                    if rank == 0:
                        print(f"Warning: failed to load Stage I alignment projections: {e}")
            elif rank == 0:
                print("Warning: the Stage I checkpoint has no paper-aligned projection set")
        else:
            if rank == 0:
                print(f"Stage I checkpoint not found: {args.stage1_ckpt}")

    # Fall back to a standalone anchor checkpoint when needed.
    if not anchors_loaded and args.anchor_ckpt and os.path.exists(args.anchor_ckpt):
        try:
            anchor_checkpoint = torch.load(args.anchor_ckpt, map_location="cpu")
            if isinstance(anchor_checkpoint, dict) and "structured_text_anchors" in anchor_checkpoint:
                anchor_state_dict = anchor_checkpoint["structured_text_anchors"]
            elif isinstance(anchor_checkpoint, dict) and "prompt_learner" in anchor_checkpoint:
                anchor_state_dict = anchor_checkpoint["prompt_learner"]
            else:
                anchor_state_dict = anchor_checkpoint
            missing, unexpected = load_structured_text_anchor_weights(
                text_anchor_learner, anchor_state_dict
            )
            if rank == 0:
                print(f"Loaded structured text anchors from {args.anchor_ckpt}")
                if missing:
                    print(f"  missing keys (ignored): {missing[:8]}{'...' if len(missing) > 8 else ''}")
                if unexpected:
                    print(f"  unexpected keys (ignored): {unexpected[:8]}{'...' if len(unexpected) > 8 else ''}")
            anchors_loaded = True
        except Exception as e:
            if rank == 0:
                print(f"Warning: failed to load the structured text-anchor checkpoint: {e}")

    if not anchors_loaded and rank == 0:
        print("Warning: no structured text-anchor weights were loaded")

    # The structured text anchors remain fixed throughout Stage II.
    for p in text_anchor_learner.parameters():
        p.requires_grad = False
    text_anchor_learner.eval()

    # Verify that every structured text-anchor parameter is frozen.
    frozen_count = 0
    trainable_count = 0
    for name, param in text_anchor_learner.named_parameters():
        if param.requires_grad:
            trainable_count += 1
            if rank == 0:
                print(f"Warning: structured text-anchor parameter {name} is still trainable")
        else:
            frozen_count += 1
    if rank == 0:
        logger.info(f"Structured text anchors frozen - frozen: {frozen_count}, trainable: {trainable_count}")

    # Ensure the visual encoder follows the Stage II path.
    if hasattr(model, "base"):
        try:
            model.base.stage = 2
        except Exception:
            pass

    # Train the visual branch while keeping every W_a projection fixed.
    for name, param in model.named_parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if name.startswith("base."):
            param.requires_grad = True
    if hasattr(model, "base") and hasattr(model.base, "stage_visual_learning"):
        for p in model.base.stage_visual_learning.alignment_projections.parameters():
            p.requires_grad = False

    # Unfreeze classifier / bottleneck layers
    for name, param in model.named_parameters():
        if ("classifier" in name) or ("fc" in name) or ("bottleneck" in name) or ("arcface" in name):
            param.requires_grad = True

    # Log trainable parameter counts
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    projection_trainable = any(
        p.requires_grad
        for p in model.base.stage_visual_learning.alignment_projections.parameters()
    )
    if rank == 0:
        logger.info(f"Stage II trainable params: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
        logger.info(f"Stage I alignment projections frozen (trainable={projection_trainable})")

    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)

    # Create optimizer
    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)

    scheduler = create_scheduler(cfg, optimizer)

    do_train_stage2(
        cfg, model, center_criterion, train_loader, val_loader,
        optimizer, optimizer_center, scheduler, loss_func, num_query,
        local_rank, text_anchor_learner
    )
