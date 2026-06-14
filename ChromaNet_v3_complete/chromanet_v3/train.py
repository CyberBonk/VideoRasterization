"""
train.py — ChromaNet v3 Entry Point

Usage:
    python train.py --config configs/default.yaml
    python train.py --config configs/default.yaml --resume checkpoints/checkpoint_epoch010.pth
    python train.py --config configs/default.yaml --batch-size 16  # if GPU memory issues
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import torch, yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from model.chromaNet  import build_model
from model.losses     import build_loss
from data.dataset     import build_dataloaders
from training.trainer import Trainer
from training.scheduler import build_scheduler


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",         default="configs/default.yaml")
    p.add_argument("--resume",         default=None)
    p.add_argument("--lr",             type=float, default=None)
    p.add_argument("--epochs",         type=int,   default=None)
    p.add_argument("--batch-size",     type=int,   default=None)
    p.add_argument("--data-root",      default=None)
    p.add_argument("--checkpoint-dir", default=None)
    p.add_argument("--no-amp",         action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.lr:            cfg["training"]["lr"]              = args.lr
    if args.epochs:        cfg["training"]["epochs"]          = args.epochs
    if args.batch_size:    cfg["data"]["batch_size"]          = args.batch_size
    if args.data_root:     cfg["data"]["root"]                = args.data_root
    if args.checkpoint_dir:cfg["training"]["checkpoint_dir"]  = args.checkpoint_dir
    if args.no_amp:        cfg["training"]["mixed_precision"]  = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device: {device}")
    if device.type == "cuda":
        print(f"[info] GPU: {torch.cuda.get_device_name(0)}")

    model    = build_model(cfg)
    print(f"[build] ChromaNet v3 parameters: {model.count_parameters():,}")

    loss_fn  = build_loss(cfg)
    train_loader, val_loader = build_dataloaders(cfg)

    tc        = cfg.get("training", {})
    optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=tc.get("lr", 0.001),
                                   weight_decay=tc.get("weight_decay", 0.0001))
    scheduler = build_scheduler(optimizer, cfg)
    trainer   = Trainer(model, loss_fn, optimizer, scheduler, cfg, device)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    main()
