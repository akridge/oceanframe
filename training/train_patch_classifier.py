#!/usr/bin/env python3
"""
Patch classifier trainer for NOAA Pacific benthic-cover annotation patches.

Fine-tunes a pretrained backbone (default: DINOv3 ViT-B) on
https://huggingface.co/datasets/NMFS-OSI/noaa-pacific-benthic-cover-t1-all
with class-imbalance handling, and saves a checkpoint that
`infer_segments.py` can use to classify SAM-generated segments.

DINOv3 weights are gated on Hugging Face: accept the license at
https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m and run
`huggingface-cli login` once. To skip that, use the ungated DINOv2 fallback:
`--model vit_base_patch14_reg4_dinov2.lvd142m`.

Class names, counts, and the train/val split are discovered from the dataset
at runtime, so the script works unchanged across dataset revisions (t1/t2,
region subsets, etc.).

Quick start (single GPU):
    python train_patch_classifier.py --output-dir runs/dinov3b

Smoke test on a small subset:
    python train_patch_classifier.py --limit 2000 --epochs 2 --model vit_small_patch16_dinov3.lvd1689m
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

import timm
from timm.data import resolve_model_data_config
from torchvision import transforms

from datasets import load_dataset
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


# ── Config ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset", default="NMFS-OSI/noaa-pacific-benthic-cover-t1-all",
                   help="Hugging Face dataset repo id")
    p.add_argument("--image-col", default=None, help="Image column name (auto-detected if omitted)")
    p.add_argument("--label-col", default=None, help="Label column name (auto-detected if omitted)")
    p.add_argument("--val-frac", type=float, default=0.1,
                   help="Stratified validation fraction when the dataset has no val/test split")
    p.add_argument("--limit", type=int, default=0, help="Use only N training examples (0 = all); for smoke tests")

    p.add_argument("--model", default="vit_base_patch16_dinov3.lvd1689m",
                   help="timm model name. Lighter: vit_small_patch16_dinov3.lvd1689m. "
                        "Ungated fallback (no HF login): vit_base_patch14_reg4_dinov2.lvd142m. "
                        "CPU-friendly: convnext_tiny.fcmae_ft_in22k_in1k")
    p.add_argument("--img-size", type=int, default=224, help="Input resolution")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--accum", type=int, default=1, help="Gradient accumulation steps")
    p.add_argument("--lr", type=float, default=5e-5, help="Backbone learning rate")
    p.add_argument("--head-lr-mult", type=float, default=10.0, help="Head LR = lr * this")
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=float, default=1.0)
    p.add_argument("--freeze-epochs", type=int, default=1,
                   help="Train only the classifier head for the first N epochs")
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--drop-path", type=float, default=0.1, help="Stochastic depth rate")

    p.add_argument("--balance", choices=["sampler", "loss", "none"], default="sampler",
                   help="Class-imbalance strategy: oversample rare classes (sampler), "
                        "weight the loss (loss), or neither")
    p.add_argument("--sampler-power", type=float, default=0.5,
                   help="Sampling weight = count^-power. 0.5 (sqrt) is a good middle ground; "
                        "1.0 fully equalizes classes and can overfit tiny ones")

    p.add_argument("--ema-decay", type=float, default=0.999, help="Weight EMA decay (0 disables)")
    p.add_argument("--patience", type=int, default=6, help="Early stop after N epochs without macro-F1 improvement")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    p.add_argument("--output-dir", default="runs/patch_classifier")
    p.add_argument("--export-onnx", action="store_true", help="Export best checkpoint to ONNX after training")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── Data ───────────────────────────────────────────────────────────────────────

def detect_columns(ds, image_col: str | None, label_col: str | None) -> tuple[str, str]:
    from datasets import ClassLabel, Image as HFImage
    feats = ds.features
    if image_col is None:
        image_col = next((k for k, v in feats.items() if isinstance(v, HFImage)), None)
    if label_col is None:
        label_col = next((k for k, v in feats.items() if isinstance(v, ClassLabel)), None)
        if label_col is None:
            label_col = next((k for k in ("label", "labels", "class", "category") if k in feats), None)
    if image_col is None or label_col is None:
        raise SystemExit(f"Could not detect image/label columns in features: {list(feats)}. "
                         f"Pass --image-col / --label-col explicitly.")
    return image_col, label_col


def load_splits(args):
    """Return (train_ds, val_ds, class_names, image_col, label_col)."""
    from datasets import ClassLabel
    dsd = load_dataset(args.dataset)
    train_key = "train" if "train" in dsd else list(dsd.keys())[0]
    image_col, label_col = detect_columns(dsd[train_key], args.image_col, args.label_col)

    # Stratified split needs a ClassLabel column; encode string labels if needed.
    if not isinstance(dsd[train_key].features[label_col], ClassLabel):
        dsd = dsd.class_encode_column(label_col)

    val_key = next((k for k in ("validation", "val", "test") if k in dsd), None)
    if val_key is None:
        split = dsd[train_key].train_test_split(
            test_size=args.val_frac, stratify_by_column=label_col, seed=args.seed
        )
        train_ds, val_ds = split["train"], split["test"]
        print(f"No val split in dataset; made a stratified {1 - args.val_frac:.0%}/{args.val_frac:.0%} split.")
    else:
        train_ds, val_ds = dsd[train_key], dsd[val_key]

    if args.limit:
        train_ds = train_ds.shuffle(seed=args.seed).select(range(min(args.limit, len(train_ds))))
        val_ds = val_ds.shuffle(seed=args.seed).select(range(min(max(args.limit // 5, 200), len(val_ds))))

    class_names = train_ds.features[label_col].names
    return train_ds, val_ds, class_names, image_col, label_col


def build_transforms(img_size: int, mean, std):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0), ratio=(0.85, 1.18)),
        # Benthic patches have no canonical orientation: flips + 90° rotations are free augmentation.
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomApply([transforms.RandomRotation(90)], p=0.5),
        # Mild photometric jitter; underwater color carries class signal, so keep hue shifts small.
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.03),
        transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 1.5))], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.12)),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, eval_tf


def make_torch_transform(tf, image_col, label_col):
    def apply(batch):
        batch["pixel_values"] = [tf(img.convert("RGB")) for img in batch[image_col]]
        batch["target"] = batch[label_col]
        return batch
    return apply


def collate(batch):
    x = torch.stack([b["pixel_values"] for b in batch])
    y = torch.tensor([b["target"] for b in batch], dtype=torch.long)
    return x, y


# ── Model helpers ──────────────────────────────────────────────────────────────

def create_model(name: str, num_classes: int, img_size: int, drop_path: float) -> nn.Module:
    try:
        try:
            return timm.create_model(name, pretrained=True, num_classes=num_classes,
                                     img_size=img_size, drop_path_rate=drop_path)
        except TypeError:
            # Some backbones (e.g. ConvNeXt) don't take img_size / drop_path kwargs.
            return timm.create_model(name, pretrained=True, num_classes=num_classes)
    except Exception as e:
        if "dinov3" in name and any(s in str(e).lower() for s in ("gated", "403", "401", "authoriz", "access")):
            raise SystemExit(
                f"Could not download weights for {name} — DINOv3 checkpoints are gated on "
                f"Hugging Face.\nFix: accept the license at "
                f"https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m then run "
                f"`huggingface-cli login`.\nOr use the ungated fallback: "
                f"--model vit_base_patch14_reg4_dinov2.lvd142m\nOriginal error: {e}"
            ) from e
        raise


def param_groups(model: nn.Module, lr: float, head_lr_mult: float, weight_decay: float):
    head_ids = {id(p) for p in model.get_classifier().parameters()}
    groups = {}
    for p in model.parameters():
        if not p.requires_grad:
            continue
        is_head = id(p) in head_ids
        decay = p.ndim > 1  # no weight decay on biases/norms
        key = (is_head, decay)
        groups.setdefault(key, []).append(p)
    return [
        {"params": ps,
         "lr": lr * (head_lr_mult if is_head else 1.0),
         "weight_decay": weight_decay if decay else 0.0}
        for (is_head, decay), ps in groups.items()
    ]


def set_backbone_frozen(model: nn.Module, frozen: bool) -> None:
    head_ids = {id(p) for p in model.get_classifier().parameters()}
    for p in model.parameters():
        if id(p) not in head_ids:
            p.requires_grad = not frozen


class ModelEma:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        ema_state, model_state = self.module.state_dict(), model.state_dict()
        for k, v in ema_state.items():
            src = model_state[k].detach()
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(src, alpha=1.0 - self.decay)
            else:
                v.copy_(src)


def cosine_warmup(optimizer, warmup_steps: int, total_steps: int):
    def fn(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * t))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


# ── Train / eval loops ─────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, amp_dtype):
    model.eval()
    preds, targets = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(x)
        preds.append(logits.float().argmax(1).cpu())
        targets.append(y)
    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    return {
        "macro_f1": f1_score(targets, preds, average="macro", zero_division=0),
        "balanced_acc": balanced_accuracy_score(targets, preds),
        "acc": float((preds == targets).mean()),
        "preds": preds,
        "targets": targets,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == "cuda"
    amp_dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if use_amp else None
    print(f"Device: {device}, AMP: {amp_dtype}")

    # Data
    train_ds, val_ds, class_names, image_col, label_col = load_splits(args)
    num_classes = len(class_names)
    train_labels = np.asarray(train_ds[label_col])
    counts = np.bincount(train_labels, minlength=num_classes)
    print(f"{len(train_ds)} train / {len(val_ds)} val patches, {num_classes} classes")
    dist = pd.DataFrame({"class": class_names, "train_count": counts}).sort_values("train_count", ascending=False)
    print(dist.to_string(index=False))
    dist.to_csv(out / "class_distribution.csv", index=False)

    # Model (created first so its pretraining stats drive normalization)
    model = create_model(args.model, num_classes, args.img_size, args.drop_path).to(device)
    data_cfg = resolve_model_data_config(model)
    mean, std = data_cfg["mean"], data_cfg["std"]

    train_tf, eval_tf = build_transforms(args.img_size, mean, std)
    train_ds = train_ds.with_transform(make_torch_transform(train_tf, image_col, label_col))
    val_ds = val_ds.with_transform(make_torch_transform(eval_tf, image_col, label_col))

    sampler = None
    if args.balance == "sampler":
        class_w = np.power(np.maximum(counts, 1), -args.sampler_power)
        sample_w = class_w[train_labels]
        sampler = WeightedRandomSampler(torch.as_tensor(sample_w, dtype=torch.double),
                                        num_samples=len(train_ds), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              shuffle=sampler is None, num_workers=args.workers,
                              pin_memory=device.type == "cuda", drop_last=True, collate_fn=collate,
                              persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                            num_workers=args.workers, pin_memory=device.type == "cuda",
                            collate_fn=collate, persistent_workers=args.workers > 0)

    loss_weights = None
    if args.balance == "loss":
        w = len(train_labels) / (num_classes * np.maximum(counts, 1))
        loss_weights = torch.tensor(w / w.mean(), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=loss_weights, label_smoothing=args.label_smoothing)

    optimizer = torch.optim.AdamW(param_groups(model, args.lr, args.head_lr_mult, args.weight_decay))
    steps_per_epoch = max(1, len(train_loader) // args.accum)
    scheduler = cosine_warmup(optimizer,
                              warmup_steps=int(args.warmup_epochs * steps_per_epoch),
                              total_steps=args.epochs * steps_per_epoch)
    scaler = torch.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)
    ema = ModelEma(model, args.ema_decay) if args.ema_decay > 0 else None

    ckpt_common = {
        "model_name": args.model,
        "img_size": args.img_size,
        "classes": class_names,
        "mean": mean,
        "std": std,
        "dataset": args.dataset,
    }

    best_f1, best_epoch = -1.0, -1
    for epoch in range(args.epochs):
        set_backbone_frozen(model, epoch < args.freeze_epochs)
        model.train()
        t0, running = time.time(), 0.0
        optimizer.zero_grad(set_to_none=True)
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                loss = criterion(model(x), y) / args.accum
            scaler.scale(loss).backward()
            if (i + 1) % args.accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                if ema is not None:
                    ema.update(model)
            running += loss.item() * args.accum
            if (i + 1) % 50 == 0:
                print(f"  epoch {epoch + 1} [{i + 1}/{len(train_loader)}] "
                      f"loss {running / (i + 1):.4f} lr {optimizer.param_groups[0]['lr']:.2e}")

        eval_model = ema.module if ema is not None else model
        metrics = evaluate(eval_model, val_loader, device, amp_dtype)
        print(f"Epoch {epoch + 1}/{args.epochs}  loss {running / max(1, len(train_loader)):.4f}  "
              f"val macro-F1 {metrics['macro_f1']:.4f}  balanced-acc {metrics['balanced_acc']:.4f}  "
              f"acc {metrics['acc']:.4f}  ({time.time() - t0:.0f}s)")

        torch.save({**ckpt_common, "state_dict": eval_model.state_dict(), "epoch": epoch,
                    "val_macro_f1": metrics["macro_f1"]}, out / "last.pt")

        if metrics["macro_f1"] > best_f1:
            best_f1, best_epoch = metrics["macro_f1"], epoch
            torch.save({**ckpt_common, "state_dict": eval_model.state_dict(), "epoch": epoch,
                        "val_macro_f1": best_f1}, out / "best.pt")
            report = classification_report(metrics["targets"], metrics["preds"],
                                           labels=list(range(num_classes)),
                                           target_names=class_names, zero_division=0)
            (out / "val_report.txt").write_text(report)
            cm = confusion_matrix(metrics["targets"], metrics["preds"], labels=list(range(num_classes)))
            pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(out / "confusion_matrix.csv")
            print(f"  ↳ new best; saved best.pt + reports")
        elif epoch - best_epoch >= args.patience:
            print(f"Early stopping: no macro-F1 improvement in {args.patience} epochs.")
            break

    (out / "train_config.json").write_text(json.dumps({**vars(args), "best_macro_f1": best_f1,
                                                       "best_epoch": best_epoch, "classes": class_names}, indent=2))
    print(f"Done. Best val macro-F1 {best_f1:.4f} (epoch {best_epoch + 1}). Artifacts in {out}/")

    if args.export_onnx:
        ckpt = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
        m = create_model(args.model, num_classes, args.img_size, 0.0)
        m.load_state_dict(ckpt["state_dict"])
        m.eval()
        dummy = torch.zeros(1, 3, args.img_size, args.img_size)
        torch.onnx.export(m, dummy, out / "best.onnx", input_names=["pixel_values"],
                          output_names=["logits"], dynamic_axes={"pixel_values": {0: "batch"},
                                                                 "logits": {0: "batch"}}, opset_version=17)
        print(f"Exported ONNX to {out / 'best.onnx'}")


if __name__ == "__main__":
    main()
