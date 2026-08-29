#!/usr/bin/env python3
"""
Classify SAM/SAM3 segments with a checkpoint trained by train_patch_classifier.py.

Takes a source image plus segment masks and assigns each segment a benthic
class. Masks can come from:
  --masks-dir DIR      binary mask images (PNG etc.), one segment per file
  --masks-npz FILE     .npz / .npy with an (N, H, W) boolean/uint8 array
                       (e.g. np.savez("masks.npz", masks=np.stack([m["segmentation"] for m in sam_results])))

Two ways to turn a segment into classifier inputs (--mode):
  bbox    one crop at the segment's bounding box with context padding.
          Good for small, homogeneous segments.
  points  sample N fixed-size patches (--patch-px) at points inside the mask
          and average the predictions. This mirrors how the training data was
          made (fixed-size patches around annotation points), so it stays on
          the training distribution even for large segments, and it exposes
          mixed segments (e.g. live coral + turf-covered skeleton) via
          per-class mixture fractions. Preferred for prompted-SAM3 workflows
          where one "coral" segment can be a whole colony.

Example:
    python infer_segments.py --checkpoint runs/dinov3b/best.pt \
        --image survey_0001.jpg --masks-npz survey_0001_masks.npz \
        --mode points --csv survey_0001_classes.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

import timm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="best.pt from train_patch_classifier.py")
    p.add_argument("--image", required=True, help="Source image the masks belong to")
    p.add_argument("--masks-dir", help="Directory of per-segment binary mask images")
    p.add_argument("--masks-npz", help=".npz/.npy file with an (N, H, W) mask array")
    p.add_argument("--mode", choices=["bbox", "points"], default="bbox",
                   help="bbox: one padded bounding-box crop per segment. "
                        "points: sample fixed-size patches inside the mask and average — "
                        "use for large/heterogeneous segments (see module docstring)")
    p.add_argument("--context", type=float, default=0.25,
                   help="bbox mode: padding as a fraction of max(box w, h) per side")
    p.add_argument("--apply-mask", action="store_true",
                   help="bbox mode: gray out pixels outside the segment before classifying. Off by "
                        "default: the model was trained on full rectangular patches")
    p.add_argument("--points-per-segment", type=int, default=8,
                   help="points mode: patches sampled per segment (large segments get up to 4x more, "
                        "scaled by area)")
    p.add_argument("--patch-px", type=int, default=224,
                   help="points mode: patch side length in source-image pixels. Match the ground "
                        "footprint of the training annotation patches at your survey altitude")
    p.add_argument("--seed", type=int, default=0, help="points mode: sampling seed")
    p.add_argument("--min-area", type=int, default=64, help="Skip segments smaller than this many pixels")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--out", default=None, help="Output JSON path (default: <image>_classes.json)")
    p.add_argument("--csv", default=None, help="Also write a CSV summary here")
    return p.parse_args()


def load_masks(args, img_hw: tuple[int, int]) -> tuple[list[np.ndarray], list[str]]:
    masks, names = [], []
    if args.masks_npz:
        arr = np.load(args.masks_npz)
        if hasattr(arr, "files"):  # npz — take the first array (commonly "masks")
            key = "masks" if "masks" in arr.files else arr.files[0]
            arr = arr[key]
        if arr.ndim == 2:
            arr = arr[None]
        for i, m in enumerate(arr):
            masks.append(m.astype(bool))
            names.append(f"mask_{i:04d}")
    elif args.masks_dir:
        paths = sorted(p for p in Path(args.masks_dir).iterdir()
                       if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})
        for p in paths:
            m = np.array(Image.open(p).convert("L")) > 127
            masks.append(m)
            names.append(p.stem)
    else:
        raise SystemExit("Provide --masks-dir or --masks-npz")

    for i, m in enumerate(masks):
        if m.shape != img_hw:
            raise SystemExit(f"Mask {names[i]} shape {m.shape} != image shape {img_hw}")
    return masks, names


def sample_point_patches(img: np.ndarray, mask: np.ndarray, n_points: int,
                         patch_px: int, rng: np.random.Generator) -> list[Image.Image]:
    """Fixed-size patches centered on points sampled inside the mask,
    shifted (not shrunk) to stay within image bounds."""
    ys, xs = np.nonzero(mask)
    # Scale point count with segment size: ~1 extra point per patch-area, capped at 4x.
    n = int(np.clip(round(len(ys) / (patch_px * patch_px)), n_points, n_points * 4))
    n = min(n, len(ys))
    idx = rng.choice(len(ys), size=n, replace=False)
    H, W = mask.shape
    half = patch_px // 2
    patches = []
    for cy, cx in zip(ys[idx], xs[idx]):
        y0 = int(np.clip(cy - half, 0, max(0, H - patch_px)))
        x0 = int(np.clip(cx - half, 0, max(0, W - patch_px)))
        patches.append(Image.fromarray(img[y0:y0 + patch_px, x0:x0 + patch_px]))
    return patches


def crop_segment(img: np.ndarray, mask: np.ndarray, context: float, apply_mask: bool) -> tuple[Image.Image, list[int]]:
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    pad = int(round(max(y1 - y0, x1 - x0) * context))
    H, W = mask.shape
    cy0, cy1 = max(0, y0 - pad), min(H, y1 + pad)
    cx0, cx1 = max(0, x0 - pad), min(W, x1 + pad)
    crop = img[cy0:cy1, cx0:cx1].copy()
    if apply_mask:
        # Push background toward mid-gray instead of black to soften the artificial edge.
        bg = ~mask[cy0:cy1, cx0:cx1]
        crop[bg] = (0.35 * crop[bg] + 0.65 * 128).astype(np.uint8)
    return Image.fromarray(crop), [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    classes, img_size = ckpt["classes"], ckpt["img_size"]
    try:
        model = timm.create_model(ckpt["model_name"], pretrained=False,
                                  num_classes=len(classes), img_size=img_size)
    except TypeError:
        model = timm.create_model(ckpt["model_name"], pretrained=False, num_classes=len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)

    tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(ckpt["mean"], ckpt["std"]),
    ])

    img = np.array(Image.open(args.image).convert("RGB"))
    masks, names = load_masks(args, img.shape[:2])

    rng = np.random.default_rng(args.seed)
    records, tensors, slices = [], [], []
    for mask, name in zip(masks, names):
        area = int(mask.sum())
        if area < args.min_area:
            continue
        ys, xs = np.nonzero(mask)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]
        if args.mode == "points":
            crops = sample_point_patches(img, mask, args.points_per_segment, args.patch_px, rng)
        else:
            crop, bbox = crop_segment(img, mask, args.context, args.apply_mask)
            crops = [crop]
        slices.append((len(tensors), len(tensors) + len(crops)))
        tensors.extend(tf(c) for c in crops)
        records.append({"segment": name, "area_px": area, "bbox_xywh": bbox, "num_patches": len(crops)})
    if not records:
        raise SystemExit("No segments passed --min-area filtering.")

    probs = []
    with torch.no_grad():
        for i in range(0, len(tensors), args.batch_size):
            batch = torch.stack(tensors[i:i + args.batch_size]).to(device)
            probs.append(torch.softmax(model(batch).float(), dim=1).cpu())
    probs = torch.cat(probs).numpy()

    topk = min(args.topk, len(classes))
    for rec, (a, b) in zip(records, slices):
        seg_probs = probs[a:b]
        p = seg_probs.mean(0)
        order = np.argsort(p)[::-1][:topk]
        rec["label"] = classes[order[0]]
        rec["confidence"] = round(float(p[order[0]]), 4)
        rec["topk"] = [{"label": classes[j], "prob": round(float(p[j]), 4)} for j in order]
        if args.mode == "points":
            # Per-patch votes reveal mixed segments (e.g. a colony that is part
            # live coral, part turf-covered skeleton).
            votes = seg_probs.argmax(1)
            rec["point_mixture"] = {classes[j]: round(float((votes == j).mean()), 3)
                                    for j in sorted(set(votes.tolist()))}
            rec["point_agreement"] = rec["point_mixture"].get(rec["label"], 0.0)

    # Cover summaries. percent_cover_of_image uses the full image as denominator,
    # so with prompted (e.g. "coral"-only) masks the classes sum to less than 1
    # and the remainder is reported as unsegmented — do NOT read
    # percent_cover_by_segmented_area as whole-image cover in that workflow.
    total_seg_area = sum(r["area_px"] for r in records)
    img_area = img.shape[0] * img.shape[1]
    by_seg, of_img = {}, {}
    for r in records:
        by_seg[r["label"]] = by_seg.get(r["label"], 0.0) + r["area_px"] / total_seg_area
        of_img[r["label"]] = of_img.get(r["label"], 0.0) + r["area_px"] / img_area
    cover = {k: round(v, 4) for k, v in sorted(by_seg.items(), key=lambda kv: -kv[1])}
    cover_img = {k: round(v, 4) for k, v in sorted(of_img.items(), key=lambda kv: -kv[1])}
    cover_img["_unsegmented"] = round(max(0.0, 1.0 - min(1.0, total_seg_area / img_area)), 4)

    out_path = Path(args.out) if args.out else Path(args.image).with_suffix("").with_name(
        Path(args.image).stem + "_classes.json")
    out_path.write_text(json.dumps({
        "image": args.image,
        "checkpoint": str(args.checkpoint),
        "mode": args.mode,
        "num_segments": len(records),
        "percent_cover_by_segmented_area": cover,
        "percent_cover_of_image": cover_img,
        "segments": records,
    }, indent=2))
    print(f"Classified {len(records)} segments ({len(tensors)} patches, mode={args.mode}) → {out_path}")
    print("Cover of image area:", json.dumps(cover_img, indent=2))

    if args.csv:
        pd.DataFrame([{k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in r.items()
                       if k != "topk"} for r in records]).to_csv(args.csv, index=False)
        print(f"CSV → {args.csv}")


if __name__ == "__main__":
    main()
