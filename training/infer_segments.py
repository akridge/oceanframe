#!/usr/bin/env python3
"""
Classify SAM/SAM3 segments with a checkpoint trained by train_patch_classifier.py.

Takes a source image plus segment masks and assigns each segment a benthic
class. Masks can come from:
  --masks-dir DIR      binary mask images (PNG etc.), one segment per file
  --masks-npz FILE     .npz / .npy with an (N, H, W) boolean/uint8 array
                       (e.g. np.savez("masks.npz", masks=np.stack([m["segmentation"] for m in sam_results])))

Each segment is cropped at its bounding box with context padding — matching the
square annotation patches the classifier was trained on — and optionally
background-suppressed with --apply-mask.

Example:
    python infer_segments.py --checkpoint runs/dinov2b/best.pt \
        --image survey_0001.jpg --masks-npz survey_0001_masks.npz \
        --out survey_0001_classes.json --csv survey_0001_classes.csv
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
    p.add_argument("--context", type=float, default=0.25,
                   help="Bounding-box padding as a fraction of max(box w, h) per side")
    p.add_argument("--apply-mask", action="store_true",
                   help="Gray out pixels outside the segment before classifying. Off by default: "
                        "the model was trained on full rectangular patches")
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

    records, tensors = [], []
    for mask, name in zip(masks, names):
        area = int(mask.sum())
        if area < args.min_area:
            continue
        crop, bbox = crop_segment(img, mask, args.context, args.apply_mask)
        tensors.append(tf(crop))
        records.append({"segment": name, "area_px": area, "bbox_xywh": bbox})
    if not records:
        raise SystemExit("No segments passed --min-area filtering.")

    probs = []
    with torch.no_grad():
        for i in range(0, len(tensors), args.batch_size):
            batch = torch.stack(tensors[i:i + args.batch_size]).to(device)
            probs.append(torch.softmax(model(batch).float(), dim=1).cpu())
    probs = torch.cat(probs).numpy()

    topk = min(args.topk, len(classes))
    for rec, p in zip(records, probs):
        order = np.argsort(p)[::-1][:topk]
        rec["label"] = classes[order[0]]
        rec["confidence"] = round(float(p[order[0]]), 4)
        rec["topk"] = [{"label": classes[j], "prob": round(float(p[j]), 4)} for j in order]

    # Area-weighted cover summary — the number benthic surveys actually report.
    total_area = sum(r["area_px"] for r in records)
    cover = {}
    for r in records:
        cover[r["label"]] = cover.get(r["label"], 0.0) + r["area_px"] / total_area
    cover = {k: round(v, 4) for k, v in sorted(cover.items(), key=lambda kv: -kv[1])}

    out_path = Path(args.out) if args.out else Path(args.image).with_suffix("").with_name(
        Path(args.image).stem + "_classes.json")
    out_path.write_text(json.dumps({
        "image": args.image,
        "checkpoint": str(args.checkpoint),
        "num_segments": len(records),
        "percent_cover_by_area": cover,
        "segments": records,
    }, indent=2))
    print(f"Classified {len(records)} segments → {out_path}")
    print("Area-weighted cover:", json.dumps(cover, indent=2))

    if args.csv:
        pd.DataFrame([{k: v for k, v in r.items() if k != "topk"} for r in records]).to_csv(args.csv, index=False)
        print(f"CSV → {args.csv}")


if __name__ == "__main__":
    main()
