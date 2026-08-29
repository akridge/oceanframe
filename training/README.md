# Benthic patch classifier training

Fine-tunes a pretrained vision backbone on the
[NMFS-OSI/noaa-pacific-benthic-cover-t1-all](https://huggingface.co/datasets/NMFS-OSI/noaa-pacific-benthic-cover-t1-all)
annotation patches, then uses the trained model to classify SAM/SAM3
segments and produce area-weighted percent-cover estimates.

## Why not YOLO-cls for this?

YOLO's classification mode works, but it's not the right tool for this job:

- **Patch classification is pure image classification.** YOLO's value is its
  detection head and real-time pipeline; in `-cls` mode you're just using its
  (ImageNet-pretrained, relatively small) backbone with fewer training knobs.
- **DINOv2 ViTs transfer much better to benthic texture.** Tier-1 classes
  (CCA vs. turf vs. macroalgae vs. coral) are fine-grained *texture* problems.
  Self-supervised DINOv2 features consistently beat supervised-ImageNet
  backbones on CoralNet-style benthic patch benchmarks, especially for rare
  classes and cross-region generalization.
- **Imbalance handling matters more than architecture.** Benthic cover data is
  dominated by a few classes (turf/CCA/sand) with long-tail rarities. Plain
  accuracy looks great while rare-class recall is terrible. This trainer
  optimizes and model-selects on **macro-F1** with balanced sampling —
  Ultralytics' classify trainer gives you none of that out of the box.
- **You're feeding SAM3 segments, not running detection.** SAM3 already does
  localization; the classifier only needs to be the best possible labeler of
  cropped regions. A ViT with square-crop + context padding matches the
  training patch distribution.

Default backbone: `vit_base_patch14_reg4_dinov2.lvd142m` (DINOv2 ViT-B with
registers) at 224 px. Alternatives via `--model`:

| Model | When |
|---|---|
| `vit_base_patch14_reg4_dinov2.lvd142m` | Default — best quality, ~16 GB VRAM at bs 64 |
| `vit_small_patch14_reg4_dinov2.lvd142m` | Smaller GPU / faster iteration |
| `convnext_tiny.fcmae_ft_in22k_in1k` | CPU-friendly inference target |

## Setup

```bash
pip install -r training/requirements.txt
# plus the torch build for your CUDA version, see https://pytorch.org/get-started/
```

## Train

```bash
python training/train_patch_classifier.py --output-dir runs/dinov2b
```

Useful flags:

- `--limit 2000 --epochs 2` — quick smoke test before a full run.
- `--balance sampler|loss|none` — imbalance strategy (default: square-root
  oversampling; `--sampler-power 1.0` fully equalizes classes).
- `--freeze-epochs 1` — head-only warmup epoch(s) before full fine-tune.
- `--batch-size 32 --accum 2` — same effective batch on smaller GPUs.
- `--export-onnx` — write `best.onnx` for deployment.

What it does: loads the HF dataset, auto-detects the image/label columns,
makes a stratified 90/10 val split (if the dataset ships train-only), prints
and saves the class distribution, trains with balanced sampling, strong
rotation/flip augmentation (benthic patches have no canonical orientation),
label smoothing, cosine LR with warmup, discriminative LRs (head 10× backbone),
weight EMA, AMP, and early stopping on **val macro-F1**.

Outputs in `--output-dir`:

- `best.pt` / `last.pt` — checkpoint incl. class names + normalization stats
- `val_report.txt` — per-class precision/recall/F1
- `confusion_matrix.csv`, `class_distribution.csv`, `train_config.json`

Read `val_report.txt`, not the headline accuracy — rare-class recall is where
benthic models fail.

## Classify SAM3 segments

Save your SAM3 masks for an image as a stacked array:

```python
np.savez("frame_0001_masks.npz",
         masks=np.stack([m["segmentation"] for m in sam_output]))
```

Then:

```bash
python training/infer_segments.py \
    --checkpoint runs/dinov2b/best.pt \
    --image frame_0001.jpg \
    --masks-npz frame_0001_masks.npz \
    --csv frame_0001_classes.csv
```

Each segment is cropped at its bbox with 25% context padding (matching the
training patch geometry) and classified; output JSON/CSV carries per-segment
top-k labels plus an **area-weighted percent-cover summary** per image.
`--apply-mask` suppresses background pixels inside the crop — try both; bbox
crops usually win because the model was trained on full rectangular patches.

## Tips

- If val macro-F1 plateaus low on rare classes, try `--sampler-power 1.0`
  or `--balance loss`, and check `confusion_matrix.csv` for label pairs the
  model confuses (e.g. turf vs. CCA) — those often reflect annotation
  ambiguity, not model failure.
- For cross-region deployment, hold out a *site or island* as validation
  rather than a random split to measure realistic generalization.
