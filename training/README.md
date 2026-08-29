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
- **DINO-family ViTs transfer much better to benthic texture.** Tier-1 classes
  (CCA vs. turf vs. macroalgae vs. coral) are fine-grained *texture* problems.
  Self-supervised DINOv2/DINOv3 features consistently beat supervised-ImageNet
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

Default backbone: `vit_base_patch16_dinov3.lvd1689m` (DINOv3 ViT-B) at 224 px.
Alternatives via `--model`:

| Model | When |
|---|---|
| `vit_base_patch16_dinov3.lvd1689m` | Default — best quality, ~16 GB VRAM at bs 64. Gated weights (HF login) |
| `vit_small_patch16_dinov3.lvd1689m` | Smaller GPU / faster iteration. Gated weights |
| `vit_base_patch14_reg4_dinov2.lvd142m` | Ungated fallback — no HF account needed, Apache-2.0 weights, nearly as good |
| `convnext_tiny.fcmae_ft_in22k_in1k` | CPU-friendly inference target |

Licensing note: DINOv3 checkpoints ship under Meta's custom DINOv3 license;
DINOv2 is Apache 2.0. If your deployment needs a permissive license, use the
DINOv2 fallback — everything else in the pipeline is identical.

## Setup

```bash
pip install -r training/requirements.txt
# plus the torch build for your CUDA version, see https://pytorch.org/get-started/
```

For the default DINOv3 backbone (one-time): accept the license on the
[model page](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m),
then authenticate:

```bash
huggingface-cli login
```

If weight download fails with an access error, the trainer prints this fix;
`--model vit_base_patch14_reg4_dinov2.lvd142m` skips the gate entirely.

## Train

```bash
python training/train_patch_classifier.py --output-dir runs/dinov3b
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
    --checkpoint runs/dinov3b/best.pt \
    --image frame_0001.jpg \
    --masks-npz frame_0001_masks.npz \
    --csv frame_0001_classes.csv
```

Two inference modes (`--mode`):

- **`bbox`** (default): one padded bounding-box crop per segment. Fine for
  small, homogeneous segments from exhaustive "segment everything" runs.
- **`points`**: samples ~8+ fixed-size patches (`--patch-px`) at points inside
  each mask and averages the predictions. **Use this for prompted SAM3**
  (e.g. text prompt "coral"), where one segment can be an entire colony:
  fixed-size point patches match how the training data was made (patches
  around point annotations), avoid the scale mismatch of resizing a whole
  colony down to 224 px, and the reported `point_mixture` exposes
  heterogeneous segments (live coral vs. turf-covered skeleton vs. CCA rim).

Output JSON/CSV carries per-segment top-k labels plus two cover summaries:
`percent_cover_of_image` (denominator = full image, with an `_unsegmented`
remainder) and `percent_cover_by_segmented_area`. With concept-prompted masks
("coral" only), only the image-area numbers are meaningful whole-image cover —
prompt each concept you care about, or run exhaustive segmentation, if you
need total benthic cover.

### Prompted-SAM3 ("coral" text prompt) workflow notes

The classifier doubles as a **verifier** of SAM3's proposals: SAM3's
open-vocabulary understanding of benthic taxa is loose, so some "coral"
segments will really be CCA, macroalgae, or dead skeleton — the per-segment
labels let you filter those false positives, and `point_mixture` tells you
when a colony-sized segment is only partly live coral. No dense per-pixel
model is needed for this; point sampling inside the mask approximates a dense
read-out at whatever density you choose (`--points-per-segment`).

## Tips

- If val macro-F1 plateaus low on rare classes, try `--sampler-power 1.0`
  or `--balance loss`, and check `confusion_matrix.csv` for label pairs the
  model confuses (e.g. turf vs. CCA) — those often reflect annotation
  ambiguity, not model failure.
- For cross-region deployment, hold out a *site or island* as validation
  rather than a random split to measure realistic generalization.
