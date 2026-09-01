# Solar Filament Segmentation 2026

YOLO11 (detect) -> crop -> U-Net (EdgeAttUNet) pipeline for the
[Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026)
(Kaggle / IEEE BigData Cup), on the MAGFiLO H-Alpha dataset.

Originally prototyped as a Kaggle notebook (kept for reference in
[`notebooks/`](notebooks/)); the pipeline now lives as an installable Python
package under [`src/solar_filaments/`](src/solar_filaments/) so it can be run
and iterated on locally.

## Approach

1. **Detect** - YOLO11 draws a tight bounding box around every individual filament.
2. **Crop** - each detection is cropped (padded, aspect-preserving letterbox) out of the full-disk image.
3. **Segment** - a compact U-Net with an edge-guided attention bottleneck refines the crop into a precise binary mask.
4. Per-crop masks are stitched back into a full-resolution instance label map and scored with the competition's own Panoptic Quality (PQ) + Dice self-eval logic (see [`src/solar_filaments/eval/self_eval.py`](src/solar_filaments/eval/self_eval.py), ported verbatim from the [organizers' self-evaluation notebook](https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook)).

## Setup

```bash
python3 -m venv .venv --system-site-packages   # inherits an existing CUDA-matched torch, if present
source .venv/bin/activate
pip install -e .
```

Paths are environment-variable overridable (see [`src/solar_filaments/config.py`](src/solar_filaments/config.py)):

- `FILAMENT_DATA_ROOT` - defaults to `/data/marchezi/filament_segmentation_2026/MAGFiLO_1.0_Kaggle_2026`
- `FILAMENT_WORK_DIR` - defaults to `./work` (checkpoints, YOLO runs, split.json, submission.csv)

### Getting the data

```bash
mkdir -p ~/.kaggle && chmod 700 ~/.kaggle
cp ~/Downloads/kaggle.json ~/.kaggle/kaggle.json && chmod 600 ~/.kaggle/kaggle.json
bash scripts/download_data.sh
```

## Usage

```bash
# 1. Image-level train/val split + materialize the YOLO detection dataset
filament prepare

# 2. Train the YOLO11 detector
filament train-yolo --epochs 60 --imgsz 1024
# resume an interrupted run:
filament train-yolo --resume --weights work/runs/yolo11s_filament_detect/weights/last.pt

# 3. Train the U-Net segmenter on GT crops
filament train-unet --epochs 40

# 4. Self-evaluate on the held-out val split (quick sanity check, then the full PQ/Dice/many-to-one report)
filament evaluate --quick 8 --full

# 5. Run inference on the test set and write submission.csv
filament predict
```

## Submission format

CSV with `filament_id` (`{image_id}_{n}`) and `segmentation_rle` (pycocotools
compressed RLE `counts` only, fixed 2048x2048 per image).

## Notes / open ideas

See the bottom of [`notebooks/best-score-magfilo-yolo11s-detect-crop-u-net.ipynb`](notebooks/best-score-magfilo-yolo11s-detect-crop-u-net.ipynb)
for the prioritized punch list (detection recall ceiling, NMS tuning for
fragmentation/over-merging, spine-guided oriented crops, U-Net TTA, etc.).
