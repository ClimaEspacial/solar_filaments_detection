# ThreadNet — Solar Filament Segmentation 2026

Our solution to the [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026)
(Kaggle / IEEE BigData Cup), on the MAGFiLO GONG H-alpha dataset. A YOLO11
detector proposes each filament's bounding box, a compact edge-attention
U-Net refines a mask per crop, and the crops are stitched back into a
full-resolution instance map with a greedy mask-level NMS.

**Best result: PQ = 0.4405** (held-out validation split, official organizer
self-eval logic) — see the [technical report](kaggle_comp_report_fi_segm_Marchezi_2026/main.tex)
for the full methodology, ablations, and discussion, or run through
[`notebooks/pipeline_walkthrough.ipynb`](notebooks/pipeline_walkthrough.ipynb)
for an executable, illustrated walkthrough of the whole pipeline.

Originally prototyped as a Kaggle notebook (kept for reference in
[`notebooks/`](notebooks/)); the pipeline lives as an installable Python
package under [`src/solar_filaments/`](src/solar_filaments/).

![Pipeline overview](docs/figures/pipeline.png)

## Approach

1. **Detect** — YOLO11s draws an axis-aligned box around every filament, on the full-disk image downsampled to 1024x1024.
2. **Crop** — each detection is padded and letterbox-resized (256x256) out of the native-resolution image, preserving the filament's true aspect ratio.
3. **Refine** — a compact U-Net (8.8M params) with an edge-guided attention bottleneck predicts a binary mask per crop; at inference, the prediction is optionally averaged over 4 flips and accepted only if that average doesn't change the single-view mask's topology (`--tta`).
4. **Stitch** — per-crop masks are pasted back into a full-resolution instance label map via greedy, confidence-ordered mask-level NMS (box IoU is a poor proxy for "same object" on long, thin, diagonal filaments), then scored with the competition's own Panoptic Quality (PQ) + Dice self-eval logic, ported verbatim from the [organizers' self-evaluation notebook](https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook) (see [`src/solar_filaments/eval/self_eval.py`](src/solar_filaments/eval/self_eval.py)).

### Explored extensions (validated, most not adopted)

Every extension below is an opt-in CLI flag, independently toggleable and
composable with `--fold`. Only flip-TTA improved on the plain baseline; the
others are kept because they're real, validated results (see the report for
the numbers and discussion of why each one plateaued).

| Extension | Flag | Result |
|---|---|---|
| 5-fold CV + copy-paste augmentation | `prepare --k-folds N --augment-copy-paste N` | ties baseline |
| Oriented bounding boxes (OBB) | `--obb` | better instance quality, lower recall |
| Ensemble (axis-aligned + OBB + classical) | `evaluate --ensemble` | below best individual model |
| Classical (non-learned) detector | `evaluate --classical` | standalone baseline, PQ=0.11 |
| Limb-darkening flattening + CLAHE | `--normalize` | below baseline |
| InstanceNorm U-Net | `--instance-norm` | ties baseline |
| **Flip-4 topology-safe TTA (adopted)** | `--tta` | **beats baseline (0.4381 -> 0.4405)** |

## Setup

```bash
python3 -m venv .venv --system-site-packages   # inherits an existing CUDA-matched torch, if present
source .venv/bin/activate
pip install -e .
```

`requirements.txt` pins the exact versions this solution was developed and
evaluated against, if you'd rather not resolve dependencies from
`pyproject.toml` yourself:

```bash
pip install -r requirements.txt
pip install -e . --no-deps
```

Paths are environment-variable overridable (see [`src/solar_filaments/config.py`](src/solar_filaments/config.py)):

- `FILAMENT_DATA_ROOT` — defaults to `/data/marchezi/filament_segmentation_2026/MAGFiLO_1.0_Kaggle_2026`
- `FILAMENT_WORK_DIR` — defaults to `./work` (checkpoints, YOLO runs, splits, submission.csv). Not checked into git (see `.gitignore`) — regenerate it with the commands below.

### Getting the data

```bash
mkdir -p ~/.kaggle && chmod 700 ~/.kaggle
cp ~/Downloads/kaggle.json ~/.kaggle/kaggle.json && chmod 600 ~/.kaggle/kaggle.json
bash scripts/download_data.sh
```

## Usage

### 1. Prepare the datasets

```bash
filament prepare                                          # single 90/10 split, axis-aligned only
filament prepare --k-folds 5 --augment-copy-paste 300      # 5-fold + synthetic copy-paste augmentation
filament prepare --obb                                     # also build the OBB-label dataset
filament prepare --normalize                                # also build the limb-darkening+CLAHE dataset
```

### 2. Train the detector

```bash
filament train-yolo --epochs 60 --imgsz 1024               # axis-aligned
filament train-yolo --obb --epochs 60                       # oriented boxes (needs `prepare --obb` first)
filament train-yolo --normalize --epochs 60                 # normalized images (needs `prepare --normalize` first)
filament train-yolo --fold 0 --epochs 60                    # a single k-fold fold
# resume an interrupted run:
filament train-yolo --resume --weights work/runs/yolo11s_filament_detect/weights/last.pt
```

### 3. Train the U-Net refiner

```bash
filament train-unet --epochs 40                             # axis-aligned, BatchNorm
filament train-unet --instance-norm --epochs 40              # InstanceNorm instead of BatchNorm
filament train-unet --normalize --epochs 40                  # trained on preprocessed crops
filament train-unet --obb --epochs 40                        # paired with an OBB-trained detector
```

### 4. Self-evaluate on the held-out split

```bash
filament evaluate --quick 8 --full                           # axis-aligned baseline
filament evaluate --tta --mask-nms-thresh 0.15 --full         # best-known config (PQ=0.4405)
filament evaluate --obb --full                                # OBB
filament evaluate --classical --full                          # non-learned baseline, no GPU
filament evaluate --ensemble --full                            # axis-aligned + OBB (+ classical) merge
filament evaluate --fold all --full                            # pool out-of-fold PQ across every k-fold split
```

Every model variant needs its own `--conf-thres` sweep before its PQ is
trustworthy — a threshold tuned for one configuration does not transfer to
another. See the report for the sweeps behind every number in the table above.

### 5. Generate the test-set submission

```bash
filament predict --tta --mask-nms-thresh 0.15    # best-known config
```

Writes `filament_id,segmentation_rle` rows (pycocotools compressed RLE,
fixed 2048x2048 per image) to `work/submission.csv`.

## Repository structure

```
src/solar_filaments/       installable package (CLI, data, models, inference, eval)
notebooks/                 pipeline_walkthrough.ipynb (start here) + the original Kaggle prototype
scripts/                   data download helper
kaggle_comp_report_fi_segm_Marchezi_2026/   4-page technical report (ACM template, main.tex)
docs/figures/              standalone copies of report figures, for this README
requirements.txt           pinned dependency versions
```

## License

See [`LICENSE`](LICENSE). This repository uses the MAGFiLO dataset and
competition materials under the terms of the
[Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026/rules).
