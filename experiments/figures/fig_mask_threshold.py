"""Figura: o limiar de mascara da U-Net e onde o PQ e maximo.

O ponto nao e o valor otimo em si, e o painel (b): o otimo esta onde a area mediana das
mascaras preditas iguala a do GT rasterizado pela convencao do avaliador. Isso torna o
limiar calibravel sem varredura, a partir de uma estatistica que nao precisa do rotulo
de teste.
"""
import numpy as np, pandas as pd, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO
from solar_filaments.config import CropConfig, InferenceConfig, Paths, UNetConfig, YoloConfig
from solar_filaments.data.coco import Dataset, load_image, load_split, polygon_to_mask
from solar_filaments.detectors import YoloSegDetector
from solar_filaments.eval.self_eval import get_overlap_df, get_pq_score
from solar_filaments.inference import refine_candidates
from solar_filaments.models.refiner import load_refiner
from solar_filaments.rle import encode_rle_counts

paths, dev = Paths(), torch.device("cuda")
d = Dataset(paths.train_ann_path); _, val_ids = load_split(paths.split_path)
unet = load_refiner(paths.ckpt_dir / "unet_best.pt", dev, use_edge_attention=UNetConfig().use_edge_attention)
ccfg = CropConfig()
COLS = ["filament_id", "segmentation_rle"]

# GT uma unica vez (rasterizacao pycocotools, a convencao do avaliador)
gt_rows, gt_areas = [], []
for img_id in val_ids:
    info = d.images_by_id[img_id]
    for k, ann in enumerate(d.anns_by_image.get(img_id, []), 1):
        m = polygon_to_mask(ann["segmentation"], info["height"], info["width"]).astype(bool)
        if m.any():
            gt_rows.append({"filament_id": f"{img_id}_{k}", "segmentation_rle": encode_rle_counts(m)})
            gt_areas.append(int(m.sum()))
gt_df = pd.DataFrame(gt_rows, columns=COLS)
GT_MED = float(np.median(gt_areas))
print(f"GT: {len(gt_df)} registros, area mediana {GT_MED:.0f} px")

files = [(i, d.images_by_id[i]["file_name"]) for i in val_ids]
seen, uniq = set(), []
for i, fn in files:
    if fn not in seen: seen.add(fn); uniq.append(fn)
imgs = {fn: load_image(paths.train_img_dir / fn) for fn in uniq}

res = {}
for conf in (0.15, 0.35):
    det = YoloSegDetector(YOLO(str(paths.runs_dir / f"{YoloConfig().name}_seg" / "weights" / "best.pt")),
                          conf_thres=conf, iou_thres=0.5, imgsz=2048)
    cands = {fn: det.detect(imgs[fn]) for fn in uniq}      # detector roda uma vez por conf
    for mt in (0.5, 0.6, 0.7, 0.8, 0.9):
        rows, areas = [], []
        for fn in uniq:
            lm, _ = refine_candidates(imgs[fn], cands[fn], unet, dev, crop_size=ccfg.crop_size,
                pad_factor=ccfg.pad_factor, min_crop_side=ccfg.min_crop_side, mask_thresh=mt,
                min_instance_area=25, mask_nms_thresh=0.15, use_tta=True)
            stem = fn.rsplit(".", 1)[0]
            for i in range(1, int(lm.max()) + 1):
                m = lm == i
                if m.any():
                    rows.append({"filament_id": f"{stem}_{i}", "segmentation_rle": encode_rle_counts(m)})
                    areas.append(int(m.sum()))
        pq = get_pq_score(get_overlap_df(gt_df, pd.DataFrame(rows, columns=COLS)))
        ratio = float(np.median(areas)) / GT_MED if areas else np.nan
        res[(conf, mt)] = (pq, ratio)
        print(f"  conf={conf} mask_thresh={mt}: PQ={pq:.4f}  area_pred/area_GT={ratio:.3f}", flush=True)

np.save("experiments/data/mask_threshold.npy", np.array(
    [[c, m, *res[(c, m)]] for c in (0.15, 0.35) for m in (0.5, 0.6, 0.7, 0.8, 0.9)]))

plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.linewidth": 0.8,
                     "xtick.direction": "in", "ytick.direction": "in"})
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.75))
mts = [0.5, 0.6, 0.7, 0.8, 0.9]
for conf, col, mk in ((0.15, "#1f4e79", "o"), (0.35, "#c44e52", "s")):
    pq = [res[(conf, m)][0] for m in mts]; rt = [res[(conf, m)][1] for m in mts]
    ax[0].plot(mts, pq, mk + "-", color=col, ms=4.5, lw=1.3, label=f"conf $=$ {conf}")
    ax[1].plot(rt, pq, mk + "-", color=col, ms=4.5, lw=1.3, label=f"conf $=$ {conf}")
ax[0].set_xlabel("U-Net mask threshold"); ax[0].set_ylabel("Panoptic Quality")
ax[0].legend(frameon=False, fontsize=7.5, loc="lower left")
ax[1].axvline(1.0, color="#55a868", ls="--", lw=1.1)
ax[1].text(0.995, 0.4275, "predicted area\n$=$ GT area", fontsize=7,
           color="#55a868", va="bottom", ha="right")
ax[1].annotate("", xy=(1.19, 0.4128), xytext=(0.88, 0.4128),
               arrowprops=dict(arrowstyle="->", lw=0.9, color="0.45"))
ax[1].text(1.035, 0.4136, "fatter masks score higher", fontsize=7, color="0.45", ha="center")
ax[1].set_xlabel("median predicted area / median GT area"); ax[1].set_ylabel("Panoptic Quality")
for a_, l in zip(ax, "ab"):
    a_.text(0.945, 0.95, f"({l})", transform=a_.transAxes, fontsize=9, fontweight="bold", va="top")
fig.tight_layout(pad=0.5, w_pad=1.6)
out = "Marchezi_etal_2026_SolarPhysics_FilamentSegmentation/figures/mask_threshold.pdf"
fig.savefig(out, bbox_inches="tight"); fig.savefig(out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
print("salvo:", out)
