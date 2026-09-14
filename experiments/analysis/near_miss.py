"""Os GT que perdemos estao perto de casar, ou nao estao nem perto?

PQ casa em IoU > 0.5. Se o melhor IoU dos GT nao casados se acumula logo abaixo de 0.5,
o gargalo e a qualidade da mascara (atacavel: melhor refinador levanta SQ e RQ juntos).
Se se acumula perto de zero, e deteccao pura e nenhum refinador ajuda.
"""
import numpy as np, torch
from ultralytics import YOLO
from solar_filaments.config import CropConfig, InferenceConfig, Paths, UNetConfig, YoloConfig
from solar_filaments.data.coco import Dataset, load_image, load_split, merge_annotator_records, polygon_to_mask
from solar_filaments.detectors import YoloSegDetector
from solar_filaments.inference import refine_candidates
from solar_filaments.models.unet import EdgeAttUNet

paths, dev = Paths(), torch.device("cuda")
d = Dataset(paths.train_ann_path)
_, val_ids = load_split(paths.split_path)
files = sorted({d.images_by_id[i]["file_name"] for i in val_ids if i in d.images_by_id})
unet = EdgeAttUNet(use_edge_attention=UNetConfig().use_edge_attention).to(dev)
unet.load_state_dict(torch.load(paths.ckpt_dir / "unet_best.pt", map_location=dev)); unet.eval()
icfg = InferenceConfig(conf_thres=0.20, imgsz=2048, mask_thresh=0.8, mask_nms_thresh=0.15)
ccfg = CropConfig()
yolo = YoloSegDetector(YOLO(str(paths.runs_dir / f"{YoloConfig().name}_seg" / "weights" / "best.pt")),
                       conf_thres=0.20, iou_thres=icfg.iou_thres, imgsz=2048)

best = []
for n, fn in enumerate(files, 1):
    img = load_image(paths.train_img_dir / fn)
    lm, _ = refine_candidates(img, yolo.detect(img), unet, dev, crop_size=ccfg.crop_size,
                              pad_factor=ccfg.pad_factor, min_crop_side=ccfg.min_crop_side,
                              mask_thresh=icfg.mask_thresh, min_instance_area=icfg.min_instance_area,
                              mask_nms_thresh=icfg.mask_nms_thresh, use_tta=True)
    pm = [m for m in (lm == i for i in range(1, int(lm.max()) + 1)) if m.any()]
    for c in merge_annotator_records(d, fn, "union"):
        g = polygon_to_mask(c["segmentation"], img.shape[0], img.shape[1]).astype(bool)
        best.append(max(((g & p).sum() / max((g | p).sum(), 1) for p in pm), default=0.0))
    if n % 10 == 0: print(f"  {n}/{len(files)}", flush=True)

b = np.array(best)
print(f"\n{len(b)} filamentos GT (uniao de anotadores), conf=0.20")
print(f"casados (IoU>0.5): {(b > 0.5).mean():.1%}")
print("\ndistribuicao do MELHOR IoU dos nao casados:")
bins = [(0.0, 0.01), (0.01, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.45), (0.45, 0.5)]
nm = b[b <= 0.5]
for lo, hi in bins:
    k = ((nm >= lo) & (nm < hi)).sum()
    print(f"  {lo:.2f}-{hi:.2f}: {k:4d}  {k/len(nm):5.1%}  {'#' * int(60*k/len(nm))}")
print(f"\nnao casados totais: {len(nm)}")
print(f"  zero sobreposicao (<0.01): {(nm < 0.01).sum():4d} = {(nm < 0.01).mean():.1%}  <- deteccao pura")
print(f"  quase la (0.40-0.50)     : {((nm >= 0.4) & (nm <= 0.5)).sum():4d} = {((nm >= 0.4) & (nm <= 0.5)).mean():.1%}  <- qualidade de mascara")
