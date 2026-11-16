"""PQ com e sem deteccao em tiles, mesmos pesos e mesma validacao."""
import sys, time, torch, numpy as np
sys.path.insert(0, "experiments/tiled")
from ultralytics import YOLO
from tiled_detector import TiledDetector
from solar_filaments.config import CropConfig, InferenceConfig, Paths, UNetConfig
from solar_filaments.data.coco import Dataset, load_split
from solar_filaments.detectors.yolo import YoloSegDetector
from solar_filaments.inference import predict_instances_from_candidates
from solar_filaments.models.unet import EdgeAttUNet
from solar_filaments.eval.self_eval import build_val_gt_pred_dataframes, get_overlap_df, \
    get_pq_score, summarize_overlap

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
paths, dev = Paths(), torch.device("cuda")
d = Dataset(paths.train_ann_path); _, val_ids = load_split(paths.split_path)
val_ids = [i for i in val_ids if i in d.images_by_id]
stems, keep = set(), []
for i in val_ids:                                   # N imagens fisicas distintas
    s = d.images_by_id[i]["file_name"]
    if s not in stems and len(stems) >= N: continue
    stems.add(s); keep.append(i)
print(f"{len(stems)} imagens / {len(keep)} registros de anotador")

unet = EdgeAttUNet(use_edge_attention=UNetConfig().use_edge_attention).to(dev)
unet.load_state_dict(torch.load("weights/refiner_unet_edgeatt.pt", map_location=dev)); unet.eval()
icfg = InferenceConfig(conf_thres=0.25, imgsz=2048, mask_thresh=0.5, mask_nms_thresh=0.15)
ccfg = CropConfig()
base = YoloSegDetector(YOLO("weights/detector_yolo11seg_2048.pt"),
                       conf_thres=0.25, iou_thres=icfg.iou_thres, imgsz=2048)

def run(det, tag):
    def pf(p):
        return predict_instances_from_candidates(
            p, det, unet, dev, preprocess=False, refine=True,
            crop_size=ccfg.crop_size, pad_factor=ccfg.pad_factor, min_crop_side=ccfg.min_crop_side,
            mask_thresh=icfg.mask_thresh, min_instance_area=icfg.min_instance_area,
            mask_nms_thresh=icfg.mask_nms_thresh, mask_conf_thres=icfg.mask_conf_thres,
            use_tta=True, fallback_coarse=False)
    t = time.time()
    gt, pr = build_val_gt_pred_dataframes(d, paths.train_img_dir, keep, None, None, dev,
                                          ccfg, icfg, predict_fn=pf)
    ov = get_overlap_df(gt, pr); pq = get_pq_score(ov); s = summarize_overlap(ov)
    print(f"\n{tag}")
    print(f"  PQ  = {pq:.4f}")
    print(f"  SQ={s['SQ']:.4f}  RQ={s['RQ']:.4f}  TP={s['TP']}  FP={s['FP']}  FN={s['FN']}"
          f"  GT sem proposta={s['gt_miss_rate']:.1%}")
    print(f"  preds={len(pr)}  tempo={time.time()-t:.0f}s")
    return pq

pq_b = run(base, "BASELINE (quadro inteiro 2048)")
for tc in (0.25, 0.45, 0.60, 0.75):
    pq = run(TiledDetector(base, tile=1024, factor=2, overlap=0.5, tile_conf=tc),
             f"TILES 2x, conf_tile={tc}")
    print(f"  ==> delta PQ = {pq - pq_b:+.4f}")
