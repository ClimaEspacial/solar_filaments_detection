"""Materialize the YOLO detection dataset (images symlinked, single-class boxes)."""
import os
from pathlib import Path

from solar_filaments.data.coco import Dataset


def build_yolo_dataset(dataset: Dataset, train_img_dir: Path, yolo_dir: Path,
                        train_ids: set, val_ids: set) -> Path:
    """Symlinks images and writes YOLO-format label .txt files for train/val,
    plus a data.yaml. Returns the path to data.yaml."""
    for split, ids in [("train", train_ids), ("val", val_ids)]:
        (yolo_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (yolo_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img_id in ids:
            info = dataset.images_by_id[img_id]
            src = train_img_dir / info["file_name"]
            dst_img = yolo_dir / "images" / split / info["file_name"]
            if not dst_img.exists():
                os.symlink(src, dst_img)

            w, h = info["width"], info["height"]
            lines = []
            for ann in dataset.anns_by_image[img_id]:
                x, y, bw, bh = ann["bbox"]
                cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
                nw, nh = bw / w, bh / h
                lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            label_path = yolo_dir / "labels" / split / (Path(info["file_name"]).stem + ".txt")
            label_path.write_text("\n".join(lines))

    data_yaml = f"""path: {yolo_dir}
train: images/train
val: images/val
names:
  0: filament
"""
    yaml_path = yolo_dir / "data.yaml"
    yaml_path.write_text(data_yaml)
    return yaml_path
