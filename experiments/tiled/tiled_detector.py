"""Deteccao em tiles ampliados (SAHI) sobre o detector existente.

Diagnostico de 2026-10-15: os filamentos perdidos nao sao fracos, sao CURTOS
(45 px de comprimento contra 78 px dos detectados). Ampliando um recorte por um
fator f, eles chegam ao detector na escala em que ele ja funciona. Nenhum
retreino: envolve o YoloSegDetector e devolve Candidate nas coordenadas da
imagem inteira, entao o refinador e o NMS de mascara a jusante nao mudam.
"""
from typing import List
import cv2, numpy as np
from solar_filaments.detectors.base import Candidate


class TiledDetector:
    def __init__(self, base, tile: int = 1024, factor: int = 2, overlap: float = 0.5,
                 include_full: bool = True, disk_only: bool = True, tile_conf: float = None):
        self.base, self.tile, self.factor = base, tile, factor
        self.tile_conf = tile_conf
        self.stride = max(1, int(tile * (1.0 - overlap)))
        self.include_full, self.disk_only = include_full, disk_only

    def _origins(self, H, W):
        ys = list(range(0, max(1, H - self.tile + 1), self.stride))
        xs = list(range(0, max(1, W - self.tile + 1), self.stride))
        if ys[-1] != H - self.tile: ys.append(H - self.tile)
        if xs[-1] != W - self.tile: xs.append(W - self.tile)
        return [(y, x) for y in ys for x in xs]

    def detect(self, img: np.ndarray) -> List[Candidate]:
        H, W = img.shape
        out = list(self.base.detect(img)) if self.include_full else []
        thr = img.mean() * 0.25 if self.disk_only else -1
        saved_sz, saved_cf = self.base.imgsz, self.base.conf_thres
        self.base.imgsz = self.tile * self.factor
        if self.tile_conf is not None:
            self.base.conf_thres = self.tile_conf
        try:
            for y0, x0 in self._origins(H, W):
                sub = img[y0:y0 + self.tile, x0:x0 + self.tile]
                if self.disk_only and sub.mean() < thr:
                    continue                      # tile fora do disco solar
                up = cv2.resize(sub, (self.tile * self.factor,) * 2, interpolation=cv2.INTER_CUBIC)
                for c in self.base.detect(up):
                    b = c.box_xyxy.astype(np.float32) / self.factor
                    b[0] += x0; b[1] += y0; b[2] += x0; b[3] += y0
                    m = None
                    if c.coarse_mask is not None and c.coarse_mask.any():
                        small = cv2.resize(c.coarse_mask.astype(np.uint8), (self.tile,) * 2,
                                           interpolation=cv2.INTER_NEAREST)
                        m = np.zeros((H, W), bool)
                        m[y0:y0 + self.tile, x0:x0 + self.tile] = small > 0
                        if not m.any(): m = None
                    out.append(Candidate(b, float(c.score), m))
        finally:
            self.base.imgsz, self.base.conf_thres = saved_sz, saved_cf
        return out
