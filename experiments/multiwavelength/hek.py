"""Projeta poligonos de filamento do HEK/AAFDCC em pixels GONG.

Testa as duas hipoteses de orientacao (norte solar para cima vs. rotacao pelo
angulo P) e reporta qual concorda com o ground truth -- determinacao empirica.
"""
import json, re, urllib.request, urllib.parse, datetime as dt
import numpy as np, cv2

HEK = "https://www.lmsal.com/hek/her"

def hek_filaments(iso, window_min=120):
    t = dt.datetime.strptime(iso, "%Y-%m-%dT%H:%M:%SZ")
    a = (t - dt.timedelta(minutes=window_min)).strftime("%Y-%m-%dT%H:%M:%S")
    b = (t + dt.timedelta(minutes=window_min)).strftime("%Y-%m-%dT%H:%M:%S")
    q = dict(cosec=2, cmd="search", type="column", event_type="FI",
             event_starttime=a, event_endtime=b, event_coordsys="helioprojective",
             x1=-1200, x2=1200, y1=-1200, y2=1200, result_limit=500,
             return_="frm_name,event_starttime,hpc_boundcc")
    url = HEK + "?" + urllib.parse.urlencode(q).replace("return_=", "return=")
    with urllib.request.urlopen(url, timeout=120) as r:
        res = json.load(r).get("result", [])
    out = []
    for e in res:
        bc = e.get("hpc_boundcc") or ""
        m = re.findall(r"(-?[\d.]+)\s+(-?[\d.]+)", bc)
        if len(m) < 4: continue
        out.append((np.array(m, float),
                    dt.datetime.strptime(e["event_starttime"], "%Y-%m-%dT%H:%M:%S")))
    return out

def to_px(arc, cx, cy, r_px, rsun_arcsec, P_deg=0.0):
    s = r_px / rsun_arcsec
    x, y = arc[:, 0] * s, arc[:, 1] * s
    if P_deg:
        a = np.deg2rad(P_deg)
        x, y = x * np.cos(a) - y * np.sin(a), x * np.sin(a) + y * np.cos(a)
    return np.column_stack([cx + x, cy - y])

def rasterize(polys, shape=(2048, 2048)):
    m = np.zeros(shape, np.uint8)
    for p in polys:
        cv2.fillPoly(m, [p.astype(np.int32)], 1)
    return m.astype(bool)
