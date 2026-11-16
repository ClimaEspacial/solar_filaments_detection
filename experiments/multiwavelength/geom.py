"""Per-frame GONG disk fit + Helioviewer AIA fetch matched to that geometry."""
import json, re, urllib.request, urllib.parse
import numpy as np, cv2

HV = "https://api.helioviewer.org/v2/"

def fit_disk(gray):
    """Least-squares circle fit to the limb, from the Otsu disk mask boundary."""
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, lab, stats, _ = cv2.connectedComponentsWithStats((th > 0).astype(np.uint8), 8)
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (lab == big).astype(np.uint8)
    cnt = max(cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0],
              key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    # algebraic circle fit:  x^2+y^2 + D x + E y + F = 0
    x, y = cnt[:, 0], cnt[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    D, E, F = np.linalg.lstsq(A, -(x**2 + y**2), rcond=None)[0]
    cx, cy = -D / 2, -E / 2
    r = np.sqrt(cx**2 + cy**2 - F)
    return cx, cy, r

def ts_of(name):
    m = re.search(r"(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})", name)
    y, mo, d, hh, mi, ss = map(int, m.groups())
    return f"{y:04d}-{mo:02d}-{d:02d}T{hh:02d}:{mi:02d}:{ss:02d}Z"

def _get(ep, **kw):
    url = HV + ep + "/?" + urllib.parse.urlencode(kw, safe="[],")
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()

def aia_meta(iso, source=13):
    return json.loads(_get("getClosestImage", date=iso, sourceId=source))

def aia_matched(iso, cx, cy, r_px, size, source=13):
    """AIA frame resampled so the solar disk lands on the same centre/radius
    as the GONG frame -- i.e. directly comparable pixel for pixel."""
    m = aia_meta(iso, source)
    rsun_arcsec = m["rsun"] * m["scale"]        # native px * arcsec/px
    scale = rsun_arcsec / r_px                   # arcsec per output pixel
    # offset of disk centre from image centre, in arcsec (y flipped: image y down)
    x0 = (size / 2 - cx) * scale * -1
    y0 = (size / 2 - cy) * scale
    png = _get("takeScreenshot", date=iso, imageScale=scale,
               layers=f"[SDO,AIA,{ {13:304,10:171,14:211,8:94}[source] },1,100]",
               x0=x0, y0=y0, width=size, height=size,
               display="true", watermark="false")
    arr = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_GRAYSCALE)
    return arr, m


def fit_limb(gray, n_rays=360, smooth=9):
    """Robust solar limb: along rays from an approximate centre, take the radius
    of steepest intensity drop, then fit a circle. Immune to the grey annulus
    present in the GONG JPEGs, which Otsu-based masking absorbs into the disk."""
    h, w = gray.shape
    img = cv2.GaussianBlur(gray, (0, 0), 2).astype(np.float32)
    cx, cy = w / 2.0, h / 2.0
    for _ in range(3):                      # iterate: centre -> rays -> centre
        pts = []
        for th in np.linspace(0, 2 * np.pi, n_rays, endpoint=False):
            rr = np.arange(0.55 * min(h, w) / 2, 0.98 * min(h, w) / 2)
            xs, ys = cx + rr * np.cos(th), cy + rr * np.sin(th)
            ok = (xs >= 0) & (xs < w - 1) & (ys >= 0) & (ys < h - 1)
            rr, xs, ys = rr[ok], xs[ok], ys[ok]
            if len(rr) < 20:
                continue
            v = img[ys.astype(int), xs.astype(int)]
            v = np.convolve(v, np.ones(smooth) / smooth, "same")
            d = np.gradient(v)
            k = int(np.argmin(d))           # steepest drop = limb
            if 5 < k < len(rr) - 5:
                pts.append((xs[k], ys[k]))
        p = np.array(pts, float)
        x, y = p[:, 0], p[:, 1]
        A = np.column_stack([x, y, np.ones_like(x)])
        D, E, F = np.linalg.lstsq(A, -(x**2 + y**2), rcond=None)[0]
        cx, cy = -D / 2, -E / 2
        r = np.sqrt(cx**2 + cy**2 - F)
    return cx, cy, r
