"""Angulos P (posicao) e B0 (latitude heliografica do centro do disco).

Meeus, "Astronomical Algorithms", cap. 29 (Physical Ephemeris of the Sun). Precisamos
deles porque as imagens do GONG nao estao corrigidas: o eixo de rotacao solar aparece
girado de ate +-26 graus ao longo do ano (P) e o centro do disco corresponde a uma
latitude heliografica de ate +-7.25 graus (B0). Sem essas correcoes, a atribuicao de
hemisferio a um filamento perto do equador e essencialmente aleatoria.
"""
import numpy as np

def julian_day(y, m, d, hh=0, mm=0, ss=0):
    if m <= 2:
        y, m = y - 1, m + 12
    a = y // 100
    b = 2 - a + a // 4
    return (int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + b - 1524.5
            + (hh + mm / 60 + ss / 3600) / 24)

def sun_P_B0(jd):
    """Retorna (P, B0) em graus."""
    t = (jd - 2451545.0) / 36525.0
    rad = np.deg2rad
    # longitude media e anomalia media
    L0 = 280.46646 + 36000.76983 * t + 0.0003032 * t * t
    M = 357.52911 + 35999.05029 * t - 0.0001537 * t * t
    # equacao do centro -> longitude verdadeira
    C = ((1.914602 - 0.004817 * t - 0.000014 * t * t) * np.sin(rad(M))
         + (0.019993 - 0.000101 * t) * np.sin(rad(2 * M))
         + 0.000289 * np.sin(rad(3 * M)))
    true_long = L0 + C
    omega = 125.04 - 1934.136 * t
    lam = true_long - 0.00569 - 0.00478 * np.sin(rad(omega))          # longitude aparente
    eps = (23 + 26 / 60 + 21.448 / 3600
           - (46.8150 * t + 0.00059 * t * t - 0.001813 * t ** 3) / 3600
           + 0.00256 * np.cos(rad(omega)))
    I = 7.25                                                          # inclinacao do equador solar
    K = 73.6667 + 1.3958333 * (jd - 2396758.0) / 36525.0              # longitude do no ascendente
    x = np.rad2deg(np.arctan(-np.cos(rad(lam)) * np.tan(rad(eps))))
    y = np.rad2deg(np.arctan(-np.cos(rad(lam - K)) * np.tan(rad(I))))
    P = x + y
    B0 = np.rad2deg(np.arcsin(np.sin(rad(lam - K)) * np.sin(rad(I))))
    return P, B0

def heliographic_lat(dx, dy, r_sun, P_deg, B0_deg):
    """Latitude heliografica de um ponto do disco.

    (dx, dy) em pixels a partir do centro do disco, com dy positivo para CIMA na imagem.
    Desfaz a rotacao P e projeta usando B0. Retorna NaN fora do disco."""
    x, y = dx / r_sun, dy / r_sun
    rho2 = x * x + y * y
    if rho2 >= 1.0:
        return np.nan
    p, b0 = np.deg2rad(P_deg), np.deg2rad(B0_deg)
    xs = x * np.cos(p) + y * np.sin(p)          # eixo de rotacao solar agora vertical
    ys = -x * np.sin(p) + y * np.cos(p)
    zs = np.sqrt(max(0.0, 1.0 - xs * xs - ys * ys))
    return np.rad2deg(np.arcsin(ys * np.cos(b0) + zs * np.sin(b0)))
