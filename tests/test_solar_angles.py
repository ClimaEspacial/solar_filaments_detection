"""The solar position angle P and the heliographic latitude of disk centre B0.

These underpin the chirality result: GONG frames are not rotated to solar north, so without
removing P and B0 the hemisphere of a filament near the equator is essentially arbitrary.
Reference values are the standard solar ephemeris.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "figures"))
from solar_angles import heliographic_lat, julian_day, sun_P_B0  # noqa: E402


def test_b0_reaches_its_extremes_on_the_right_dates():
    """B0 peaks at +7.25 deg around 6 September and bottoms at -7.25 deg around 6 March."""
    _, b_sep = sun_P_B0(julian_day(2014, 9, 6, 12))
    _, b_mar = sun_P_B0(julian_day(2014, 3, 6, 12))
    assert b_sep == pytest.approx(7.25, abs=0.05)
    assert b_mar == pytest.approx(-7.25, abs=0.05)


def test_p_crosses_zero_in_early_january_and_early_july():
    for y, m, d in ((2014, 1, 5), (2014, 7, 7)):
        p, _ = sun_P_B0(julian_day(y, m, d, 12))
        assert abs(p) < 0.5, f"{y}-{m:02d}-{d:02d}: P = {p:.2f}"


def test_p_stays_within_its_physical_range():
    """P is bounded by roughly +-26 deg over the year."""
    vals = [sun_P_B0(julian_day(2014, m, d, 12))[0] for m in range(1, 13) for d in (1, 15)]
    assert max(vals) < 27 and min(vals) > -27
    assert max(vals) > 20 and min(vals) < -20, "the annual swing should be nearly full range"


def test_disk_centre_maps_to_b0_and_off_disk_is_undefined():
    import numpy as np
    p, b0 = sun_P_B0(julian_day(2014, 6, 9, 20))
    assert heliographic_lat(0.0, 0.0, 900.0, p, b0) == pytest.approx(b0, abs=1e-6)
    assert np.isnan(heliographic_lat(1000.0, 0.0, 900.0, p, b0))


def test_removing_p_moves_a_point_across_the_equator_when_it_should():
    """The whole point of the correction: at large P a point just above the image centre
    can belong to the southern hemisphere."""
    p, b0 = sun_P_B0(julian_day(2014, 3, 6, 12))     # P near -22.8 deg
    assert p < -20
    lat_corrected = heliographic_lat(600.0, 20.0, 900.0, p, b0)
    lat_naive = heliographic_lat(600.0, 20.0, 900.0, 0.0, 0.0)
    assert abs(lat_corrected - lat_naive) > 5.0
