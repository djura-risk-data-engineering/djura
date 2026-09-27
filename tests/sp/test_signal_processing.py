"""Tests for djura.signal_processing on NGA-West2 RSN 179.

The two horizontal components of Imperial Valley 1979 at El Centro Array #4
(140 and 230 degrees) are checked against the values PEER publishes for
them: the peaks in the record headers and the RotD50 spectrum of the
NGA-West2 flatfile.
"""
from pathlib import Path

import numpy as np
import pytest

from djura import signal_processing as sig

ASSETS = Path(__file__).parent / "assets"
FILE_1 = ASSETS / "RSN179_IMPVALL.H_H-E04140.AT2"
FILE_2 = ASSETS / "RSN179_IMPVALL.H_H-E04230.AT2"

PERIODS = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0])

# SA_RotD50 [g] of RSN 179 at PERIODS, from the NGA-West2 flatfile
# (NGA_West2_Flatfile_RotD50_d050_public_version.xlsx)
PUBLISHED_SA_ROTD50 = np.array([
    0.5170316, 0.6554268, 0.8026243, 0.6905098, 0.5357689, 0.2985871,
    0.2112692])

# PGA [g], PGV [cm/s] and PGD [cm], RotD50, from the same flatfile
PUBLISHED_PEAKS_ROTD50 = {"pga": 0.38078, "pgv": 79.11, "pgd": 55.113}

# PGA [g], PGV [cm/s] and PGD [cm] from the headers of the two files
PUBLISHED_PEAKS = {
    FILE_1.name: (0.48431, 39.6246, 25.1238),
    FILE_2.name: (0.37043, 80.3737, 74.2297),
}


@pytest.fixture(scope="module")
def records():
    dt, npts, _, _, acc1 = sig.read_nga(str(FILE_1))
    _, _, _, _, acc2 = sig.read_nga(str(FILE_2))
    return dt, npts, acc1, acc2


@pytest.fixture(scope="module")
def pair(records):
    dt, _, acc1, acc2 = records
    return sig.GroundMotion(acc1, acc2, dt, unit="g", solver="exact")


def test_read_nga(records):
    dt, npts, acc1, acc2 = records

    assert dt == pytest.approx(0.005)
    assert npts == 7818
    assert len(acc1) == len(acc2) == npts


@pytest.mark.parametrize("path", [FILE_1, FILE_2], ids=lambda p: p.name)
def test_peaks_match_header(records, path):
    dt, _, acc1, acc2 = records
    acc = acc1 if path == FILE_1 else acc2
    gm = sig.Component(acc, dt, unit="g")

    pga, pgv, pgd = PUBLISHED_PEAKS[path.name]
    assert gm.pga == pytest.approx(pga, rel=1e-4)
    # The headers imply g = 980.5 cm/s2 where the package uses 981.0, which
    # puts PGV and PGD 0.05 per cent high
    assert gm.pgv == pytest.approx(pgv, rel=1e-3)
    assert gm.pgd == pytest.approx(pgd, rel=1e-3)


def test_rotd50_spectrum_matches_nga_west2(pair):
    sa = pair.rotd("psa", 50, PERIODS)

    np.testing.assert_allclose(sa, PUBLISHED_SA_ROTD50, rtol=1e-4)


@pytest.mark.parametrize("name", ["pga", "pgv", "pgd"])
def test_rotd50_peaks_match_nga_west2(pair, name):
    value = float(np.ravel(pair.rotd(name, 50))[0])

    assert value == pytest.approx(PUBLISHED_PEAKS_ROTD50[name], rel=1e-4)


def test_rotation_at_0_and_90_degrees_gives_the_components(records, pair):
    dt, _, acc1, acc2 = records
    first = sig.Component(acc1, dt, unit="g", solver="exact")
    second = sig.Component(acc2, dt, unit="g", solver="exact")

    np.testing.assert_allclose(
        pair.at_angle("psa", 0.0, PERIODS), first.psa(PERIODS), rtol=1e-6)
    np.testing.assert_allclose(
        pair.at_angle("psa", 90.0, PERIODS), second.psa(PERIODS), rtol=1e-6)


def test_rotd_percentiles_are_ordered(pair):
    sa50, sa100 = pair.rotd("psa", [50, 100], PERIODS)
    sa_1 = pair.at_angle("psa", 0.0, PERIODS)
    sa_2 = pair.at_angle("psa", 90.0, PERIODS)

    assert np.all(sa50 <= sa100)
    # RotD100 is the largest response over all angles, so no smaller than
    # either as-recorded component
    assert np.all(sa100 >= np.maximum(sa_1, sa_2) * (1 - 1e-6))


def test_newmark_close_to_exact(records):
    dt, _, acc1, acc2 = records
    exact = sig.GroundMotion(acc1, acc2, dt, unit="g", solver="exact")
    newmark = sig.GroundMotion(acc1, acc2, dt, unit="g", solver="newmark")

    np.testing.assert_allclose(
        newmark.rotd("psa", 50, PERIODS), exact.rotd("psa", 50, PERIODS),
        rtol=0.025)


@pytest.mark.parametrize("method", ["filtered", "baseline_corrected"])
def test_conditioned_pair_keeps_its_settings(records, method):
    dt, _, acc1, acc2 = records
    angles = np.arange(0.0, 180.0, 5.0)
    gm = sig.GroundMotion(acc1, acc2, dt, vertical=acc1, unit="g",
                          solver="exact", precision=64, angles=angles)
    new = getattr(gm, method)()

    assert new.precision == 64
    assert new.solver == "exact"
    np.testing.assert_array_equal(new.angles, gm.angles)
    assert new.vertical is not None
    assert new.first.npts == gm.first.npts


@pytest.mark.parametrize("method", ["filtered", "baseline_corrected"])
def test_conditioned_component_keeps_its_precision(records, method):
    dt, _, acc1, _ = records
    gm = sig.Component(acc1, dt, unit="g", precision=64)

    assert getattr(gm, method)().precision == 64


def test_rotd_fallback_keeps_the_precision(records, monkeypatch):
    dt, _, acc1, acc2 = records
    pair = sig.GroundMotion(acc1, acc2, dt, unit="g", precision=64,
                            angles=[0.0, 45.0, 90.0])
    built = []
    init = sig.Component.__init__

    def spy(self, *args, **kwargs):
        init(self, *args, **kwargs)
        built.append(self.precision)

    # Arias intensity has no rotated route, so it is evaluated per angle
    monkeypatch.setattr(sig.Component, "__init__", spy)
    pair.rotd("ia", 50)

    assert built and set(built) == {64}


def test_64_bit_spectra_are_solved_at_the_requested_period(records):
    dt, _, acc1, _ = records
    gm = sig.Component(acc1, dt, unit="g", solver="exact", precision=64)
    psa = gm.psa([0.1])
    sd = gm.sd_rel([0.1])

    # psa and sd_rel share one cached solve at exactly 0.1 s
    assert len(gm._cache) == 1
    assert psa.dtype == np.float64
    assert psa[0] == pytest.approx(
        sd[0] * (2 * np.pi / 0.1) ** 2 / sig.G_TO_CM, rel=1e-12)


def test_pair_needs_both_horizontals(records):
    dt, _, acc1, _ = records

    with pytest.raises(ValueError, match="both horizontal"):
        sig.GroundMotion(acc1, None, dt)


def test_64_bit_sa_avg_reuses_the_precomputed_grid(records):
    dt, _, acc1, acc2 = records
    pair = sig.GroundMotion(acc1, acc2, dt, unit="g", precision=64,
                            angles=[0.0, 90.0])
    pair.precompute(sa_avg_periods=[1.0])
    solved = len(pair._cache)
    sa_avg = pair.rotd("sa_avg", 50, [1.0])

    assert sa_avg.dtype == np.float64
    # Served from the precomputed grid, with no further solve
    assert len(pair._cache) == solved


def test_pulse_classifier_gets_the_pair_precision(records, monkeypatch):
    pytest.importorskip("pywt")
    from djura.signal_processing import pulse

    dt, _, acc1, acc2 = records
    pair = sig.GroundMotion(acc1, acc2, dt, unit="g", precision=64)
    seen = []
    init = pulse.PulseClassifier.__init__

    def spy(self, ag1, ag2, dt):
        seen.append((ag1.dtype, ag2.dtype))
        init(self, ag1, ag2, dt)

    monkeypatch.setattr(pulse.PulseClassifier, "__init__", spy)
    pair.pulse

    assert seen == [(np.float64, np.float64)]


def test_64_bit_spectrum_solves_the_unrounded_trace(records):
    dt, _, acc1, _ = records
    acc = np.asarray(acc1, dtype=np.float64)
    periods = np.array([0.02, 0.1, 1.0])
    gm = sig.Component(acc, dt, unit="g", solver="exact", precision=64)

    factors = sig.ims.resampling_factors(dt, periods)
    expected = []
    for period, factor in zip(periods, factors):
        trace, step = sig.ims.resample(acc, dt, int(factor), np.float64)
        u = sig.ims.response(trace, step, [period], 0.05, "exact",
                             precision=64)
        expected.append(np.max(np.abs(u)) * sig.G_TO_CM)

    np.testing.assert_allclose(gm.sd_rel(periods), expected, rtol=1e-12)


def test_64_bit_rotated_absolute_response_matches_the_component(records):
    dt, _, acc1, acc2 = records
    periods = np.array([0.1, 1.0])
    pair = sig.GroundMotion(acc1, acc2, dt, unit="g", solver="exact",
                            precision=64, angles=[0.0, 90.0])

    np.testing.assert_allclose(pair.at_angle("sd_abs", 0.0, periods),
                               pair.first.sd_abs(periods), rtol=1e-10)


def test_64_bit_fourier_and_fiv3_results(records):
    dt, _, acc1, acc2 = records
    pair = sig.GroundMotion(acc1, acc2, dt, unit="g", precision=64,
                            angles=[0.0, 90.0])

    assert pair.first.fas[1].dtype == np.float64
    assert pair.first.pas[1].dtype == np.float64
    assert pair.first.fiv3([1.0]).dtype == np.float64
    assert pair.rotd("fiv3", 50, [1.0]).dtype == np.float64


# The five candidates of RSN 179 from the MATLAB code of Shahi and Baker
# (2014): (column, scale, Tp [s], pulse indicator, is_pulse, angle [deg]).
# MATLAB's columns are 1-based, so these are one lower.
MATLAB_PULSE_CANDIDATES = [
    (1260, 684, 4.788, 20.0610, True, 89.38),
    (827, 767, 5.369, 17.8138, True, 89.87),
    (1655, 540, 3.780, -2.5460, False, -81.10),
    (520, 1291, 9.037, 4.9389, True, 74.36),
    (1876, 794, 5.558, 1.0880, True, 73.11),
]


def test_pulse_period_matches_shahi_and_baker(pair):
    pytest.importorskip("pywt")

    # Listed as 4.79 s in their published classification
    assert pair.tpulse == pytest.approx(4.788, abs=1e-3)


@pytest.mark.parametrize("index", range(5))
def test_pulse_candidates_match_shahi_and_baker(pair, index):
    pytest.importorskip("pywt")
    column, scale, tp, indicator, is_pulse, angle = (
        MATLAB_PULSE_CANDIDATES[index])
    classifier = pair.pulse
    data = classifier.pulse_data[index]

    assert classifier.columns[index] == column
    assert data["pulse_scale"] == scale
    assert data["Tp"] == pytest.approx(tp, abs=1e-3)
    assert data["pulse_indicator"] == pytest.approx(indicator, abs=1e-4)
    assert not data["late"]
    assert bool(data["is_pulse"]) is is_pulse
    assert np.degrees(classifier.rot_angles[index]) == pytest.approx(
        angle, abs=0.01)
