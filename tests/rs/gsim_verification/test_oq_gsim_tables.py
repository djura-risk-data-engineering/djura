"""
Verification of djura's GMMs against the OpenQuake verification tables.

Each entry of ``oq_gsim_tables.json`` is one ``check`` of OpenQuake's own test
suite: a GMM, its arguments, its tables and its tolerances. The tables are
downloaded on first use from the pinned OpenQuake tag into the djura cache and
compared the way OpenQuake compares them, as the percentage discrepancy of the
median and of each standard deviation. Only the GMMs named are run::

    pytest tests/rs/gsim_verification --gsim BooreEtAl2014,ChiouYoungs2014
"""

import inspect
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from djura.data_loader import _cache_dir
from djura.record_selection.gsim import imt as imt_module
from djura.record_selection.gsim import models
from djura.record_selection.gsim.contexts import Context

MANIFEST = json.loads(
    (Path(__file__).resolve().parent / "oq_gsim_tables.json").read_text())
TAG = MANIFEST["tag"]
DATA_URL = (f"https://raw.githubusercontent.com/gem/oq-engine/{TAG}"
            "/openquake/hazardlib/tests/gsim/data")
CACHE = _cache_dir() / "oq-gsim-tables" / TAG

#: Position of each OpenQuake result type in djura's list of standard
#: deviations, which follows OpenQuake's order: total, inter-, intra-event
STD_INDEX = {"TOTAL_STDDEV": 0, "INTER_EVENT_STDDEV": 1,
             "INTRA_EVENT_STDDEV": 2}
STD_NAMES = {"TOTAL_STDDEV": "Total", "INTER_EVENT_STDDEV": "Inter event",
             "INTRA_EVENT_STDDEV": "Intra event"}
INPUT_PREFIXES = ("rup_", "dist_", "site_")

#: Medians djura reports in other units than OpenQuake, per GMM module and
#: IMT: the factor turning OpenQuake's value into djura's. CB19 CAV is given
#: in g.s like the package's other CAV models, OpenQuake's in m/s
UNIT_FACTORS = {
    "campbell_bozorgnia_2014": {"CAV": 1 / 9.80665},
}

pytestmark = [pytest.mark.slow, pytest.mark.gsim_verification]


def pytest_generate_tests(metafunc):
    """One case per check of the GMMs named with --gsim, none without it"""
    if "entry" not in metafunc.fixturenames:
        return
    option = metafunc.config.getoption("--gsim")
    names = {name.strip() for name in (option or "").split(",")
             if name.strip()}
    unknown = names - {entry["gsim"] for entry in MANIFEST["entries"]}
    if unknown:
        raise pytest.UsageError(
            f"no OpenQuake verification table for {', '.join(sorted(unknown))}"
            "; see the 'gsim' entries of oq_gsim_tables.json")

    if not names:
        skip = pytest.mark.skip(reason="name the GMMs to verify with --gsim")
        metafunc.parametrize("entry", [pytest.param(None, marks=skip)])
        return

    metafunc.parametrize("entry", [
        pytest.param(entry, id=f"{entry['gsim']}"
                               f"{entry['kwargs'] or ''}-"
                               f"{Path(entry['tables'][0]).stem}")
        for entry in MANIFEST["entries"] if entry["gsim"] in names])


def table(name):
    """Path of a cached table, downloaded on first use"""
    path = CACHE / name
    if path.exists():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    # GitHub's raw host intermittently fails the TLS handshake
    for attempt in range(5):
        try:
            with urllib.request.urlopen(f"{DATA_URL}/{name}",
                                        timeout=60) as response:
                path.write_bytes(response.read())
            return path
        except urllib.error.URLError as error:
            reason = error
            time.sleep(2 * (attempt + 1))
    pytest.skip(f"{name} could not be downloaded: {reason}")


def to_imt(column):
    """djura IMT of a table column, None if the column is not an IMT"""
    if column.startswith(INPUT_PREFIXES + ("Unnamed",)) or \
            column in ("result_type", "damping"):
        return None
    try:
        return imt_module.SA(float(column))
    except ValueError:
        pass
    # Period-dependent measures other than SA, e.g. Sa_avg2(0.1)
    match = re.fullmatch(r"(\w+)\(([\d.]+)\)", column)
    if match and callable(getattr(imt_module, match.group(1), None)):
        return getattr(imt_module, match.group(1))(float(match.group(2)))
    return imt_module.from_string(column.upper())


def as_sites(values, num_sites):
    """One value per site, whatever shape a GMM returns"""
    return np.broadcast_to(np.asarray(values, float).reshape(-1), num_sites)


def evaluate(gsim, ctx, imt, num_sites):
    """Median and standard deviations of one context, in OpenQuake's order"""
    if "imts" in inspect.signature(gsim.get_mean_and_stddevs).parameters:
        # OpenQuake's compute form, filling arrays in place
        mean, sig, tau, phi = np.zeros((4, 1, num_sites))
        gsim.get_mean_and_stddevs(ctx, [imt], mean, sig, tau, phi)
        return mean[0], [sig[0], tau[0], phi[0]]

    out = gsim.get_mean_and_stddevs(ctx=ctx, imt=imt)
    mean, stds = out[0], out[1] if len(out) == 2 else out[1:]
    if not isinstance(stds, (list, tuple)):
        stds = [stds]
    return (as_sites(mean, num_sites),
            [as_sites(std, num_sites) for std in stds])


def predict(gsim, params, imt):
    """Median and standard deviations at every site of one rupture

    Sites are evaluated together, or one at a time for the GMMs written for
    a single scenario, which is how djura calls them
    """
    num_sites = len(next(iter(params.values())))
    try:
        return evaluate(gsim, Context(params), imt, num_sites)
    except (ValueError, IndexError):
        if num_sites == 1:
            raise
    rows = [evaluate(gsim, Context({key: value[i:i + 1]
                                    for key, value in params.items()}),
                     imt, 1) for i in range(num_sites)]
    mean = np.concatenate([row[0] for row in rows])
    stds = [np.concatenate([row[1][k] for row in rows])
            for k in range(len(rows[0][1]))]
    return mean, stds


def contexts(frame, gsim):
    """Context parameters per rupture, the sites held as arrays"""
    rup = [col for col in frame.columns if col.startswith("rup_")]
    sites = [col for col in frame.columns
             if col.startswith(("dist_", "site_"))]
    distances = [col for col in sites if col.startswith("dist_")]
    required = getattr(gsim, "REQUIRES_RUPTURE_PARAMETERS", set())

    for _, group in (frame.groupby(rup, sort=False) if rup
                     else [(None, frame)]):
        # djura's GMMs expect one value per site for every parameter
        params = {col.split("_", 1)[1]: group[col].to_numpy()
                  for col in rup + sites}
        # OpenQuake's harness: a missing rake is zero, and rrup falls back on
        # the distance the table provides
        if "rake" in required and "rake" not in params:
            params["rake"] = np.zeros(len(group))
        if "dist_rrup" not in distances and distances:
            params["rrup"] = group[distances[0]].to_numpy()
        yield params, group


def discrepancies(entry):
    """Worst discrepancy per result type, in per cent, with its location"""
    gsim = getattr(models, entry["gsim"])(**entry["kwargs"])
    defined = {str(getattr(std, "value", std)) for std in getattr(
        gsim, "DEFINED_FOR_STANDARD_DEVIATION_TYPES", ())}
    factors = UNIT_FACTORS.get(type(gsim).__module__.rsplit(".", 1)[-1], {})
    worst = {}

    for name in entry["tables"]:
        frame = pd.read_csv(table(name))
        imts = {col: imt for col in frame.columns
                if (imt := to_imt(col)) is not None}
        for params, group in contexts(frame, gsim):
            for result_type, rows in group.groupby("result_type"):
                if result_type != "MEAN" and defined and \
                        STD_NAMES.get(result_type) not in defined:
                    continue
                for col, imt in imts.items():
                    mean, stds = predict(gsim, params, imt)
                    rows_at = group.index.get_indexer(rows.index)
                    if result_type == "MEAN":
                        got = mean[rows_at]
                        if imt.string != "MMI":
                            got = np.exp(got)
                    else:
                        index = STD_INDEX[result_type]
                        if index >= len(stds):
                            continue
                        got = stds[index][rows_at]
                    expected = rows[col].to_numpy(float)
                    if result_type == "MEAN":
                        expected = expected * factors.get(imt.string, 1.0)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        percent = np.abs(got / expected * 100 - 100)
                    if np.all(np.isnan(percent)):
                        continue
                    at = int(np.nanargmax(percent))
                    if percent[at] > worst.get(result_type, (-1,))[0]:
                        worst[result_type] = (
                            float(percent[at]), col, name,
                            float(expected[at]), float(got[at]))
    return worst


def test_matches_openquake(entry):
    """The median and standard deviations within OpenQuake's tolerances"""
    worst = discrepancies(entry)
    assert worst, "no value of the tables was compared"

    failures = []
    for result_type, (percent, col, name, expected, got) in worst.items():
        tolerance = entry["mean_tolerance" if result_type == "MEAN"
                          else "std_tolerance"]
        if percent > tolerance:
            failures.append(
                f"{result_type} {col} in {name}: expected {expected:.6g}, "
                f"got {got:.6g}, {percent:.3g}% > {tolerance}%")
    assert not failures, "\n".join(failures)
