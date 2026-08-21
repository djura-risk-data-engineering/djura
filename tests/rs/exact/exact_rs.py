r"""Exact conditional record selection from an OpenQuake datastore.

The script has two steps:

1. ``build_context()`` -- the *exact* approach to processing OQ-engine
   disaggregation: instead of matching approximate magnitude/distance
   bins, the full rupture context is read directly from the datastore
   (``calc_<id>.hdf5``) produced by the OQ-engine and pickled to
   ``data/ctx.pickle``.
2. ``run_selection(im_ref)`` -- GCIM conditional record selection driven
   by that context, looped over every conditioning IM and every PoE, and
   fanned out over a process pool.

Assumptions for this example
----------------------------
* An OpenQuake PSHA calculation was run with the inputs in ``assets/``,
  that is, ``assets/job.ini`` together with ``assets/inputs/``::

      oq engine --run assets/job.ini

* The resulting datastore was copied here as ``data/calc_9.hdf5``.
  OQ-engine writes datastores into its own data directory, typically:

      C:\Users\<your-username>\oqdata\calc_<id>.hdf5
      C:\Users\<your-username>\Documents\oqdata\calc_<id>.hdf5
      ~/oqdata/calc_<id>.hdf5

* Running this script writes ``data/ctx.pickle``, the pickled context
  consumed by ``GCIM(..., dis_oq=ctx)``.

Requirements
------------
``get_context_from_dstore`` is NOT part of the ``djura`` package. It
lives in the companion repository
https://github.com/djura-risk-data-engineering/djura-tools
and it requires the OQ-engine to be installed, since it reads the
datastore through the OpenQuake API.

Installation steps::

    # 1. install the OQ-engine (see https://github.com/gem/oq-engine
    #    for platform-specific installers)
    pip install openquake.engine

    # 2. get the companion repository with the datastore helpers
    git clone https://github.com/djura-risk-data-engineering/djura-tools
    cd djura-tools

    # 3. install djura itself
    pip install djura

Then run this file as a script (it is intentionally not collected as a
test, as it depends on the optional OQ-engine installation)::

    python tests/rs/exact/exact_rs.py
"""

# flake8: noqa
import gc
import multiprocessing as mp
import pickle
import re
from configparser import RawConfigParser
from pathlib import Path

from djura.record_selection.gcim import GCIM
from djura.utilities import export_results

path = Path(__file__).resolve().parent

# Datastore ID, i.e. data/calc_<dstore>.hdf5
dstore = "9"

# Pickled context produced by build_context()
ctx_pickle = path / "data" / "ctx.pickle"


def read_job_ini(job_ini=path / "assets" / "job.ini"):
    """Read the conditional IMs and the PoEs out of the OQ job file.

    Returns the keys of ``intensity_measure_types_and_levels`` (in file
    order) and the ``poes`` of the ``[output]`` section, so the selection
    below stays in sync with the PSHA run that produced the datastore.
    """
    parser = RawConfigParser()
    parser.read(job_ini, encoding="utf-8")

    imtls = parser["calculation"]["intensity_measure_types_and_levels"]
    # Keys of the dict literal, i.e. '"PGA": logscale(...)' -> PGA
    im_refs = re.findall(r"""["']([^"']+)["']\s*:""", imtls)

    poes = [float(poe)
            for poe in parser["output"]["poes"].replace(",", " ").split()]

    return im_refs, poes


# Conditional IMs to select records for, and the probabilities of
# exceedance, i.e. the intensity levels records are selected at. Both are
# taken from assets/job.ini, so every IM the PSHA run produced a hazard
# curve for is covered at every hazard level of the datastore. An im_ref
# does not have to match the conditional IM used when building the
# context; trim the list to the IMs of interest to keep the run short.
im_refs, poes = read_job_ini()

# Selection settings (number of records, seed, scaling limits, ...).
# The hazard-related entries of this file (gmms, site-parameters,
# ruptures, imi, im-star) are overridden by the datastore context.
defaults = path / "assets" / "default_data.json"


def build_context(im_ref="SA(0.5)"):
    """Step 1: read the exact rupture context out of the datastore."""
    # Provided by https://github.com/djura-risk-data-engineering/djura-tools
    # (requires the OQ-engine, see the module docstring)
    from djura.hazard.dstore import get_context_from_dstore

    hdf_path = path / "data" / f"calc_{dstore}.hdf5"
    ctx, oq = get_context_from_dstore(hdf_path, im_ref=im_ref)

    # Save the context into a file, i.e. data/ctx.pickle
    export_results(ctx_pickle.with_suffix(""), ctx, "pickle")

    return ctx, oq


def run_selection(im_ref):
    """Step 2: select records conditioned on `im_ref`, one PoE at a time.

    The pickled context carries every rupture, GMM logic-tree leaf and
    site parameter of the PSHA run, so `im_ref` is the only thing left to
    switch: `oq_data['im_ref']` tells GCIM which IM to condition on, and
    `poe_for_selection` picks the intensity level off the hazard curve.

    Two files are written per PoE under ``data/records/<im_ref>/``:
    ``target_<poe>.json`` (the GCIM target distribution) and
    ``records_<poe>.json`` (the selected and scaled records).
    """
    records_path = path / "data" / "records" / im_ref
    records_path.mkdir(parents=True, exist_ok=True)

    with open(ctx_pickle, "rb") as f:
        oq_data = pickle.load(f)
    oq_data["im_ref"] = im_ref

    for poe in poes:
        gcim = GCIM(defaults, conditional=True,
                    dis_oq=oq_data, poe_for_selection=poe)

        print(f"Created GCIM object with id {id(gcim)} for poe: {poe}")
        print(f"Conditional IM for poe: {poe}", gcim.data["im-star"])

        target = gcim.create()
        export_results(records_path / f"target_{poe}", target, "json")

        records = gcim.select()
        export_results(records_path / f"records_{poe}", records, "json")

        # Each GCIM instance holds on to the record metadata, so drop it
        # before moving to the next PoE to keep the memory flat
        del gcim
        gc.collect()


if __name__ == "__main__":
    # Uncomment to (re)generate data/ctx.pickle from the datastore
    build_context()

    # One process per conditional IM, 5 at a time. Lower the pool size if
    # the record metadata does not fit in memory that many times over.
    with mp.Pool(5) as pool:
        outs = pool.imap(run_selection, tuple(im_refs))
        for _ in outs:
            print("Success")
