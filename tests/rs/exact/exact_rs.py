r"""Build the exact rupture context from an OpenQuake datastore.

This is the *exact* approach to processing OQ-engine disaggregation:
instead of matching approximate magnitude/distance bins, the full
rupture context is read directly from the datastore
(``calc_<id>.hdf5``) produced by the OQ-engine.

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

    python tests/rs/exact/test_exact.py
"""

# flake8: noqa
from pathlib import Path

path = Path(__file__).resolve().parent

# Conditional IM, does not have to match the conditional
# IM used during conditional selection of OQ engine
im_ref = "SA(0.5)"

# Datastore ID, i.e. data/calc_<dstore>.hdf5
dstore = "9"


def main():
    # Provided by https://github.com/djura-risk-data-engineering/djura-tools
    # (requires the OQ-engine, see the module docstring)
    from djura.hazard.dstore import get_context_from_dstore
    from djura.utilities import export_results

    hdf_path = path / "data" / f"calc_{dstore}.hdf5"
    ctx, oq = get_context_from_dstore(hdf_path, im_ref=im_ref)

    # Save the context into a file, i.e. data/ctx.pickle
    export_results(path / "data" / "ctx", ctx, "pickle")

    return ctx, oq


if __name__ == "__main__":
    main()
