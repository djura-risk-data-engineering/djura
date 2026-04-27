# Third-Party Attribution — `gsim`

This directory contains code adapted from the **OpenQuake Engine**
(<https://github.com/gem/oq-engine>), Copyright © GEM Foundation,
licensed under the **GNU Affero General Public License v3.0 or later**
(AGPL-3.0-or-later).

The `djura` package is also licensed under AGPL-3.0-or-later, so the
combined work is distributed under the same terms. A copy of the AGPL-3.0
license is included in the repository root as `LICENSE`.

## Scope of adaptation

The `djura.record_selection.gsim` subpackage vendors a subset of the
`openquake.hazardlib.gsim` machinery — specifically:

- A trimmed `GMPE` base class (`base.py`)
- Coefficient table handling (`coeffs_table.py`)
- Intensity-measure types (`imt.py`)
- Rupture / site / distance contexts (`contexts.py`)
- Constants and utility helpers (`const.py`, `utils.py`)
- A curated set of ground-motion model implementations under `models/`

The adaptations were made to remove heavy runtime dependencies on the
full OpenQuake engine while preserving the scientific behavior of the
underlying models.

## Upstream reference

- Upstream project: `gem/oq-engine`
- Adapted from version: **3.23.0**

## Modifications

Where adapted code has been modified, the modifications are noted in
the affected file's header. Modifications include (non-exhaustive):

- Removal of OpenQuake-internal imports and dispatch machinery
- Simplification of the `GMPE` base class
- Local re-implementations of context / IMT plumbing

## Reporting

If you redistribute `djura` or run it as a network service, you must
comply with the AGPL-3.0 — including making the corresponding source
code available to your users. See `LICENSE` for the full terms.
