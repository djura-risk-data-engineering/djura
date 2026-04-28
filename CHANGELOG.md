# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial package skeleton with submodules:
  `djura.record_selection`, `djura.hazard_consistency`,
  `djura.vulnerability_modeller`, `djura.slf`.
- `djura.cite()` helper and per-submodule `cite()` functions.
- `CITATION.cff` with per-submodule references.
- AGPL-3.0-or-later license, README, Contributor License Agreement.
- GitHub Actions CI matrix (Python 3.10–3.13 on Ubuntu; Python 3.12 on macOS and Windows).
- **`djura.record_selection`** — first scientific submodule migrated in:
  - GCIM (Generalized Conditional Intensity Measure) record selection
    workflow with conditional and unconditional variants
    (`gcim`, `_gcim`, `_gcim_conditional`, `_gcim_unconditional`,
    `_gcim_select`, `_filter`).
  - Ground motion model (GMM) tooling: epsilon calculation, intensity
    measure handling, NGA-West2 database support, ground-motion-to-
    response-spectrum utilities (`gmm_tools`, `intensity_measure`,
    `nga_west2`, `gm_to_rs`).
  - Correlation models for spectral and non-spectral IMs
    (`correlations`, `correlation_models`) including ANN-based models.
  - Vendored subset of OpenQuake's `hazardlib.gsim` machinery under
    `record_selection.gsim` — base GMPE class, IMTs, contexts,
    coefficient tables, and 45+ ground motion model implementations
    (Abrahamson, Boore, Campbell-Bozorgnia, Chiou-Youngs, Akkar,
    Bindi, Kotha, NGA-East, Aristeidou, etc.) covering active shallow
    crustal, subduction interface, and subduction slab regimes.
  - Bundled data assets (correlation tables, GMM neural-network
    weights, TBEC2018 parameter table) shipped with the wheel.
  - Helpers, metrics, plotting, data readers, and Pydantic input
    models for GMM validation.
- **`djura.hazard_consistency`** — second scientific submodule migrated in:
  - `HazardModel` — reads hazard curves, fits power-law hazard models,
    and derives return periods, PoEs, MAFEs, and IM levels for given
    limit states.
  - `HazardFit` — hazard curve fitting and mean annual frequency of
    exceedance (MAFE) calculations, including SAC/FEMA and
    least-squares approaches.
  - `HazardConsistency` — checks consistency between selected ground
    motion records and target hazard curves.
  - `hazard.Europe` — fetches European seismic hazard data from the
    EFEHR web service (SHARE model).
  - Pydantic input schemas for hazard model validation (`models.py`).

### Changed

- Added `h5py`, `shapely`, `pyyaml`, `pydantic`, and `requests` as
  core runtime dependencies (required by vendored GMMs, record-selection
  I/O, and hazard web-service queries).

### Notes

- The `record_selection.gsim` subpackage is adapted from the
  [OpenQuake Engine](https://github.com/gem/oq-engine) (© GEM
  Foundation, AGPL-3.0-or-later); see
  `src/djura/record_selection/gsim/NOTICE.md` for full attribution.

## [0.1.0] - TBD

- First scaffolded release. No scientific code migrated yet.
