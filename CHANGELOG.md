# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.3] - 2026-06-12

### Changed
- The default auto-downloaded record-selection dataset is now
  `flatfile_shallow.pickle` (served from the `data-v2` GitHub Release)
  instead of `NGA_W2_v2.pickle`. The NGA-West2 dataset remains usable via
  the `DJURA_METADATA_PATH` environment variable.

### Added
- Documentation page describing the record-selection metadata schema and
  how to build and supply a custom metadata file in place of the bundled
  dataset.

## [0.2.0] - 2026-06-08

### Added
- Documentation example page for the
  [djura-tools](https://github.com/djura-risk-data-engineering/djura-tools)
  companion repository, including the disaggregation processing, hazard
  consistency, and record-selection input scripts.

## [0.1.10] - 2026-06-08

### Removed
- Drop Zenodo DOI integration (`.zenodo.json` and the GitHub Release
  workflow job); the project DOI is now minted directly via DataCite.

## [0.1.6] - 2026-05-29

### Changed
- Update citations with conference details and DOIs

## [0.1.6] - 2026-05-29

### Changed
- Update copyright and add documentation section
  
## [0.1.5] - 2026-05-14

### Changed

- Clarify in the README that djura is **dual-licensed**: AGPL-3.0-or-later
  for academic, research, and other open-source use, with a separate
  commercial license available from Risk - Data - Engineering S.r.l.
  (Italy) for closed-source products, proprietary SaaS, and internal
  commercial deployments incompatible with the AGPL.
- Add a dedicated **Commercial licensing** section to the README with
  contact details (`info@djura.it`) and enumeration of use cases that
  require a commercial license.
- Note explicitly that OpenQuake-derived portions of the package remain
  under AGPL-3.0-or-later in all distributions.

## [0.1.4] - 2026-05-07

### Added

- File-reading APIs in `GCIM`, `FF`, `_VulnerabilityModeller`, and
  `HazardModel` now accept plain `str` paths in addition to `Path`
  objects — the argument is converted to `Path` internally, so callers
  no longer need to wrap strings explicitly.
- Documentation examples section on Read the Docs covering all six
  submodules:
  - **Record selection**: standalone GCIM workflow (`input1`, conditional)
    and OQ PSHA disaggregation-driven workflow with downloadable input files.
  - **Hazard consistency**: full three-step workflow (parse OQ hazard curves,
    load selected record IMs, run consistency check).
  - **Fragility converter**: exact IM conversion with `FF` (OQ disaggregation
    context) and closed-form approximation with `FFApproximate`, including
    Hellinger-distance accuracy metrics.
  - **EDP-IM**: four subpages covering bilinear analytical backbone,
    file-based backbone (parametric / idealized / SPO), masonry infill
    frame, and base-isolated structure.
  - **SLF**: two subpages — PSD inventory with grouping and PFA inventory,
    each with downloadable input files linked to the repository.
  - **Vulnerability modeller**: full MDOF assessment workflow (backbone
    fitting, EDP-IM prediction, demand estimation, SLF assembly, VMMDOF).
- Downloadable file links on all example pages pointing to the
  corresponding test assets in the repository.

### Changed

- `pydantic` minimum version bumped to `>=2.13.4` (was `>=2.13.3`).
- GitHub Actions: `pypa/gh-action-pypi-publish` 1.12.4 → 1.14.0,
  `actions/attest-build-provenance` 1.4.4 → 4.1.0,
  `actions/cache` 4.2.0 → 5.0.5,
  `github/codeql-action` 3.28.18 → 4.35.4.
- Dev dependencies: `pytest` and `pytest-cov` bumped to latest patch
  releases via Dependabot.

### Fixed

- Removed empty `html_static_path` and `templates_path` entries from
  `docs/source/conf.py` that caused Sphinx warnings treated as errors on
  Read the Docs (both `latest` and `stable` versions).

## [0.1.0] - 2026-05-04

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
- **`djura.edp_im`** — third scientific submodule migrated in:
  - `EDPIMModel` / `EDPIMInfillModel` / `EDPIMIsolModel` — XGBoost-based
    ML models predicting EDP-IM relationships for bare-frame, infilled,
    and isolated structural systems.
  - `BackboneModel` — backbone curve generation from predicted EDP-IM
    relationships.
  - `XGBPredict` — low-level XGBoost inference wrapper with Pydantic
    input validation and built-in scalers.
- **`djura.fragility_converter`** — fourth scientific submodule migrated in:
  - `FF` / `IMModel` — converts fragility functions from one intensity
    measure to another using ground motion model-based IM relationships.
  - `FFApproximate` — approximate conversion method for fragility and
    vulnerability models targeting regional risk workflows.
  - Reference: O'Reilly, G. J., Ozsarac, V., & Shahnazaryan, D. (2025).
    *Conversion of seismic fragility and vulnerability models to
    alternative intensity measures for regional risk analysis.*
    **Earthquake Spectra** (Under Review).
- **`djura.slf`** — fifth scientific submodule migrated in:
  - `SLF` — generates storey loss functions via Monte Carlo simulation,
    fragility-based damage-state assignment, cost sampling, and
    regression fitting (Weibull and Papadopoulos methods).
  - Pydantic input schemas for components, fragility, damage states,
    costs, simulation settings, and fitted loss models (`models.py`).
  - Regression helpers (`regression_methods.py`) and aggregation
    utilities (`utilities.py`).

- **`djura.vulnerability_modeller`** — sixth scientific submodule
  migrated in:
  - `_VulnerabilityModeller` — transforms demand distributions,
    defines capacity curves, and computes EDP-loss via SLF integration.
  - `Backbone` / `SPOModel` / `DispForceModel` — SPO-based backbone
    curve generation for bare-frame and infilled systems.
  - `Demands` — demand model for MDOF systems.
  - `EAL` — expected annual loss computation from limit-state ELR/MAFE
    pairs using an analytical loss-curve fit.
  - `VMMDOF` — MDOF vulnerability modelling with SLF-based loss
    disaggregation.
  - `PFAProfile` — peak floor acceleration profile estimation.
  - Fragility fitting utilities (`fragility.py`) and hazard model
    helpers (`hazard_model.py`).

### Changed

- Added `shapely`, `pyyaml`, `pydantic`, `joblib`, and `pwlf` as core
  runtime dependencies (required by vendored GMMs, record-selection I/O,
  and backbone curve fitting). `h5py` and `xgboost` are optional extras
  (`djura[hdf5]` and `djura[xgboost]` respectively).
- Enforced 79-character line limit across all non-vendored Python
  files; long import lines in `gsim/models/__init__.py` wrapped with
  parenthesised multi-line form; invalid escape sequence in
  `vulnerability_modeller/eal.py` fixed.

### Notes

- The `record_selection.gsim` subpackage is adapted from the
  [OpenQuake Engine](https://github.com/gem/oq-engine) (© GEM
  Foundation, AGPL-3.0-or-later); see
  `src/djura/record_selection/gsim/NOTICE.md` for full attribution.

