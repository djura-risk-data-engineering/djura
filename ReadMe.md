# djura

[![CI](https://github.com/djura-risk-data-engineering/djura/actions/workflows/ci.yml/badge.svg)](https://github.com/djura-risk-data-engineering/djura/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue.svg)](https://www.python.org/)
[![License: AGPL v3+](https://img.shields.io/badge/license-AGPL--3.0--or--later-blue.svg)](LICENSE)

**djura** is a scientific Python toolkit for earthquake engineering. It bundles
into a single installable package the core algorithms used across the DJURA
research stack — ground motion record selection, hazard-consistent intensity
measure analysis, structural vulnerability modelling, and storey loss function
generation — with no web-server, database, or cloud-storage dependencies.

The package is intended for **research and educational use**, and is released
under the **GNU AGPL-3.0** license so that it composes cleanly with other
copyleft scientific tools (e.g. `openquake.engine`).

## Submodules

| Import path                       | Purpose                                                   |
| --------------------------------- | --------------------------------------------------------- |
| `djura.record_selection`          | GCIM-based ground motion record selection                 |
| `djura.hazard_consistency`        | Hazard-consistent intensity measure analysis              |
| `djura.edp_im`                    | ML-based EDP-IM relationship prediction                   |
| `djura.fragility_converter`       | Fragility/vulnerability model conversion across IMs       |
| `djura.vulnerability_modeller`    | Seismic vulnerability and loss modelling (incl. ML models)|
| `djura.slf`                       | Storey loss function generation                           |

## Installation

```bash
pip install djura
```

Optional extras:

```bash
pip install "djura[plot]"   # adds matplotlib
pip install "djura[hdf5]"   # adds h5py
pip install "djura[all]"    # all of the above
```

## Quickstart

```python
import djura

print(djura.__version__)

# Per-submodule example imports
from djura import record_selection
from djura import hazard_consistency
from djura import edp_im
from djura import vulnerability_modeller
from djura import slf
```

(Per-submodule quickstarts will be added as code is migrated in.)

## Bundled dataset (NGA-West2)

The NGA-West2 metadata pickle (~107 MB uncompressed) is **not** shipped
inside the wheel. It is hosted as a gzip-compressed asset on a GitHub
Release and downloaded automatically the first time it is needed:

```python
from djura.data_loader import load_data, clear_cache

data = load_data()       # downloads on first call, then loads from cache
clear_cache()            # delete the cached file to force a re-download
```

The cache lives at `~/.cache/djura/NGA_W2_v2.pickle`.

### Publishing a new data release (maintainers)

The release is produced by the `release-data` GitHub Actions workflow,
which compresses the pickle and uploads it to a tagged GitHub Release.
Trigger it manually with the GitHub CLI:

```bash
gh workflow run release-data.yml -f version=data-v1
```

After the release is published, update `GITHUB_RELEASE_URL` in
`src/djura/data_loader.py` to point at the new tag.

## How to cite

If you use **djura** in academic or research work, please cite the package
**and** the paper(s) backing the submodule(s) you use.

```python
import djura

# Umbrella package citation
print(djura.cite())

# Per-submodule citation
print(djura.cite("vulnerability_modeller"))

# All citations
print(djura.cite(all=True))
```

| Submodule                       | Reference                                                                                                                                                                       |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `record_selection`              | Shahnazaryan, D., Ozsarac, V., & O'Reilly, G. J. (2025). *DJURA Ground Motion Record Selector: A Software Solution for Earthquake Engineering*. COMPDYN 2025.                   |
| `hazard_consistency`            | Shahnazaryan, D., Ozsarac, V., & O'Reilly, G. J. (2025). *DJURA Ground Motion Record Selector: A Software Solution for Earthquake Engineering*. COMPDYN 2025.                   |
| `edp_im`                        | Shahnazaryan, D., & O'Reilly, G. J. (2024). *Next-generation non-linear and collapse prediction models for short- to long-period systems via machine learning methods*. **Engineering Structures**, 306, 117801. doi:[10.1016/j.engstruct.2024.117801](https://doi.org/10.1016/j.engstruct.2024.117801) |
| `vulnerability_modeller`        | O'Reilly, G. J., & Shahnazaryan, D. (2024). *On the utility of story loss functions for regional seismic vulnerability modeling and risk assessment*. **Earthquake Spectra**, 40(3), 1933–1955. doi:[10.1177/87552930241245940](https://doi.org/10.1177/87552930241245940) |
| `fragility_converter`           | O'Reilly, G. J., Ozsarac, V., & Shahnazaryan, D. (2025). *Conversion of seismic fragility and vulnerability models to alternative intensity measures for regional risk analysis*. **Earthquake Spectra** (Under Review). |
| `slf`                           | Shahnazaryan, D., Ozsarac, V., & O'Reilly, G. J. (2025). *The Role of Story Loss Functions in Regional Seismic Vulnerability Modelling and Risk Assessment*.                    |

A `CITATION.cff` file is provided so that GitHub renders a "Cite this repository" button automatically.

## Contributing

Contributions are welcome. By submitting a Contribution, you agree to the
terms of the [Contributor License Agreement](CLA.md), which (among other
things) allows the maintainer to relicense the project — for example, to
offer a separate commercial license alongside AGPL-3.0.

## License

Copyright © 2025–2026 Djura | Risk - Data - Engineering S.r.l.

Licensed under the GNU Affero General Public License v3.0 or later
(SPDX: `AGPL-3.0-or-later`). See [LICENSE](LICENSE) for the full text.

This package vendors a subset of code adapted from the
[OpenQuake Engine](https://github.com/gem/oq-engine) (© GEM Foundation,
AGPL-3.0-or-later); see
[`src/djura/record_selection/gsim/NOTICE.md`](src/djura/record_selection/gsim/NOTICE.md)
for attribution details.
