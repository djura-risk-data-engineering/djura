# djura

[![CI](https://github.com/djura-risk-data-engineering/djura/actions/workflows/ci.yml/badge.svg)](https://github.com/djura-risk-data-engineering/djura/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/djura)](https://pypi.org/p/djura)
[![Docs](https://readthedocs.org/projects/djura/badge/?version=latest)](https://djura.readthedocs.io)
[![Python](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue.svg)](https://www.python.org/)
[![License: AGPL v3+](https://img.shields.io/badge/license-AGPL--3.0--or--later-blue.svg)](LICENSE)

**djura** is a scientific Python toolkit developed and maintained by Djura | Risk - Data - Engineering S.r.l. for general engineering applications. It bundles
into a single installable package the core algorithms used across the djura
research stack: ground motion record selection, hazard-consistent intensity
measure analysis, structural vulnerability modelling, and storey loss function
generation with no web-server, database, or cloud-storage dependencies.

The package is intended for **research and educational use** only, and is released
under the **GNU AGPL-3.0-or-later** license so that it composes cleanly with
other copyleft scientific tools (e.g. `openquake.engine`).

> **Commercial use?** djura is **dual-licensed**. The AGPL-3.0-or-later terms
> below apply to academic, research, and other open-source use only. If you want to
> use djura for commercial or revenue-generating purposes - including as part of a closed-source product, as part of an internal commercial workflow, or a
> network-accessible service without releasing your own source code under the
> AGPL - you need a **separate commercial license**. Contact
> [info@djura.it](mailto:info@djura.it) to arrange one. See
> [Commercial licensing](#commercial-licensing) below.

> **AGPL-3.0 notice** - If you use `djura` as part of a network-accessible
> service (API, web application, SaaS backend), the AGPL requires that you
> make the complete corresponding source code available to your users. Running
> `djura` in a private research environment or on your own workstation is not
> affected. See the [LICENSE](LICENSE) file for the full terms.

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
**Optional runtime extras:**

```bash
pip install "djura[plot]"   # adds matplotlib
pip install "djura[hdf5]"   # adds h5py
pip install "djura[all]"    # all of the above
```

**For contributors** — install development and/or documentation dependencies
using Poetry dependency groups:

```bash
poetry install --with dev        # testing and linting (pytest, flake8)
poetry install --with docs       # Sphinx + furo for building the docs
poetry install --with dev,docs   # everything
```

> `sphinx-autodoc-typehints` in the `docs` group requires Python ≥ 3.12
> and is skipped automatically on earlier versions.

## Documentation

For documentation on how to use the various djura packages, as well as example applications and tutorials, please refer to the [readthedocs](https://djura.readthedocs.io/en/latest/index.html) resources.

Additionally, several [blog posts](https://www.djura.it/blog) have been created with supplemental material on how to use these packages via the user interface available at our website [www.djura.it](https://apps.djura.it/login).

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

The dataset is derived from the
[NGA-West2 Ground Motion Database](https://ngawest2.berkeley.edu)
(PEER, UC Berkeley). It contains metadata only, no waveform records,
and has been extended with fields computed by this project. See
[`src/djura/record_selection/assets/ATTRIBUTION.md`](src/djura/record_selection/assets/ATTRIBUTION.md)
for full attribution and instructions on downloading the underlying
waveforms from PEER.

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
| `edp_im`                        | Shahnazaryan, D., & O'Reilly, G. J. (2024). *Next-generation non-linear and collapse prediction models for short- to long-period systems via machine learning methods*. **Engineering Structures**, 306, 117801. doi:[10.1016/j.engstruct.2024.117801](https://doi.org/10.1016/j.engstruct.2024.117801) |
| `vulnerability_modeller`        | O'Reilly, G. J., & Shahnazaryan, D. (2024). *On the utility of story loss functions for regional seismic vulnerability modeling and risk assessment*. **Earthquake Spectra**, 40(3), 1933–1955. doi:[10.1177/87552930241245940](https://doi.org/10.1177/87552930241245940) |
| `fragility_converter`           | O'Reilly, G. J., Ozsarac, V., & Shahnazaryan, D. (2025). *Conversion of seismic fragility and vulnerability models to alternative intensity measures for regional risk analysis*. **Earthquake Spectra** (Under Review). |
| `slf`                           | Shahnazaryan, D., Ozsarac, V., & O'Reilly, G. J. (2025). *The Role of Story Loss Functions in Regional Seismic Vulnerability Modelling and Risk Assessment*. 10th International Conference on Computational Methods in Structural Dynamics and Earthquake Engineering (COMPDYN 2025), Rhodes, Greece, Jun. 2025, pp. 780–804. doi: [10.7712/120125.12447.25302](https://doi.org/10.7712/120125.12447.25302)                    |

A `CITATION.cff` file is provided so that GitHub renders a "Cite this repository" button automatically.

## Contributing

Contributions are welcome. By submitting a Contribution, you agree to the
terms of the [Contributor License Agreement](CLA.md), which (among other
things) allows the maintainer to relicense the project. For example, to
offer a separate commercial license alongside AGPL-3.0.

## License

Copyright © 2025–2026 Djura | Risk - Data - Engineering S.r.l. (Italy). All rights reserved.

djura is **dual-licensed**:

- **Open-source license** — GNU Affero General Public License v3.0 or later
  (SPDX: `AGPL-3.0-or-later`). See [LICENSE](LICENSE) for the full text. This
  is the license that applies by default and covers academic, research, and
  other AGPL-compatible open-source use.
- **Commercial license** — available from Djura | Risk - Data - Engineering S.r.l. See [Commercial licensing](#commercial-licensing) below.

This package vendors a subset of code adapted from the
[OpenQuake Engine](https://github.com/gem/oq-engine) (© GEM Foundation,
AGPL-3.0-or-later); see
[`src/djura/record_selection/gsim/NOTICE.md`](src/djura/record_selection/gsim/NOTICE.md)
for attribution details. The OpenQuake-derived portions remain under
AGPL-3.0-or-later in all distributions.

## Commercial licensing

The AGPL-3.0-or-later imposes a strong copyleft obligation: if you distribute djura, or expose its functionality over a network (API, web app, SaaS backend, hosted analysis service, etc.), you must make the **complete corresponding source code** of your application available to its users under the AGPL.

If that is not compatible with your business — for example because you want to use djura for **commercial or revenue-generating purposes**, including:

- embedding djura in a **closed-source commercial product**;
- offering a **proprietary SaaS** or hosted service powered by djura without releasing your own source under the AGPL;
- using djura in **internal commercial workflows** without releasing your source code under the AGPL;
- receiving **warranties, indemnification, or commercial support** that the AGPL explicitly disclaims;

then you need a **commercial license** from **Djura | Risk - Data - Engineering S.r.l.** (Italy), the copyright holder.

To request a commercial license, please contact:

📧 **[info@djura.it](mailto:info@djura.it)**

Please include a short description of the intended use case (organisation,
product, deployment model, expected user base). We will reply with licensing
terms.

> Academic researchers, students, and other AGPL-compatible users do **not**
> need to contact us — the AGPL grant in [LICENSE](LICENSE) already covers
> you.
