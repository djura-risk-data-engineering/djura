# djura

[![CI](https://github.com/djura-risk-data-engineering/djura/actions/workflows/ci.yml/badge.svg)](https://github.com/djura-risk-data-engineering/djura/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/djura)](https://pypi.org/p/djura)
[![Docs](https://readthedocs.org/projects/djura/badge/?version=latest)](https://djura.readthedocs.io)
[![Python](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue.svg)](https://www.python.org/)
[![License: AGPL v3+](https://img.shields.io/badge/license-AGPL--3.0--or--later-blue.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.60756%2FDJURA--HD26-blue.svg)](https://doi.org/10.60756/DJURA-HD26)

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

djura is organised as a set of applications that can be installed
independently. A bare install provides the shared core (numpy, scipy,
pydantic) and no application dependencies:

```bash
pip install djura
```

**Install the applications you need** — the extra is named after the
submodule:

```bash
pip install "djura[record_selection]"        # ground motion record selection
pip install "djura[hazard_consistency]"      # hazard-consistent IM analysis
pip install "djura[edp_im]"                  # EDP-IM prediction
pip install "djura[fragility_converter]"     # fragility/vulnerability conversion
pip install "djura[vulnerability_modeller]"  # vulnerability and loss modelling
pip install "djura[slf]"                     # storey loss functions
```

Extras combine, so several applications can be installed at once:

```bash
pip install "djura[record_selection,slf]"
```

**Everything at once** — equivalent to the pre-2.0 behaviour of a bare
`pip install djura`:

```bash
pip install "djura[all]"
```

**Optional accelerators and file formats:**

```bash
pip install "djura[record_selection,hdf5]"   # adds h5py for GMPE tables
pip install "djura[edp_im,xgboost]"          # adds gradient-boosted models
```

> **Upgrading from 1.x:** a bare `pip install djura` no longer installs every
> application's dependencies. Replace it with `pip install "djura[all]"` to
> keep the previous behaviour, or name only the applications you use.
> Importing an application whose extra is missing raises an `ImportError`
> naming the command to run.

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

A complete list of the supported ground motion models and intensity measure correlation models, with citations to their scientific publications, is given in [MODELS.md](MODELS.md).

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

## Bundled dataset

The bundled metadata pickle (~220 MB uncompressed) is **not** shipped
inside the wheel. It is hosted as a gzip-compressed asset on a GitHub
Release and downloaded automatically the first time it is needed:

```python
from djura.data_loader import load_data, clear_cache

data = load_data()       # downloads on first call, then loads from cache
clear_cache()            # delete the cached file to force a re-download
```

The cache lives at `~/.cache/djura/flatfile_shallow_v1.pickle`.

This is the only dataset distributed with djura. **Any additional flatfile
must be provided by the user**: a different ground motion database, a
regional subset, or an extended version of your own. Nothing in the package
needs to be modified to use one: map the records onto the common metadata
schema, then point `DJURA_METADATA_PATH` at your file and the selection
routines run unchanged. See the
[custom metadata guide](https://djura.readthedocs.io/en/latest/custom_metadata.html)
for the schema reference and a step-by-step example.

The bundled dataset contains metadata only, no waveform records, and has
been extended with fields computed by this project. See
[`src/djura/record_selection/assets/ATTRIBUTION.md`](src/djura/record_selection/assets/ATTRIBUTION.md)
for full attribution and for instructions on obtaining the underlying
waveforms.

### Publishing a new data release (maintainers)

The release is produced by the `release-data` GitHub Actions workflow,
which compresses the pickle and uploads it to a tagged GitHub Release.

1. Put the new pickle at
   `src/djura/record_selection/assets/flatfile_shallow_v1.pickle`. The
   workflow reads that exact path and fails if it is absent. It is not
   committed: the file is far too large for the repository and for the
   wheel.

2. Compute the SHA-256 of the compressed asset **the same way the workflow
   compresses it**, `gzip -9 -n` (the `-n` strips the filename and
   timestamp header, so the `.gz` is byte-for-byte reproducible and its
   digest is stable):

   ```bash
   gzip -9 -nc src/djura/record_selection/assets/flatfile_shallow_v1.pickle \
     > flatfile_shallow_v1.pickle.gz
   sha256sum flatfile_shallow_v1.pickle.gz
   ```

3. Publish the release, tagging it one version above the last data
   release:

   ```bash
   gh workflow run release-data.yml -f version=data-v3
   ```

   The workflow prints the size and the SHA-256 of the asset it uploaded;
   it must match the digest from step 2.

4. Update `src/djura/data_loader.py` to the new asset: `DATA_FILENAME`,
   the tag and filename in `GITHUB_RELEASE_URL`, and `EXPECTED_SHA256`.
   A mismatched digest makes the download fail with a checksum error
   rather than silently loading the wrong data.

5. Verify end to end from a clean cache:

   ```bash
   python -c "from djura.data_loader import clear_cache, load_data; \
   clear_cache(); print(len(load_data()['magnitude']))"
   ```

Users who already have the previous dataset cached keep using it until
they call `clear_cache()`, the cache being keyed by filename, so a renamed
asset triggers a fresh download on its own.

## How to cite

If you use **djura** in academic or research work, please cite the package
**and** the paper(s) backing the submodule(s) you use.

The software itself has a persistent DOI: [10.60756/DJURA-HD26](https://doi.org/10.60756/DJURA-HD26).

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

Contributions are welcome. Please read the [contributing guide](CONTRIBUTING.md)
for development setup, testing, and the pull-request process, and note that
participation is governed by our [Code of Conduct](CODE_OF_CONDUCT.md).

By submitting a Contribution, you agree to the terms of the
[Contributor License Agreement](CLA.md), which (among other things) allows the
maintainer to relicense the project. For example, to offer a separate
commercial license alongside AGPL-3.0.

## Maintenance and sustainability

djura is actively developed and maintained by
**Djura | Risk - Data - Engineering S.r.l.**, with development led by the
authors listed in [`CITATION.cff`](CITATION.cff). The package consolidates the
algorithms underpinning the company's research and commercial activity, which
gives its continued maintenance a durable institutional basis beyond any
single contributor or grant.

- **Releases and versioning.** The project follows semantic versioning, with
  versions derived automatically from git tags. Each release is published to
  [PyPI](https://pypi.org/p/djura) and archived with a persistent DOI
  ([10.60756/DJURA-HD26](https://doi.org/10.60756/DJURA-HD26)). Notable
  changes are recorded in [`CHANGELOG.md`](CHANGELOG.md).
- **Quality assurance.** Every push and pull request runs continuous
  integration (linting, the test suite across Linux/macOS/Windows and the
  supported Python versions, and a packaging/metadata check). CodeQL scanning
  and Dependabot dependency updates are enabled.
- **Issue tracking and support.** Bugs and feature requests are handled
  through the [GitHub issue tracker](https://github.com/djura-risk-data-engineering/djura/issues).
  Security reports follow the process in [`SECURITY.md`](SECURITY.md).
- **Contributions.** External contributions are welcomed under the process in
  [`CONTRIBUTING.md`](CONTRIBUTING.md) and the [Code of Conduct](CODE_OF_CONDUCT.md).
- **Longevity.** Should the company ever discontinue maintenance, the
  AGPL-3.0-or-later license and the public, DOI-archived releases ensure the
  community can continue to use, fork, and maintain the software.

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
