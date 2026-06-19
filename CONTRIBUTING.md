# Contributing to djura

Thank you for your interest in improving **djura**. Contributions of all
kinds are welcome: bug reports, feature requests, documentation, and code.

This project is maintained by **Djura | Risk - Data - Engineering S.r.l.**
and released under the **GNU AGPL-3.0-or-later** license.

## Code of Conduct

By participating in this project you agree to abide by our
[Code of Conduct](CODE_OF_CONDUCT.md). Please report unacceptable behaviour
to [info@djura.it](mailto:info@djura.it).

## Contributor License Agreement

By submitting a contribution (pull request, patch, or otherwise) you agree to
the terms of the [Contributor License Agreement](CLA.md), which - among other
things - allows the maintainer to relicense the project (for example, to offer
a separate commercial license alongside AGPL-3.0).

## Reporting bugs and requesting features

Please open an issue on the
[GitHub issue tracker](https://github.com/djura-risk-data-engineering/djura/issues).
For bug reports, include:

- a minimal, reproducible example;
- the djura version (`python -c "import djura; print(djura.__version__)"`);
- your operating system and Python version;
- the full traceback, if any.

## Setting up a development environment

djura uses [Poetry](https://python-poetry.org/) with
[poetry-dynamic-versioning](https://github.com/mtkennerly/poetry-dynamic-versioning).

```bash
git clone https://github.com/djura-risk-data-engineering/djura.git
cd djura
poetry install --with dev          # testing and linting
poetry install --with dev,docs     # also documentation dependencies
```

## Running the test suite

```bash
# fast tests only (default in CI)
poetry run pytest -m "not slow"

# the full suite, including slow tests that require the local NGA-West2 dataset
export DJURA_METADATA_PATH=src/djura/record_selection/assets/NGA_W2_v2.pickle
poetry run pytest

# with coverage
poetry run pytest --cov=djura --cov-report=term-missing -m "not slow"
```

On Windows (PowerShell), set the dataset path with:

```powershell
$env:DJURA_METADATA_PATH = "src/djura/record_selection/assets/NGA_W2_v2.pickle"
```

## Code style

The project uses [`flake8`](https://flake8.pycqa.org/) for linting, with a
maximum line length of 79 characters. CI fails on lint errors.

```bash
poetry run flake8 src tests
```

Please match the conventions of the surrounding code: clear docstrings
(NumPy style), descriptive names, and small, focused functions.

## Building the documentation

```bash
poetry install --with docs
cd docs
make html        # output in docs/build/html/index.html
```

## Pull request process

1. Fork the repository and create a topic branch off `main`.
2. Make your change, with tests and documentation where appropriate.
3. Ensure `flake8` and `pytest -m "not slow"` pass locally.
4. Update `CHANGELOG.md` under the unreleased section if your change is
   user-facing.
5. Open a pull request against `main` with a clear description of the change
   and its motivation. CI (lint, tests on Linux/macOS/Windows, build check)
   must pass before review.

We aim to review pull requests promptly. Thank you for contributing!
