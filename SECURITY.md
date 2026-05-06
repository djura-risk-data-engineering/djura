# Security Policy

## Reporting a Vulnerability

If you believe you have found a security vulnerability in `djura`, please
report it privately. **Do not open a public GitHub issue.**

- **Email:** info@djura.it

Encrypted reports are welcome; contact us at the address above to exchange a
PGP key if needed.

Please include:

- A description of the vulnerability and its potential impact.
- Steps to reproduce, or a minimal proof-of-concept.
- The affected version(s) of `djura` and your Python environment.
- Whether the issue is already public or coordinated with another party.

We aim to acknowledge new reports within **5 business days** and to provide a
remediation plan within **30 days** for confirmed issues. Please give us
reasonable time to investigate and release a fix before any public disclosure.

## Supported Versions

Security fixes are issued for the latest minor release on PyPI. Older minor
versions receive fixes only for critical issues at the maintainers'
discretion.

| Version  | Supported          |
| -------- | ------------------ |
| 0.1.x    | :white_check_mark: |
| < 0.1    | :x:                |

This table will be updated as new minor releases ship.

## Scope

In scope:

- The `djura` Python package as published on PyPI.
- Build, release, and CI workflows in this repository.

Out of scope:

- Vulnerabilities in third-party dependencies (please report those upstream;
  we will track and pull in fixes).
- Issues that require a malicious local user with the ability to execute
  arbitrary code or write to the Python environment.
- Loading untrusted pickle, HDF5, or model files from the network — Python's
  pickle format is **not** a safe deserialization boundary, and `djura` does
  not attempt to make it one. Treat any data passed to loaders as trusted.

## Hardening Recommendations for Users

- Install only from PyPI, and verify the version with `pip show djura`.
- Pin `djura` (and its dependencies) in your environment for reproducibility.
- Do not load `.pickle` / `.pkl` / `.h5` files received from untrusted
  sources; convert to a safe format (JSON, Parquet, `safetensors`) first.
- Keep `numpy`, `scipy`, `pandas`, `xgboost`, and other scientific
  dependencies up to date — most CVEs in the scientific Python stack are
  fixed in patch releases.
