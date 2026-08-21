# Record-Selection Metadata — Attribution and Usage Notes

## Data sources

The default metadata distributed in `flatfile_shallow_v1.pickle` includes
records from the following ground-motion database:

- **ESM (Engineering Strong-Motion) flatfile**, distributed by the
  **ESM database** (ORFEUS / INGV and partners).
  - Flatfile: <https://esm-db.eu/#/products/flat_file>
  - Maintainer: ESM database consortium

The source of each record is recorded in the `database` field of the
metadata. All credit for the underlying ground motion records, station
and event parameters, and source-to-site distance computations belongs to
the ESM database and its contributors.

This is the only dataset distributed with djura. Any other flatfile must
be provided by the user: map the records onto the common metadata schema
and point `DJURA_METADATA_PATH` at the file.

## Modifications

The pickle distributed via this project's GitHub Releases is **not** a
verbatim copy of the source database. It contains:

- A subset of the original record-level metadata (site information, event
  information, and waveform **filenames**).
- Additional computed and derived fields produced by this project
  (e.g. precomputed intensity measures and indexing structures used by
  `djura.record_selection`).

It does **not** contain:

- Any original waveform time series (acceleration, velocity, or
  displacement records).
- Any data redistributed in violation of the ESM terms of use.

## Obtaining the waveform records

The metadata file references waveforms by their original filenames but
does not include the waveforms themselves.

### ESM records

1. Register for a free account at <https://esm-db.eu>.
2. Sign in to the ESM database.
3. Locate records of interest through the ESM web interface (or web
   services), e.g. by event identifier and station, matching the
   filenames stored in this metadata file.
4. Download the corresponding waveform files from ESM and place them in
   your own working directory.

Users are responsible for complying with the
[ESM terms of use and licensing](https://esm-db.eu) when downloading and
redistributing waveform data.

## How to cite

If you use this metadata in academic or engineering work, please cite the
ESM database alongside any `djura` references; see the project
[README](../../../../README.md) and `CITATION.cff` for details.
