# Record-Selection Metadata — Attribution and Usage Notes

## Data sources

The default metadata distributed in `flatfile_shallow.pickle` combines
records from two ground-motion databases:

- **NGA-West2 Ground Motion Database**, hosted by the
  **Pacific Earthquake Engineering Research Center (PEER)** at the
  University of California, Berkeley.
  - Database: <https://ngawest2.berkeley.edu>
  - Maintainer: PEER, UC Berkeley
- **ESM (Engineering Strong-Motion) flatfile**, distributed by the
  **ESM database** (ORFEUS / INGV and partners).
  - Flatfile: <https://esm-db.eu/#/products/flat_file>
  - Maintainer: ESM database consortium

The source of each record is recorded in the `database` field of the
metadata. All credit for the underlying ground motion records, station
and event parameters, and source-to-site distance computations belongs to
PEER and the NGA-West2 project contributors and to the ESM database and
its contributors, respectively.

> The legacy `NGA_W2_v2.pickle` dataset (NGA-West2 only) is still bundled
> and can be selected via the `DJURA_METADATA_PATH` environment variable;
> for that file the ESM notes below do not apply.

## Modifications

The pickle distributed via this project's GitHub Releases is **not** a
verbatim copy of either database. It contains:

- A subset of the original record-level metadata from each source
  (site information, event information, and waveform **filenames**).
- Additional computed and derived fields produced by this project
  (e.g. precomputed intensity measures and indexing structures used by
  `djura.record_selection`).

It does **not** contain:

- Any original waveform time series (acceleration, velocity, or
  displacement records) from either database.
- Any data redistributed in violation of the PEER or ESM terms of use.

## Obtaining the waveform records

The metadata file references waveforms by their original filenames but
does not include the waveforms themselves. Use the `database` field to
determine which source a record comes from, then obtain the waveform from
that provider.

### NGA-West2 records

1. Register for a free account at <https://ngawest2.berkeley.edu>.
2. Sign in to the NGA-West2 Ground Motion Database.
3. Use the database's search interface to locate records of interest,
   or look them up directly by the filenames stored in this metadata
   file.
4. Download the corresponding waveform files from PEER and place them
   in your own working directory.

Users are responsible for complying with the
[PEER NGA-West2 terms of use](https://ngawest2.berkeley.edu) when
downloading and redistributing waveform data.

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
NGA-West2 project and the ESM database — whichever sources your selected
records draw from — alongside any `djura` references; see the project
[README](../../../../README.md) and `CITATION.cff` for details.
