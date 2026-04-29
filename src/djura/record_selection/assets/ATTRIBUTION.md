# NGA-West2 Metadata — Attribution and Usage Notes

## Data source

The metadata distributed in `NGA_W2_v2.pickle` is derived from the
**NGA-West2 Ground Motion Database**, hosted by the
**Pacific Earthquake Engineering Research Center (PEER)** at the
University of California, Berkeley.

- Database: <https://ngawest2.berkeley.edu>
- Maintainer: PEER, UC Berkeley

All credit for the underlying ground motion records, station and event
parameters, and source-to-site distance computations belongs to PEER
and the NGA-West2 project contributors.

## Modifications

The pickle distributed via this project's GitHub Releases is **not** a
verbatim copy of the NGA-West2 database. It contains:

- A subset of the original NGA-West2 record-level metadata
  (site information, event information, and waveform **filenames**).
- Additional computed and derived fields produced by this project
  (e.g. precomputed intensity measures and indexing structures used by
  `djura.record_selection`).

It does **not** contain:

- Any original NGA-West2 waveform time series (acceleration, velocity,
  or displacement records).
- Any data redistributed in violation of the PEER terms of use.

## Obtaining the waveform records

The metadata file references waveforms by their original NGA-West2
filenames but does not include the waveforms themselves. To obtain the
actual records:

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

## How to cite

If you use this metadata in academic or engineering work, please cite
the NGA-West2 project alongside any `djura` references; see the project
[README](../../../../ReadMe.md) and `CITATION.cff` for details.
