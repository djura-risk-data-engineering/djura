# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026 Djura | Risk - Data - Engineering S.r.l.
import pickle
import logging


logger = logging.getLogger()


def read_metadata(metadata):
    if not isinstance(metadata, dict):
        if metadata.suffix == ".pickle" or metadata.suffix == ".pkl":
            try:
                with open(metadata, "rb") as f:
                    metadata = pickle.load(f)
            except Exception as e:
                print("An unexpected error occurred when reading "
                      f"metadata file: {e}")

        elif metadata.suffix == ".npz":
            from numpy import load

            try:
                metadata = load(metadata, allow_pickle=True)
            except Exception as e:
                print("An unexpected error occurred when reading "
                      f"metadata file: {e}")

        else:
            raise ValueError(
                f"Wrong metadata filetype: {metadata.suffix}")
    return metadata
