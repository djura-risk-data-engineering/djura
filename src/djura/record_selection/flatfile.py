# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026 Djura | Risk - Data - Engineering S.r.l.
from typing import Sequence

from scipy.interpolate import interp1d
import numpy as np

from .intensity_measure import IntensityMeasure
from .constants import SUPPORTED_IMS


class Flatfile:

    def __init__(
            self, metadata: dict, verbosity: int = 0) -> None:
        self._meta = None
        self.im_name: str = None
        self.metadata = metadata
        self.verbosity = verbosity
        self._IM = IntensityMeasure()

        self.damping = self.metadata["damping"]

        self._handles = {
            "SA": self.add_missing_sa,
            "SA_vert": self.add_missing_sa,
            "Sa_avg2": self.add_missing_sa_avg,
            "Sa_avg3": self.add_missing_sa_avg,
            "FIV3": self.add_missing_fiv3,
        }

    def get_metadata_keys(self, get_meta: bool = False):
        print(self.metadata.keys())
        if get_meta:
            print(self.metadata["__meta__"])

    def add_missing_im(self, im_name: str, period: float):
        if im_name not in SUPPORTED_IMS:
            raise ValueError(
                f"IM {im_name} is not supported. Supported IM names include:\n"
                f"{list(SUPPORTED_IMS)}")

        period = np.round(period, 5)

        self.im_name = im_name

        self._handles[im_name](period)

    def _interpolator(
            self, im_key: str, period_key: str, period: float = None):
        self.metadata[period_key] = np.round(self.metadata[period_key], 5)
        f = interp1d(
            self.metadata[period_key], self.metadata[im_key], axis=1)
        vals: np.ndarray = f(period)
        vals.shape = (len(vals), 1)
        im = np.append(self.metadata[im_key], vals, axis=1)
        return im

    def _add_missing_period(
            self, period_key: str, im_keys: Sequence[str], period: float):
        """Add a period to the metadata, interpolating every component

        The periods are read before the components are interpolated, since
        _interpolator rounds the stored periods as it goes, and the ordering
        they give is shared by every component.

        Parameters
        ----------
        period_key : str
            Metadata key holding the periods of the intensity measure
        im_keys : Sequence[str]
            Metadata keys of the components to interpolate
        period : float
            Period of interest
        """
        if period in self.metadata[period_key]:
            return

        periods = np.append(self.metadata[period_key], period)
        order = np.argsort(periods)

        for im_key in im_keys:
            im = self._interpolator(im_key, period_key, period)
            self.metadata[im_key] = im[:, order]

        self.metadata[period_key] = np.sort(periods)

    def add_missing_sa(self, period: float):
        self._add_missing_period(
            "Periods_SA",
            ("SA_1", "SA_2", "SA_vert", "SA_RotD50", "SA_RotD100"),
            period)

    def add_missing_sa_avg(self, period: float):
        name = self.im_name
        self._add_missing_period(
            "Periods_Sa_avg",
            (f"{name}_1", f"{name}_2",
             f"{name}_RotD50", f"{name}_RotD100"),
            period)

    def add_missing_fiv3(self, period: float):
        name = self.im_name
        self._add_missing_period(
            "Periods_FIV3", (f"{name}_1", f"{name}_2"), period)

    def get_im(self, im_type, period, acc, dt, damping, bounds, size):
        if "avg" in im_type.lower():
            return self._IM.get_sa_avg(acc, dt, period, damping, bounds, size)
        elif im_type.lower() in ("sat", "sa"):
            return self._IM.get_sat(period, acc, dt, damping)

        raise ValueError(f"IM type {im_type} not supported...")
