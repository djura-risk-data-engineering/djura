# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026 Djura | Risk - Data - Engineering S.r.l.
from scipy.interpolate import interp1d
import numpy as np

from .intensity_measure import IntensityMeasure
from .constants import SUPPORTED_IMS


class NGAWest2:

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

    def add_missing_sa(self, period: float):
        period_key = "Periods_SA"
        if period in self.metadata[period_key]:
            return

        periods = np.append(self.metadata[period_key], period)

        im = self._interpolator("SA_1", period_key, period)
        self.metadata['SA_1'] = im[:, np.argsort(periods)]

        im = self._interpolator("SA_2", period_key, period)
        self.metadata['SA_2'] = im[:, np.argsort(periods)]

        im = self._interpolator("SA_vert", period_key, period)
        self.metadata['SA_vert'] = im[:, np.argsort(periods)]

        im = self._interpolator("SA_RotD50", period_key, period)
        self.metadata['SA_RotD50'] = im[:, np.argsort(periods)]

        im = self._interpolator("SA_RotD100", period_key, period)
        self.metadata['SA_RotD100'] = im[:, np.argsort(periods)]

        self.metadata[period_key] = np.sort(periods)

    def add_missing_sa_avg(self, period: float):
        period_key = "Periods_Sa_avg"
        if period in self.metadata[period_key]:
            return
        periods = np.append(self.metadata[period_key], period)

        im = self._interpolator(
            f"{self.im_name}_1", period_key, period)
        self.metadata[f"{self.im_name}_1"] = im[:, np.argsort(periods)]

        im = self._interpolator(
            f"{self.im_name}_2", period_key, period)
        self.metadata[f"{self.im_name}_2"] = im[:, np.argsort(periods)]

        im = self._interpolator(
            f"{self.im_name}_RotD50", period_key, period)
        self.metadata[f"{self.im_name}_RotD50"] = im[:, np.argsort(periods)]

        im = self._interpolator(
            f"{self.im_name}_RotD100", period_key, period)
        self.metadata[f"{self.im_name}_RotD100"] = im[:, np.argsort(periods)]

        self.metadata[period_key] = np.sort(periods)

    def add_missing_fiv3(self, period: float):
        period_key = "Periods_FIV3"
        if period in self.metadata[period_key]:
            return

        periods = np.append(self.metadata[period_key], period)

        im = self._interpolator(
            f"{self.im_name}_1", period_key, period)
        self.metadata[f"{self.im_name}_1"] = im[:, np.argsort(periods)]

        im = self._interpolator(
            f"{self.im_name}_2", period_key, period)
        self.metadata[f"{self.im_name}_2"] = im[:, np.argsort(periods)]

        self.metadata[period_key] = np.sort(periods)

    def get_im(self, im_type, period, acc, dt, damping, bounds, size):
        if "avg" in im_type.lower():
            return self._IM.get_sa_avg(acc, dt, period, damping, bounds, size)
        elif im_type.lower() == "sat" or im_type.lower() == "sa":
            return self._IM.get_sat(period, acc, dt, damping)

        raise ValueError(f"IM type {im_type} not supported...")
