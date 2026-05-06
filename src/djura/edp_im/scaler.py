# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026 Djura | Risk - Data - Engineering S.r.l.
class MinMaxScaler:
    def __init__(self, lower_bound, upper_bound) -> None:
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def transform(self, data, min=0, max=1):
        x_std = (data - self.lower_bound) / \
            (self.upper_bound - self.lower_bound)

        x_scaled = x_std * (max - min) + min
        return x_scaled
