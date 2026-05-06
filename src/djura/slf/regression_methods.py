# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026 Djura | Risk - Data - Engineering S.r.l.
import numpy as np


def weibull(x, popt):
    a, b, c = popt
    return a * (1 - np.exp(-((x / b) ** c)))


def papadopoulos(x, popt):
    a, b, c, d, e = popt
    return e * x**a / (b**a + x**a) + (1 - e) * x**c / (d**c + x**c)
