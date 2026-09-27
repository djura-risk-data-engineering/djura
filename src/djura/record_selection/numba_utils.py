# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026 Djura | Risk - Data - Engineering S.r.l.
import numpy as np
from numba import njit
from math import log, erf, sqrt


@njit
def lognormal_cdf(x, mu, sigma):
    if x <= 0:
        return 0.0
    z = (log(x) - mu) / (sigma * sqrt(2))
    return 0.5 * (1 + erf(z))


@njit
def compute_ks_statistic_1d(data, mu, sigma):
    n = len(data)
    sorted_data = np.sort(data)
    max_diff = 0.0
    for i in range(n):
        F_empirical = (i + 1) / n
        F_theoretical = lognormal_cdf(sorted_data[i], mu, sigma)
        diff = abs(F_empirical - F_theoretical)
        if diff > max_diff:
            max_diff = diff
    return max_diff


@njit
def get_mean_std_skewness(arr):
    n_rows, n_cols = arr.shape
    skew = np.zeros(n_cols)
    std = np.zeros(n_cols)
    mu = np.zeros(n_cols)
    for i in range(n_cols):
        mu[i] = arr[:, i].mean()
        std[i] = arr[:, i].std()
        if std[i] > 0.0:
            for j in range(n_rows):
                skew[i] += ((arr[j, i] - mu[i]) / std[i]) ** 3
            skew[i] /= n_rows

    return mu, std, skew


@njit
def get_mean_std(arr):
    _, n_cols = arr.shape
    std = np.zeros(n_cols)
    mu = np.zeros(n_cols)
    for i in range(n_cols):
        mu[i] = arr[:, i].mean()
        std[i] = arr[:, i].std()

    return mu, std


@njit
def greedy_algorithm(
    scaled_imi, sf, mu_imi, sigma_imi, rec_id, ln_imi_db,
    num_rec, error_weights, penalty, db_idxs, alpha, im_weights,
    selected_record_id, dev_min
):

    for _rec, (imi, db_idx) in enumerate(zip(ln_imi_db, db_idxs)):

        imi_trial = np.concatenate(
            (scaled_imi, imi[np.newaxis, :] + np.log(sf[_rec] ** alpha)),
            axis=0)

        # Compute deviations from target GCIM distribution
        # im_weights added
        # mu_trial, std_trial, skew_trial = get_mean_std_skewness(imi_trial)
        mu_trial, std_trial = get_mean_std(imi_trial)
        dev_mean = (mu_trial - mu_imi) * im_weights
        dev_stddev = (std_trial - sigma_imi) * im_weights

        dev_total = error_weights[0] * np.sum(dev_mean ** 2) \
            + error_weights[1] * np.sum(dev_stddev ** 2) \
            # + error_weights[2] * np.sum(skew_trial ** 2)

        # Avoid repetition of records
        if np.any(rec_id == db_idx):
            dev_total += dev_min + 1e8

        elif penalty > 0:
            for r in range(num_rec):
                dev_total += np.sum(np.abs(
                    np.exp(imi_trial[r]) > np.exp(mu_imi + 3.0 * sigma_imi)
                )) * penalty

                dev_total += np.sum(np.abs(
                    np.exp(imi_trial[r]) < np.exp(mu_imi - 3.0 * sigma_imi)
                )) * penalty

        if dev_total < dev_min:
            selected_record_id = db_idx
            dev_min = dev_total

    return selected_record_id, dev_min


@njit
def greedy_algorithm_fast(
    scaled_imi, sf, mu_imi, sigma_imi, rec_id, ln_imi_db,
    num_rec, error_weights, penalty, db_idxs, alpha, im_weights,
    selected_record_id, dev_min
):
    """Faster form of ``greedy_algorithm``, with the same arguments and
    outputs.

    Every candidate shares the records in ``scaled_imi``, so their sums of
    ln(IM) and ln(IM)^2 and their count of values outside the +/-3 sigma
    bounds are computed once, and each candidate only adds its own row. The
    bounds are compared in log space, exp being monotonic. This takes the
    cost per candidate from O(num_rec * n_im) to O(n_im).
    """
    n_fix, n_im = scaled_imi.shape
    n = n_fix + 1

    upper = mu_imi + 3.0 * sigma_imi
    lower = mu_imi - 3.0 * sigma_imi

    # Sums and bound violations of the records shared by every candidate
    s1 = np.zeros(n_im)
    s2 = np.zeros(n_im)
    for i in range(n_fix):
        for j in range(n_im):
            s1[j] += scaled_imi[i, j]
            s2[j] += scaled_imi[i, j] ** 2

    penalty_fix = 0.0
    if penalty > 0:
        for i in range(min(num_rec, n_fix)):
            for j in range(n_im):
                v = scaled_imi[i, j]
                penalty_fix += ((v > upper[j]) + (v < lower[j])) * penalty

    # The candidate is the last row, within the first 'num_rec' only when
    # the kept records number fewer
    candidate_penalised = num_rec > n_fix

    for _rec in range(ln_imi_db.shape[0]):
        db_idx = db_idxs[_rec]

        dev_total = 0.0
        penalty_rec = 0.0
        for j in range(n_im):
            x = ln_imi_db[_rec, j] + np.log(sf[_rec] ** alpha[j])
            mu = (s1[j] + x) / n
            std = np.sqrt(max((s2[j] + x * x) / n - mu * mu, 0.0))
            dev_mean = (mu - mu_imi[j]) * im_weights[j]
            dev_stddev = (std - sigma_imi[j]) * im_weights[j]
            dev_total += error_weights[0] * dev_mean ** 2 \
                + error_weights[1] * dev_stddev ** 2
            if candidate_penalised:
                penalty_rec += ((x > upper[j]) + (x < lower[j])) * penalty

        # Avoid repetition of records
        if np.any(rec_id == db_idx):
            dev_total += dev_min + 1e8

        elif penalty > 0:
            dev_total += penalty_fix + penalty_rec

        if dev_total < dev_min:
            selected_record_id = db_idx
            dev_min = dev_total

    return selected_record_id, dev_min
