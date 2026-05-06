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

        # # New: Compute total KS-based error across all IMI components
        # dev_total = 0.0
        # for i in range(imi_trial.shape[1]):
        #     # Transform back from log-space to IMI space
        #     imi_column = np.exp(imi_trial[:, i])
        #     ks = compute_ks_statistic_1d(imi_column, np.exp(mu_imi[i]),
        #                                  sigma_imi[i])
        #     dev_total += error_weights[2] * im_weights[i] * ks

        # Avoid repetition of records
        if np.any(rec_id == db_idx):
            dev_total += dev_min + 1e8

        elif penalty > 0:
            for r in range(num_rec):
                dev_total += np.sum(np.abs(
                    np.exp(imi_trial[r]) > np.exp(mu_imi + 3.0 * sigma_imi)
                )) * penalty

                dev_total += np.sum(np.abs(
                    np.exp(imi_trial[r]) > np.exp(mu_imi - 3.0 * sigma_imi)
                )) * penalty

        if dev_total < dev_min:
            selected_record_id = db_idx
            dev_min = dev_total

    return selected_record_id, dev_min


@njit
def greedy_optimization_code(
    error_metric, coupled_scaling, penalty, rec_ids, rec_eq_ids, min_id,
    eq_id_db, max_sf_h, min_sf_h, max_sf_v, min_sf_v, max_rec_per_event,
    weights, min_error,
    rec_im_h, target_spectrum_h, im_h_db, sf1_h_db, rec_sf1_h, rec_sf2_h,
    rec_im_v, target_spectrum_v, im_v_db, sf1_v_db, rec_sf1_v, rec_sf2_v,
    mean_lower_lim_h, mean_upper_lim_h, mean_lower_lim_v, mean_upper_lim_v,
    avg_sa_h_mean_ratio, min_sa_h_rec_target_ratio, max_sa_h_rec_target_ratio
) -> tuple[int, float, float]:
    """
    Details
    -------
    Greedy subset modification to select new candidate records such that:
    - Error for records' mean and target spectra is equal or less than before.
    - Error calculations include weights set for horizontal and vertical
    component.
    - Error calculations include penalization for bad spectra.
    - The limits for scaling factor (lower and upper) are satisfied.
    - The limits for mean spectrum (lower and upper) are satisfied.
    - The limit for the maximum number of records per event is satisfied.
    - The limit for the average of ratio of the target spectral acceleration
    values to the records' mean spectral acceleration values is satisfied
    (CEN2024).
    - The limit for the ratio of the target spectral acceleration values
    to the individual records' spectral acceleration values is satisfied
    (CEN2024).

    The method is defined separately so that njit can be used as wrapper and
    the routine can be run faster.

    Parameters
    ----------
    error_metric : str
        Error metric used for error calculations
        1: Root-mean-squared-error (RMSE)
        2: Logarithmic sum of squared differences (SSD)
    coupled_scaling : bool
        Flag indicated whether to use coupled scaling for horizontal and
        vertical components or not.
    penalty : float
        The factor used to penalize the bad record spectra
        (above 2*target or below 0.5*target)
    rec_ids : numpy.ndarray (1-D)
        Record IDs of the reduced candidate set records in the database
    rec_eq_ids : numpy.ndarray (1-D)
        Event IDs of the reduced candidate set records in the database
    min_id : int
        ID of the eliminated record from candidate record set
    eq_id_db : numpy.ndarray (2-D)
        Event IDs of the records in the filtered database
    min_sf_h : float
        Minimum allowed scaling factor (Horizontal Component)
    max_sf_h : float
        Maximum allowed scaling factor (Horizontal Component)
    min_sf_v : float
        Minimum allowed scaling factor (Vertical Component)
    max_sf_v : float
        Maximum allowed scaling factor (Vertical Component)
    max_rec_per_event : int
        The limit for the maximum number of records belong to the same event
    weights : numpy.ndarray (1-D)
        Error weights for horizontal and vertical spectral matching (size: 2)
    min_error : float
        Minimum error in the matching for the previous candidate set.
        (target spectra vs. mean spectra of records)
    rec_im_h : numpy.ndarray (2-D)
        Unscaled spectra of the reduced candidate record set
        (size: num_records - 1) (Horizontal Component)
    target_spectrum_h : numpy.ndarray (1-D)
        Target spectrum (Horizontal Component)
    im_h_db : numpy.ndarray (2-D)
        Spectra of the records in the filtered database (Horizontal Component)
    sf1_h_db : numpy.ndarray (1-D)
        Scaling factors, which provides the best matching to the target, for
        the records in the filtered database (Horizontal Component)
    rec_sf1_h : numpy.ndarray (2-D)
        Scaling factors, which provides the best matching to the target, for
        the records in the reduced candidate set (Horizontal Component)
    rec_sf2_h : numpy.ndarray (2-D)
        Scaling factors which shifts the set mean above the `mean_lower_lim_h`
        for the records in the previous candidate set (Horizontal Component)
    rec_im_v : numpy.ndarray (2-D)
        Unscaled spectra of the reduced candidate record set
        (size: num_records - 1) (Vertical Component)
    target_spectrum_v : numpy.ndarray (1-D)
        Target spectrum (Vertical Component)
    im_v_db : numpy.ndarray (2-D)
        Spectra of the records in the filtered database (Vertical Component)
    sf1_v_db : numpy.ndarray (1-D)
        Scaling factors, which provides the best matching to the target, for
        the records in the filtered database (Vertical Component)
    rec_sf1_v : numpy.ndarray (2-D)
        Scaling factors, which provides the best matching to the target, for
        the records in the reduced candidate set (Vertical Component)
    rec_sf2_v : numpy.ndarray (2-D)
        Scaling factors which shifts the set mean above the `mean_lower_lim_v`
        for the records in the previous candidate set (Vertical Component)
    mean_lower_lim : numpy.ndarray (1-D)
        Lower limit values for mean spectrum of selected records
        (Horizontal Component)
    mean_upper_lim : numpy.ndarray (1-D)
        Upper limit values for mean spectrum of selected records
        (Horizontal Component)
    mean_lower_lim_v : numpy.ndarray (1-D)
        Lower limit values for mean spectrum of selected records
        (Vertical Component)
    mean_upper_lim_v : numpy.ndarray (1-D)
        Upper limit values for mean spectrum of selected records
        (Vertical Component)
    avg_sa_h_mean_ratio : float
        The minimum allowed value for the average of ratio of the target
        spectral acceleration values to the records' mean spectral
        acceleration values (Horizontal Component)
        =mean(Sa_rec.mean(T)/Target(T))
    min_sa_h_rec_target_ratio : float
        The minimum allowed value for the ratio of the target
        spectral acceleration values to the individual records' spectral
        acceleration values (Horizontal Component)
    max_sa_h_rec_target_ratio : float
        The maximum allowed value for the ratio of the target
        spectral acceleration values to the individual records' spectral
        acceleration values (Horizontal Component)

    Returns
    -------
    min_id : int
        ID of the new selected record leading to smaller or equal error.
    rec_sf2 : numpy.ndarray (2-D)
        Scale factor for adjusting the mean of the candidate record set
        which includes the new record (size: num_records)
    rec_sf1_v : numpy.ndarray (2-D)
        Scaling factors which shifts the set mean above the `mean_lower_lim_v`
        for the records of the new candidate set (Horizontal Component)
    rec_sf2_v : numpy.ndarray (2-D)
        Scaling factors which shifts the set mean above the `mean_lower_lim_v`
        for the records of the new candidate set (Vertical Component)
    min_error : float
        Minimum error in the matching for the new candidate set.
        (target spectra vs. mean spectra of records)
    """

    def mean(arr):
        """
        Computes the mean of a 2-D array along axis=0.
        Required for computations since njit is used as wrapper.
        """
        out = np.zeros(arr.shape[1])
        for i in range(arr.shape[1]):
            out[i] = arr[:, i].mean()

        return out

    # Get scale ims for reduced candidate set
    rec_im_h_scaled1 = rec_im_h * rec_sf1_h
    if np.any(target_spectrum_v):
        rec_im_v_scaled1 = rec_im_v * rec_sf1_v

    for j in range(im_h_db.shape[0]):
        tmp = eq_id_db[j]
        # record should not be repeated and number of eqs from the same event
        # should not exceed 3
        bool1 = np.any(rec_ids == j)
        bool2 = np.sum(rec_eq_ids == tmp) < max_rec_per_event
        if not bool1 and bool2:
            # Add to the sample the scaled spectra (scaled based on mse)
            trial_im_h_scaled1 = np.zeros((1, len(im_h_db[j, :])))
            trial_im_h_scaled1[:, :] = im_h_db[j, :] * sf1_h_db[j]
            sf1_h_trial = np.append(rec_sf1_h.flatten(), sf1_h_db[j])
            rec_trial_im_h_scaled1 = np.concatenate(
                (rec_im_h_scaled1, trial_im_h_scaled1), axis=0)

            # Compute the 2nd scaling factor (due to lower limit)
            rec_trial_im_h_mean1 = mean(rec_trial_im_h_scaled1)
            sf2_h_trial = np.max(mean_lower_lim_h / rec_trial_im_h_mean1)
            sf2_h_trial = max(
                sf2_h_trial,
                avg_sa_h_mean_ratio / np.mean(rec_trial_im_h_mean1
                                              / target_spectrum_h)
            )
            sf2_h_trial = max(sf2_h_trial, 1.0)  # no need to decrease

            # Do the same for vertical component
            if np.any(target_spectrum_v):
                # Add to the sample the scaled spectra (scaled based on mse)
                trial_im_v_scaled1 = np.zeros((1, len(im_v_db[j, :])))
                trial_im_v_scaled1[:, :] = im_v_db[j, :] * sf1_v_db[j]
                sf1_v_trial = np.append(rec_sf1_v.flatten(), sf1_v_db[j])
                rec_trial_im_v_scaled1 = np.concatenate(
                    (rec_im_v_scaled1, trial_im_v_scaled1), axis=0)

                # Compute the 2nd scaling factor (due to lower limit)
                rec_trial_im_v_mean1 = mean(rec_trial_im_v_scaled1)
                if np.any(mean_lower_lim_v):
                    sf2_v_trial = np.max(
                        mean_lower_lim_v / rec_trial_im_v_mean1)
                    sf2_v_trial = max(sf2_v_trial, 1.0)  # no need to decrease
                else:
                    sf2_v_trial = 1.0

                if coupled_scaling:
                    sf2_trial = max(sf2_v_trial, sf2_h_trial)
                    sf2_h_trial = sf2_trial
                    sf2_v_trial = sf2_trial

            # Compute error in mean for scaled IMs
            rec_trial_im_h_mean = sf2_h_trial * rec_trial_im_h_mean1
            if error_metric == 1:  # rmse
                error_h = ((mean((target_spectrum_h.reshape(-1, 1)
                                  - rec_trial_im_h_mean.reshape(-1, 1))**2)
                            )**0.5)[0]
            elif error_metric == 2:  # ssd
                error_h = np.sum((np.log(target_spectrum_h)
                                 - np.log(rec_trial_im_h_mean))**2
                                 )
            if np.any(target_spectrum_v):  # Vertical component is considered
                error_h = weights[0] * error_h
                rec_trial_im_v_mean = sf2_v_trial * rec_trial_im_v_mean1
                if error_metric == 1:  # rmse
                    error_v = ((mean((target_spectrum_v.reshape(-1, 1)
                                      - rec_trial_im_v_mean.reshape(-1, 1))**2)
                                )**0.5)[0]
                elif error_metric == 2:  # ssd
                    error_v = np.sum((np.log(target_spectrum_v)
                                      - np.log(rec_trial_im_v_mean))**2
                                     )
                error_v = weights[1] * error_v
                error = error_h + error_v
            else:
                error = error_h

            # Penalize bad spectrum
            rec_trial_im_h = sf2_h_trial * rec_trial_im_h_scaled1
            if np.any(rec_trial_im_h / target_spectrum_h > 2.0):
                count = np.sum(rec_trial_im_h / target_spectrum_h > 2.0)
                error = error + count * penalty * error_h
            if np.any(rec_trial_im_h / target_spectrum_h < 0.5):
                count = np.sum(rec_trial_im_h / target_spectrum_h < 0.5)
                error = error + count * penalty * error_h
            if np.any(target_spectrum_v):
                rec_trial_im_v = sf2_v_trial * rec_trial_im_v_scaled1
                if np.any(rec_trial_im_v / target_spectrum_v > 2.0):
                    count = np.sum(rec_trial_im_v / target_spectrum_v > 2.0)
                    error = error + count * penalty * error_v
                if np.any(rec_trial_im_v / target_spectrum_v < 0.5):
                    count = np.sum(rec_trial_im_v / target_spectrum_v < 0.5)
                    error = error + count * penalty * error_v

            # Check final scaling factors for horizontal component
            sf_h_trial = sf1_h_trial * sf2_h_trial
            ok_sf_h_low = not np.any(sf_h_trial < max_sf_h)
            ok_sf_h_up = not np.any(sf_h_trial > min_sf_h)
            # Check mean against target upper limit (hor. comp)
            if np.any(mean_upper_lim_h):
                ok_tgt_h_up = not np.any(
                    mean_upper_lim_h - rec_trial_im_h_mean < 0)
            else:
                ok_tgt_h_up = True
            if np.any(rec_trial_im_h / target_spectrum_h
                      < min_sa_h_rec_target_ratio):
                ok_min_im_h = False
            else:
                ok_min_im_h = True
            if np.any(rec_trial_im_h / target_spectrum_h
                      > max_sa_h_rec_target_ratio):
                ok_max_im_h = False
            else:
                ok_max_im_h = True
            # Check if the set satisfies all the requirements (hor. comp)
            ok_h = (ok_sf_h_low and ok_sf_h_up and ok_tgt_h_up
                    and ok_min_im_h and ok_max_im_h)

            if np.any(target_spectrum_v):  # Checks for vertical component
                # Check final scaling factors
                sf_v_trial = sf1_v_trial * sf2_v_trial
                ok_sf_v_low = not np.any(sf_v_trial < max_sf_v)
                ok_sf_v_up = not np.any(sf_v_trial > min_sf_v)
                # Check mean against target upper limit
                if np.any(mean_upper_lim_v):
                    ok_tgt_v_up = not np.any(
                        mean_upper_lim_v - rec_trial_im_v_mean < 0)
                else:
                    ok_tgt_v_up = True
                # Check if the set satisfies all the requirements
                ok_v = ok_sf_v_low and ok_sf_v_up and ok_tgt_v_up
                # Both horizontal and vertical component should be ok
                ok = ok_h and ok_v
            else:
                ok = ok_h

            if ok and error < min_error:
                # Should cause improvement
                min_id = j
                rec_sf2_h = sf2_h_trial
                min_error = error
                if np.any(target_spectrum_v):
                    rec_sf2_v = sf2_v_trial

    return min_id, rec_sf2_h, rec_sf2_v, min_error


@njit
def search_for_initial_set_code(
    max_iter, ids, eq_id_dbf, im_h_dbf, sf1_h_dbf, select_v, im_v_dbf,
    sf1_v_dbf, sa_h_lower_lim, avg_sa_h_mean_ratio, target_spectrum_h,
    sa_v_lower_lim, min_sa_h_rec_target_ratio, max_sa_h_rec_target_ratio,
    lower_sf_lim_h, upper_sf_lim_h, lower_sf_lim_v, upper_sf_lim_v,
    target_spectrum_v, penalty, weights, num_records, max_record_per_event,
    error_metric, coupled_scaling, sa_h_upper_lim, sa_v_upper_lim, special_ids
):

    def mean(arr):
        """
        Computes the mean of a 2-D array along axis=0.
        Required for computations since njit is used as wrapper.
        """
        out = np.zeros(arr.shape[1])
        for i in range(arr.shape[1]):
            out[i] = arr[:, i].mean()

        return out

    for _ in range(max_iter):
        ids_sorted = np.random.permutation(ids)
        # Move the special id into begining
        if np.any(special_ids):
            for sid in special_ids:
                mask = ids_sorted != sid
                ids_sorted = ids_sorted[mask]
            ids_sorted = np.concatenate((special_ids, ids_sorted))

        ok = True
        rec_ids = np.zeros(num_records, dtype=np.int32)
        rec_eq_ids = np.zeros(num_records, dtype=np.int32)
        eqs = np.zeros(np.max(eq_id_dbf) + 1, dtype=np.int32)
        i = 0

        for rec_id in ids_sorted:
            eq = eq_id_dbf[rec_id]

            # Increment count for the current eq
            eqs[eq] += 1

            if eqs[eq] <= max_record_per_event:
                rec_ids[i] = rec_id
                rec_eq_ids[i] = eq
                i += 1

            if i >= num_records:
                break

        rec_im_h = im_h_dbf[rec_ids]
        rec_sf1_h = sf1_h_dbf[rec_ids]

        if select_v:
            rec_im_v = im_v_dbf[rec_ids]
            rec_sf1_v = sf1_v_dbf[rec_ids]

        rec_mean_h = mean(rec_sf1_h.reshape(-1, 1) * rec_im_h)
        rec_sf2_h = max(
            np.max(sa_h_lower_lim / rec_mean_h),
            avg_sa_h_mean_ratio / np.mean(rec_mean_h / target_spectrum_h),
            1.0,
        )

        if select_v:
            if np.any(sa_v_lower_lim):
                rec_mean_v = mean(rec_sf1_v.reshape(-1, 1) * rec_im_v)
                rec_sf2_v = max(np.max(sa_v_lower_lim / rec_mean_v), 1.0)
                if coupled_scaling:
                    rec_sf2_h = rec_sf2_v = max(rec_sf2_h, rec_sf2_v)
            else:
                rec_sf2_v = 1.0
        else:
            rec_sf2_v = 0.0

        rec_sf_h = rec_sf1_h * rec_sf2_h
        if select_v:
            rec_sf_v = rec_sf1_v * rec_sf2_v

        scaled_rec_im_h = rec_sf_h.reshape(-1, 1) * rec_im_h
        scaled_rec_im_h_mean = mean(scaled_rec_im_h)

        if error_metric == "rmse":
            error_h = np.mean((scaled_rec_im_h_mean - target_spectrum_h) ** 2
                              ) ** 0.5
        elif error_metric == "ssd":
            error_h = np.sum(
                (np.log(scaled_rec_im_h_mean) - np.log(target_spectrum_h)) ** 2
            )

        if select_v:
            error_h = weights[0] * error_h
            scaled_rec_im_v = rec_sf_v.reshape(-1, 1) * rec_im_v
            scaled_rec_im_v_mean = mean(scaled_rec_im_v)
            if error_metric == "rmse":
                error_v = np.mean(
                    (scaled_rec_im_v_mean - target_spectrum_v) ** 2
                ) ** 0.5
            elif error_metric == "ssd":
                error_v = np.sum(
                    (np.log(scaled_rec_im_v_mean) - np.log(target_spectrum_v)
                     ) ** 2
                )
            error_v = weights[1] * error_v
            error = error_h + error_v
        else:
            error = error_h

        if np.any(scaled_rec_im_h / target_spectrum_h > 2.0):
            count = np.sum(scaled_rec_im_h / target_spectrum_h > 2.0)
            error += count * penalty * error_h
        if np.any(scaled_rec_im_h / target_spectrum_h < 0.5):
            count = np.sum(scaled_rec_im_h / target_spectrum_h < 0.5)
            error += count * penalty * error_h
        if select_v:
            if np.any(scaled_rec_im_v / target_spectrum_v > 2.0):
                count = np.sum(scaled_rec_im_v / target_spectrum_v > 2.0)
                error += count * penalty * error_v
            if np.any(scaled_rec_im_v / target_spectrum_v < 0.5):
                count = np.sum(scaled_rec_im_v / target_spectrum_v < 0.5)
                error += count * penalty * error_v

        # Using & instead of * ensures Numba performs a bitwise AND operation
        # on the Boolean arrays, resulting in the expected element-wise
        # logical AND
        mask_h = (lower_sf_lim_h <= rec_sf_h) & (rec_sf_h <= upper_sf_lim_h)
        if select_v:
            mask_v = (lower_sf_lim_v <= rec_sf_v) \
                & (rec_sf_v <= upper_sf_lim_v)
            mask1 = ~(mask_h & mask_v)
        else:
            mask1 = ~mask_h

        if np.any(mask1):
            ok = False
        mask2 = scaled_rec_im_h / target_spectrum_h < min_sa_h_rec_target_ratio
        if np.any(mask2):
            ok = False
        mask3 = scaled_rec_im_h / target_spectrum_h > max_sa_h_rec_target_ratio
        if np.any(mask3):
            ok = False
        if np.any(sa_h_upper_lim):
            if np.any(sa_h_upper_lim - scaled_rec_im_h_mean < 0):
                ok = False
        if select_v and np.any(sa_v_upper_lim):
            if np.any(sa_v_upper_lim - scaled_rec_im_v_mean < 0):
                ok = False

        if ok:
            break

    return rec_ids, rec_eq_ids, rec_sf2_h, rec_sf2_v, error, ok
