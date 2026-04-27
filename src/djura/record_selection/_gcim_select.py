from typing import List
import numpy as np
from scipy.stats import lognorm, kstest, ksone, skew
from scipy.optimize import minimize
from statsmodels.distributions.empirical_distribution import ECDF
import re

from .constants import MECHANISM_MAP, SUPPORTED_IMS
from .numba_utils import greedy_algorithm
from .utilities import random_multivariate_normal
# from .utilities import compute_ks_error


class _GCIMSelect:
    NEGLIGIBLE = 1e-16
    scaling: bool = False
    conditional: bool = False

    # Error weights applied to realization mean, stddev and skewness
    # These weights can be changed/adjusted accordingly
    ERROR_WEIGHTS = np.array([1.0, 2.0, 0.3])

    # During the application of greedy algorithm to penalize selected
    # spectra more than 3 stddev from the target for any IMi with > 0
    PENALTY = 0

    # Number of points for the comparison of selected data with the limits
    N_P = 200

    # Large number
    LARGE = 1e8

    # Tolerance in %
    TOLERANCE = 10.0

    # Algorithm
    _algorithm = "greedy"

    def __init__(
        self,
        output_create: dict,
        metadata: dict,
    ) -> None:
        self.selected_scaled_total: dict = {}
        self.selected_scaled_best: dict = {}
        self.output_create = output_create
        self.metadata = metadata

        # By default use LHS random multivariate normal simulation
        self._simulate = self._simulate_realization_lhs

    def select(
        self,
        nreplicate: int,
        num_records: int,
        context_limits: dict,
        seed: int,
        ks_alpha: float,
        im_weights: List[float],
        max_scaling_factor: float,
        greedy_loops: int,
    ) -> None:
        """Ground motion record suite selection based on GCIM distribution

        Parameters
        ----------
        nreplicate : int
            Number of replicates
        num_records : int
            Number of records to select
        context_limits : dict
            Causal context value limits
        seed : int
            Random seed number
        ks_alpha : float
            KS test alpha
        im_weights : List[float]
            IMis and associated weights for selection prioritisation
        max_scaling_factor : float
            Maximum scaling factor allowed
        greedy_loops : int
            Number of greedy loops to run
        """

        nreplicate = int(nreplicate)
        num_records = int(num_records)
        seed = int(seed)
        ks_alpha = float(ks_alpha)
        max_scaling_factor = float(max_scaling_factor)
        greedy_loops = int(greedy_loops)

        # Prepare GCIM creator data
        target = self.output_create["target"]
        num_components = self.output_create["num_components"]
        component_definition = self.output_create["component_definition"]
        if num_components == 1:
            component_definition = "rotd50"
            self.output_create["component_definition"] = "rotd50"

        imi = target["IMi"]
        alpha_hash = self._get_alpha(list(imi.keys()))

        # Get IM* values from the database
        # If im-star is not present, means, unconditional seleciton
        # is being performed
        im_star_db = self._get_im_star_db()

        # The structure of the database is based on NGA-W2 metadata
        im_known, context, filename1, filename2, rsn, eq_id, allowed_idxs = \
            self._filter_database(imi, num_records, context_limits,
                                  num_components, component_definition)

        if im_star_db is not None:
            alpha_hash.update(self._get_alpha([
                self.output_create["im-star"]["type"]
            ]))

            im_star_db = im_star_db[allowed_idxs]

        total_rec = len(rsn)

        # Combine all IMi together
        imi_db, mu_imi, sigma_imi, correlations, alpha, covariance = \
            self._combine_imi(im_known, target, alpha_hash, total_rec)

        # number of IMs
        num_im = imi_db.shape[1]

        rec_context = {}
        for _arg in context.keys():
            rec_context[_arg] = np.zeros((nreplicate, num_records))

        # Generate random realizations of IMi (num_records, num_imi)
        realization = self._simulate(
            nreplicate, num_records, mu_imi, covariance, sigma_imi, seed)

        if self._algorithm == "bradley":
            self._algorithm_bradley(
                num_records, total_rec, num_im, imi_db, im_weights,
                realization, mu_imi, sigma_imi, alpha, max_scaling_factor,
                context, rec_context, rsn, ks_alpha)

            self.selected_scaled_best["IMi"] = imi
            return

        # Scaling factors
        if max_scaling_factor == 1.0:
            self.scaling = False
        else:
            self.scaling = True

        # Ln of IMi of the entire database
        ln_imi_db = np.log(imi_db)
        total_rec, num_im = ln_imi_db.shape

        # initialize
        sf = np.zeros((num_records, total_rec))
        unscaled_imi = np.zeros((num_records, num_im))
        scaled_imi = np.zeros((num_records, num_im))
        rec_id = np.zeros(num_records, dtype=int)
        rec_rsn = np.zeros(num_records, dtype=int)
        sf_opt = np.zeros(num_records)
        residuals = np.zeros((num_records, total_rec))
        residuals_opt = np.zeros(num_records)

        for rec in range(num_records):
            # Retrieve scaling factors for all prospective records
            sf[rec] = self._get_scaling_factors(
                im_star_db, ln_imi_db, total_rec, alpha, alpha_hash,
                realization[rec])

            # Filter based on requested scaling factor bounds
            residuals[rec] = self._compute_residuals(
                max_scaling_factor, sf[rec], residuals[rec],
                ln_imi_db, realization[rec], im_weights, alpha
            )

            # Select the record ID with the smallest residual
            # with respect to the realization
            _min_id = np.argmin(residuals[rec])

            if _min_id in rec_id:
                unique_value_found = False
                # to avoid repeated records
                _sorted_ids = np.argsort(residuals[rec])
                for value in _sorted_ids:
                    if value not in rec_id:
                        rec_id[rec] = value
                        unique_value_found = True
                        break

                if not unique_value_found:
                    # If all failed
                    rec_id[rec] = _min_id
            else:
                rec_id[rec] = _min_id

            # Get RSN
            rec_rsn[rec] = rsn[rec_id[rec]]

            # Get the corresponding scaling factor and minimum residual
            sf_opt[rec] = sf[rec][rec_id[rec]]
            residuals_opt[rec] = residuals[rec][rec_id[rec]]

            # Get the unscaled IMi, each entry has shape (n_imi,)
            unscaled_imi[rec] = ln_imi_db[rec_id[rec]]

            # Compute the scaled IMi, each entry has shape (n_imi,)
            scaled_imi[rec] = np.log(
                np.exp(unscaled_imi[rec]) * sf_opt[rec] ** alpha
            )

            # Set IMs of already selected records to a large value
            # to force very large residuals
            ln_imi_db[rec_id[rec], :] = 10.0

        # Redefine ln(imi)
        ln_imi_db = np.log(imi_db)

        # Compute the initial error
        im_weights = np.array(im_weights)
        dev_mean = (np.mean(scaled_imi, axis=0) - mu_imi) * im_weights
        dev_stddev = (np.std(scaled_imi, axis=0) - sigma_imi) * im_weights
        # dev_skew = np.nan_to_num(
        #     skew(scaled_imi, axis=0), nan=0.0) * im_weights
        dev_total = self.ERROR_WEIGHTS[0] * np.sum(dev_mean ** 2) \
            + self.ERROR_WEIGHTS[1] * np.sum(dev_stddev ** 2) \
            # + self.ERROR_WEIGHTS[2] * np.sum(dev_skew ** 2)

        # dev_total = compute_ks_error(
        #     scaled_imi, mu_imi, sigma_imi,
        #     im_weights=im_weights,
        #     error_weights=self.ERROR_WEIGHTS
        # )

        # Greedy subset modification algorithm
        for _ in range(greedy_loops):

            for rec in range(num_records):
                # -> this needs to be made more efficient
                sel_rec_id = rec_id[rec]
                scaled_imi = np.delete(scaled_imi, rec, 0)
                rec_id = np.delete(rec_id, rec)

                # Use records that are within the scaling factor bounds
                mask = (1 / max_scaling_factor <= sf[rec]) \
                    * (sf[rec] <= max_scaling_factor)
                filtered_ln_imi_db = ln_imi_db[mask]
                filtered_sf = sf[rec][mask]

                # Database indexes of corresponding records
                db_idxs = np.column_stack(np.where(mask))
                db_idxs = np.reshape(db_idxs, (db_idxs.shape[0], ))

                # filtered_ln_imi_db = ln_imi_db
                # filtered_sf = sf[rec]
                # db_idxs = np.arange(len(filtered_sf))
                sel_rec_id, dev_total = greedy_algorithm(
                    scaled_imi, filtered_sf, mu_imi, sigma_imi, rec_id,
                    filtered_ln_imi_db, num_records, self.ERROR_WEIGHTS,
                    self.PENALTY, db_idxs, alpha, im_weights,
                    sel_rec_id, dev_total
                )

                # Add the record information to the outputs
                sf_opt[rec] = sf[rec][sel_rec_id]

                # Update the sample IMi
                _idx = np.where(db_idxs == sel_rec_id)[0][0]

                scaled_imi = np.concatenate((
                    scaled_imi[:rec],
                    filtered_ln_imi_db[_idx].reshape(1, ln_imi_db.shape[1])
                    + np.log(sf_opt[rec] ** alpha),
                    scaled_imi[rec:]
                ), axis=0)

                rec_id = np.concatenate((
                    rec_id[:rec], np.array([sel_rec_id]), rec_id[rec:]
                ))

            # Estimate errors
            median_error, std_error = self._estimate_errors(
                scaled_imi, mu_imi, sigma_imi)

            if median_error <= self.TOLERANCE and std_error <= self.TOLERANCE:
                # If errors are within a tolerable limit, stop algorithm
                break

        # Get the unscaled IMis
        unscaled_imi = np.exp(ln_imi_db[rec_id])
        scaled_imi = np.exp(scaled_imi)

        # KS tests
        ks_tests = self._perform_ks_tests(
            mu_imi, sigma_imi, realization,
            unscaled_imi, scaled_imi, ks_alpha
        )

        # ECDFs
        scaled_imi_transposed = scaled_imi.transpose()

        ecdfx_scal = np.zeros((scaled_imi.shape[1], scaled_imi.shape[0] + 1))
        ecdfy_scal = np.zeros((scaled_imi.shape[1], scaled_imi.shape[0] + 1))

        for i, im in enumerate(scaled_imi_transposed):
            ecdfx_scal[i] = ECDF(im).x
            ecdfy_scal[i] = ECDF(im).y

        ecdfx_scal[np.isinf(ecdfx_scal)] = 0

        # Filenames
        rec_f1 = filename1[rec_id]

        rec_ids = []  # rec IDs in the metadata
        rec_comp = []
        rec_filenames = []
        for fname in rec_f1:
            if self.output_create["num_components"] == 3:
                idx = list(self.metadata['Filename_1']).index(fname)
                fname2 = self.metadata['Filename_2'][idx]
                fname3 = self.metadata['Filename_vert'][idx]
                rec_filenames.append(f"{fname}, {fname2}, {fname3}")
                rec_comp.append('H1, H2, V')
            elif self.output_create["num_components"] == 2:
                idx = list(self.metadata['Filename_1']).index(fname)
                fname2 = self.metadata['Filename_2'][idx]
                rec_filenames.append(f"{fname}, {fname2}")
                rec_comp.append('H1, H2')
            elif fname in self.metadata['Filename_1']:
                idx = list(self.metadata['Filename_1']).index(fname)
                rec_filenames.append(f"{fname}")
                rec_comp.append('H1')
            elif fname in self.metadata['Filename_2']:
                idx = list(self.metadata['Filename_2']).index(fname)
                rec_filenames.append(f"{fname}")
                rec_comp.append('H2')
            rec_ids.append(idx)

        # Mechanism
        mech = self.metadata['mechanism'][rec_ids]
        mech = [MECHANISM_MAP[x] for x in mech]

        low_freq_u = self.metadata['lowest_usable_freq'][rec_ids]

        # All selected records for export
        self.selected_scaled_total = {
            'IM_MeanLn': mu_imi,
            'IM_SigmaLn': sigma_imi,
            'context': rec_context,
            'Random_Realization': realization,
        }

        self.selected_scaled_best = {
            'IMi': imi,
            'sf': sf_opt,
            'rsn': rsn[rec_id],
            'eq_name': self.metadata['EQ_name'][rec_ids],
            'eq_year': self.metadata['EQ_year'][rec_ids],
            'station_name': self.metadata['Station_name'][rec_ids],
            'magnitude': self.metadata['magnitude'][rec_ids],
            'vs30': self.metadata['Vs30'][rec_ids],
            'rjb': self.metadata['Rjb'][rec_ids],
            'rrup': self.metadata['Rrup'][rec_ids],
            'mechanism': np.array(mech),
            'usable_max_period': np.divide(1.0, low_freq_u,
                                           out=np.full_like(low_freq_u, 100),
                                           where=low_freq_u != 0),
            'rec_filenames': np.array(rec_filenames),
            'gm_components': np.array(rec_comp),

            'Scaled_IMs': scaled_imi,
            'Rec_ID': rec_ids,
            'median_error': median_error,
            'std_error': std_error,
            'context': {},
            'ecdfx_scal': ecdfx_scal,
            'ecdfy_scal': ecdfy_scal,
            'duration': self.metadata['duration'][rec_ids],
            'dt': self.metadata['dt'][rec_ids],
            'im_idxs': target['im_idxs'],
        }

        for _arg, _val in rec_context.items():
            self.selected_scaled_best['context'][_arg + "_selecRec"] = _val

        self.selected_scaled_best.update(ks_tests)

    def _perform_ks_tests(
            self, mu_imi, sigma_imi, realization, unscaled_imi, scaled_imi,
            ks_alpha):

        num_records = realization.shape[0]
        num_im = len(sigma_imi)

        min_limit = np.zeros((num_im))
        max_limit = np.zeros((num_im))

        x_theoretical = np.zeros((num_im, self.N_P))
        cdf_theoretical = np.zeros((num_im, self.N_P))
        ksstat_sim = np.zeros((num_im))
        ksstat_scaled = np.zeros((num_im))
        p_sim = np.zeros((num_im))
        p_scaled = np.zeros((num_im))
        median_imi = np.exp(mu_imi)

        # Check the probabilities for min and max limit
        min_limit = np.minimum(
            lognorm(sigma_imi, scale=median_imi).ppf(0.001),
            np.min(np.exp(realization), axis=0))
        min_limit = np.minimum(min_limit, np.min(scaled_imi, axis=0))
        min_limit = np.minimum(min_limit, np.min(unscaled_imi, axis=0))

        max_limit = np.maximum(
            lognorm(sigma_imi, scale=median_imi).ppf(0.990),
            np.max(np.exp(realization), axis=0))
        max_limit = np.maximum(
            max_limit, np.max(scaled_imi, axis=0))
        max_limit = np.maximum(max_limit, np.max(unscaled_imi, axis=0))

        x_theoretical = np.linspace(
            min_limit, max_limit, self.N_P,
            axis=1)

        for im in range(num_im):
            cdf_theoretical[im, :] = \
                lognorm(sigma_imi[im], scale=(median_imi[im])).cdf(
                    x_theoretical[im, :])

            # Random realizations
            ksstat_sim[im], p_sim[im] = kstest(
                np.exp(realization[:, im]),
                lognorm(sigma_imi[im], scale=(median_imi[im])).cdf)

            # Selected scaled ground motions:
            ksstat_scaled[im], p_scaled[im] = \
                kstest(
                    scaled_imi[:, im],
                    lognorm(sigma_imi[im], scale=(median_imi[im])).cdf)

            # The critical values and the hypothesis test are
            # calculated after the for loops

        ks_critical_value = ksone.ppf(1 - ks_alpha / 2, num_records)
        hypothesis_rejection_sim = ksstat_sim > ks_critical_value
        hypothesis_rejection_scaled = ksstat_scaled > ks_critical_value

        test_results = {
            'ksstat_scaled': ksstat_scaled,
            'ks_critical_value': ks_critical_value,
            'ks_alpha': ks_alpha,
            'hypothesis_rejection_scaled': hypothesis_rejection_scaled,
            'hypothesis_rejection_sim': hypothesis_rejection_sim,
            'x_theoretical': x_theoretical,
            'CDF_theoretical': cdf_theoretical,
            'p_scaled': p_scaled,
        }

        return test_results

    def _estimate_errors(self, scaled_imi, mu_imi, sigma_imi):
        if self.conditional:
            # Ignore error at IM*
            # Assuming the lowest stddev at IM*
            idx = np.argmin(sigma_imi)

            scaled_imi = np.hstack((scaled_imi[:, :idx],
                                    scaled_imi[:, idx + 1:]))
            mu_imi = np.concatenate((mu_imi[:idx], mu_imi[idx + 1:]))
            sigma_imi = np.concatenate((sigma_imi[:idx], sigma_imi[idx + 1:]))

        mean = np.exp(np.mean(scaled_imi, axis=0))
        stddev = np.std(scaled_imi, axis=0)

        median_error = np.max(np.abs(
            mean - np.exp(mu_imi)) / np.exp(mu_imi)) * 100.0

        std_error = np.max(np.abs(
            stddev - sigma_imi) / sigma_imi) * 100.0

        return median_error, std_error

    def _compute_residuals(
        self, max_sf, sfs, residuals, imi, rel, im_weights, alpha
    ):
        mask = (1 / max_sf <= sfs) * (sfs <= max_sf)
        residuals[~mask] = self.LARGE

        residuals[mask] = np.sum(
            im_weights * np.square(np.log(
                np.exp(imi[mask]) * sfs[mask][:, np.newaxis] ** alpha)
                - rel),
            axis=1)

        return residuals

    def _get_scaling_factors(
            self, im_star_db, ln_imi_db, total_rec, alpha, alpha_hash, rel):

        if not self.scaling:
            return np.ones(total_rec)

        if self.conditional:
            _alpha = alpha_hash[self.output_create["im-star"]["type"]]

            if _alpha > 0:
                im_star = self.output_create["im-star"]["value"]
                return (im_star / im_star_db) ** (1 / _alpha)

        # IM* is independent of amplitude scaling or
        # unconditional target is used

        # Compute only on alpha == 1
        ln_imi_db = ln_imi_db[:, alpha == 1]
        rel = rel[alpha == 1]

        # Try to minimize residuals
        return np.sum(np.exp(ln_imi_db) * np.exp(rel), axis=1) / \
            np.sum(np.exp(ln_imi_db) ** 2, axis=1)

    def _get_im_star_db(self):
        if "im-star" not in self.output_create:
            self.conditional = False
            return

        self.conditional = True
        im_star = self.output_create["im-star"]

        name = im_star["type"]
        period = im_star.get("period", None)

        ims = self._get_im_component(
            name, self.output_create["num_components"],
            self.output_create["component_definition"])

        im_star_vals = self._parse_imi_of_interest(
            {name: ims}, {name: [period]})

        return im_star_vals[name].flatten()

    def _algorithm_bradley(
        self, num_records, total_rec, num_im, imi_db,
        im_weights, realization, mu_imi, sigma_imi, alpha,
        max_scaling_factor, context, rec_context, rsn, ks_alpha
    ):

        # Scaling factors
        sf = np.zeros((num_records, total_rec))
        residuals = np.zeros((num_records, total_rec))
        potential_imi = np.zeros((total_rec, num_im))
        residuals_opt = np.zeros((num_records))
        rec_id = np.zeros((num_records))
        sf_opt = np.zeros((num_records))
        selected_imi = np.zeros((num_records, num_im))
        scaled_imi = np.zeros((num_records, num_im))
        rec_rsn = np.zeros((num_records))

        # For each realization (to be selected)
        for rel in range(num_records):
            # For each available (prospective) record
            for rec in range(total_rec):
                # IMi for "rec"
                imi_trial = np.asarray(imi_db[rec, :])

                # Minimize the residual for each prospective ground
                # motion with respect to the applied scaling factor
                res = minimize(
                    self._optimize_selection, 1.0,
                    args=(im_weights, realization[rel, :], imi_trial,
                          sigma_imi, alpha),
                    bounds=([1 / max_scaling_factor, max_scaling_factor],))
                sf[rel, rec] = res.x[0]
                residuals[rel, rec] = res.fun
                potential_imi[rec, :] = imi_trial

            # Retrieve records with the lowest residual with respect to
            # GCIM realizations
            residuals_opt[rel] = min(residuals[rel, :])
            # Ground motion record unique IDs
            rec_id[rel] = np.argmin(residuals[rel, :])
            selected_rec_idx = int(rec_id[rel])
            # Get corresponding Scaling Factor (SF)
            sf_opt[rel] = sf[rel, selected_rec_idx]
            # Get corresponding IM values for each IM of interest
            selected_imi[rel, :] = potential_imi[selected_rec_idx, :]
            # Get corresponding scaled IM values for each IM of
            # interest
            scaled_imi[rel, :] = (sf_opt[rel] ** alpha) \
                * selected_imi[rel, :]

            # Get corresponding context
            for _arg, _val in context.items():
                rec_context[_arg][rel] = _val[selected_rec_idx]

            # RSN of records
            rec_rsn[rel] = rsn[selected_rec_idx]

            # Set IMs of already selected records to a large value
            # to force very large residuals
            imi_db[selected_rec_idx, :] = 999

        # KS tests
        ks_tests = self._perform_ks_tests(
            mu_imi, sigma_imi, realization, selected_imi, scaled_imi, ks_alpha
        )

        # ECDFs
        scaled_imi_transposed = scaled_imi.transpose()

        ecdfx_scal = np.zeros((scaled_imi.shape[1], scaled_imi.shape[0] + 1))
        ecdfy_scal = np.zeros((scaled_imi.shape[1], scaled_imi.shape[0] + 1))

        for i, im in enumerate(scaled_imi_transposed):
            ecdfx_scal[i] = ECDF(im).x
            ecdfy_scal[i] = ECDF(im).y

        ecdfx_scal[np.isinf(ecdfx_scal)] = 0

        # All selected records for export
        self.selected_scaled_total = {
            'IM_MeanLn': mu_imi,
            'IM_SigmaLn': sigma_imi,
            'residuals_opt': residuals_opt,
            'SF_opt': sf_opt,
            'Selected_IMs': selected_imi,
            'Scaled_IMs': scaled_imi,
            'RSN_selecRec': rec_rsn.astype(int),
            'context': rec_context,
            'Random_Realization': realization,
        }

        self.selected_scaled_best = {
            'residuals_opt': residuals_opt,
            'SF_opt': sf_opt,
            'Selected_IMs': selected_imi,
            'Scaled_IMs': scaled_imi,
            'ecdfx_scal': ecdfx_scal,
            'ecdfy_scal': ecdfy_scal,
            'RSN_selecRec': rec_rsn,
            'Random_Realization': realization,
            'context': {}
        }

        for _arg, _val in rec_context.items():
            self.selected_scaled_best['context'][_arg + "_selecRec"] = _val

        self.selected_scaled_best.update(ks_tests)

    @staticmethod
    def _get_alpha(imi: List):
        """Assigns alphas for scaling depending IMi type

        Parameters
        ----------
        imi : List
            IMi types

        Returns
        -------
        dict
            IMi-alpha pairs
        """
        alpha = {}
        for name in imi:
            if name in alpha:
                continue

            if name == 'PGA':
                alpha[name] = 1
            elif name == 'PGV':
                alpha[name] = 1
            elif name == "FIV3":
                alpha[name] = 1
            elif name.startswith('SA') or name.startswith("Sa"):
                alpha[name] = 1
            elif name == 'ASI':
                alpha[name] = 1
            elif name == 'SI':
                alpha[name] = 1
            elif name == 'DSI':
                alpha[name] = 1
            elif name == "IA":      # AI - Arias Intensity
                alpha[name] = 2
            elif name == 'CAV':
                alpha[name] = 1
            elif name == 'Ds575':
                alpha[name] = 0
            elif name == 'Ds595':
                alpha[name] = 0
            # No else condition, to force Error and avoid silent failure
        return alpha

    @staticmethod
    def _optimize_selection(scale_factor, im_weights, realization, im_trial,
                            sigma, alpha):
        """Optimizes selection by minimizing the residuals based
        on the applied scale factor

        Parameters
        ----------
        scale_factor : float
            Scale factor
        im_weights : numpy.ndarray
            Weights of IMi
        realization : numpy.ndarray
            Random realization
        im_trial : numpy.ndarray
            IM values from metadata for each prospective record
        sigma : numpy.ndarray
            Stdevs of IMi
        alpha : numpy.ndarray
            Alpha factors for each IMi

        Returns
        -------
        float
            Residual value to be minimised
        """
        f = np.sum(
            im_weights * ((realization - np.log(
                (scale_factor[0]**alpha) * im_trial)) / sigma)**2)
        return f

    def _combine_imi(self, im_known: dict, target: dict, alpha: dict,
                     total_rec: int):
        """Combines IMi values, as well as GMM predictions into
        global numpy ndarrays

        Parameters
        ----------
        im_known : dict
            IMi and corresponding values from the metadata
        target : dict
            Target GCIM distribution obtained from create()
        alpha : dict
            Alpha coefficients for each IM type
        total_rec : int
            Total available records for selection

        Returns
        -------
        np.ndarrays
            Numpy ndarrays for IMi values, prediction means and stdevs,
            correlation matrices and alphas
        """

        # Indices of IMi
        im_idxs = target["im_idxs"]
        n_imi = 0
        for _idxs in im_idxs.values():
            n_imi += len(_idxs)

        # Initialize
        imi_db = np.zeros((total_rec, n_imi))
        means = np.zeros(n_imi)
        sigmas = np.zeros(n_imi)

        alpha_proc = []
        for _im, _idxs in im_idxs.items():
            values = im_known[_im]

            if len(values.shape) == 1:
                # Period-independent IMis
                values = values[:, np.newaxis]

            imi_db[:, _idxs] = values
            means[_idxs] = np.asarray(target["mu_lnIMi"][_im])
            sigmas[_idxs] = np.asarray(target["sigma_lnIMi"][_im])

            # Alpha factors
            alpha_proc += [alpha[_im]] * len(_idxs)

        # Combined correlations
        # \rho_{lnIM|Rup,IMj} for conditional case
        # \rho_{lnIM|Rup} for unconditional case
        correlations = np.asarray(target["combined_correlations"])

        # Combined covariance
        covariance, sigmas = self._compute_exact_covariance(
            correlations, sigmas)

        alpha_proc = np.asarray(alpha_proc)

        return imi_db, means, sigmas, correlations, alpha_proc, covariance

    def _compute_exact_covariance(self, corr, sigmas):
        cov = np.zeros(corr.shape)
        for i in range(corr.shape[0]):
            for j in range(corr.shape[1]):
                cov[i, j] = corr[i, j] * sigmas[i] * sigmas[j]

        # Making sure that cov is a positive semi-definite matrix
        # min_eig = np.min(np.real(np.linalg.eigvals(cov)))
        w, _ = np.linalg.eigh(cov)
        min_eig = np.min(w)
        if min_eig < 0:
            cov -= 2 * min_eig * np.eye(*cov.shape)
            # Add a small multiple of the identity matrix
            # cov += self.NEGLIGIBLE * np.eye(*cov.shape)
        # sigmas = np.sqrt(np.diagonal(cov))

        return cov, sigmas

    def _simulate_realization_lhs(
            self, nreplicate, num_records, mu, cov, sigma, seed):
        """Generates random IMi realization from considered IM_nsim

        Parameters
        ----------
        nreplicate : int
            Number of replicates (simulations)
        num_records : int
            Number of records to be selected
        mu : np.ndarray
            Means
        cov : np.ndarray
            Covariance matrix for all IMi
        sigma : np.ndarray
            Stdevs
        seed : int
            Random seed

        Returns
        -------
        np.ndarray
            Selected random realization with smallest deviation
            from nreplicate realizations (num_records, num_imi)
        """

        # Set initial seed for simulation
        np.random.seed(seed)

        total_deviation = np.zeros((nreplicate, 1))
        realizations = np.zeros((nreplicate, num_records, len(mu)))

        for rep in range(nreplicate):
            _rel = np.exp(random_multivariate_normal(
                mu, cov, num_records, "LHS"
            ))

            # Mean vs target mean
            dev_mean = np.mean(np.log(_rel), axis=0) - mu
            # Stddev vs target stddev
            dev_stddev = np.std(np.log(_rel), axis=0) - sigma
            # skewness vs target skewness (i.e., zero)
            dev_skew = skew(np.log(_rel), axis=0)
            # combine the three error metrics to compute a total error
            total_dev = self.ERROR_WEIGHTS[0] * np.sum(dev_mean ** 2) + \
                self.ERROR_WEIGHTS[1] * np.sum(dev_stddev ** 2) + \
                (self.ERROR_WEIGHTS[2]) * np.sum(dev_skew ** 2)

            total_deviation[rep] = total_dev
            realizations[rep] = _rel

        # Get the realization with minimal error
        rel_id = np.argmin(np.abs(total_deviation))

        rel = np.log(realizations[rel_id])

        return rel

    def _get_im_component(self, name, num_components, component_definition):
        """Gets IM component values

        Parameters
        ----------
        name : str
            Name of IM
        num_components : int
            Number of components
        component_definition : str
            Component definition, RotD50 or RotD100,
            applicable to SA type IMs only

        Returns
        -------
        np.ndarray
            IM values

        Raises
        ------
        ValueError
            If given IM name is not supported in the metadata
        """
        if name not in SUPPORTED_IMS:
            raise ValueError(
                f"Intensity measure (IM) {name} is not supported. Supported "
                f"IMs include: {SUPPORTED_IMS}"
            )

        if name.startswith("SA") and component_definition:
            component_definition = component_definition.lower()
        elif component_definition:
            component_definition = "geomean"

        im1 = f"{name}_1"
        im2 = f"{name}_2"

        if im1 not in self.metadata and num_components == 2:
            # period-independent IMs
            return self.metadata[name]

        if im1 not in self.metadata and num_components == 1:
            # period-independent IMs
            return np.append(self.metadata[name], self.metadata[name], axis=0)

        if num_components == 1:
            im_vals = np.append(
                self.metadata[im1], self.metadata[im2], axis=0)

            return im_vals

        # 2 components
        if component_definition == "geomean":
            im_vals = np.sqrt(self.metadata[im1] * self.metadata[im2])
        elif component_definition == 'srss':
            im_vals = np.sqrt(self.metadata[im1] ** 2
                              + self.metadata[im2] ** 2)
        elif component_definition == 'arithmeticmean':
            im_vals = (self.metadata[im1] + self.metadata[im2]) / 2
        elif component_definition == 'rotd50':
            im_vals = self.metadata[f'{name}_RotD50']
        elif component_definition == 'rotd100':
            im_vals = self.metadata[f'{name}_RotD100']
        else:
            raise ValueError(
                f"Wrong component definition: {component_definition}")

        return im_vals

    def _get_imi_database(
            self, imi, num_components, component_definition):
        """Loops for each IMi and gets the corresponding values from the
        metadata

        Parameters
        ----------
        imi : dict
            IMis
        num_components : int
            Number of components
        component_definition : str
            Component definition

        Returns
        -------
        dict
            IMis and corresponding values
        """
        im_known = {}
        for name in imi.keys():
            im_known[name] = self._get_im_component(
                name, num_components, component_definition)

        return im_known

    def _analyze_database(self, unique_key, num_components,
                          component_definition, context_limits, imi):
        """Analyze the metadata file

        Parameters
        ----------
        unique_key : str
            Unique key distinguishing the ground motion record
        num_components : int
            Number of components of ground motion to consider
        component_definition : str
            Component definition
        context_limits : dict
            Limits on causal context
        imi : dict
            IMi types and corresponding periods where applicable

        Returns
        -------
        dict
            IMi types and corresponding values queried from the metadata
        np.ndarray
            Unique identifiers of ground motion records
        dict
            Causal context from metadata
        np.ndarray
            Filename of record in 1st primary direction
        np.ndarray
            Filename of record in 2nd primary direction
        np.ndarray
            Earthquake identifier, an earthquake might have multiple
            ground motion recordings
        Raises
        ------
        ValueError
            If number of components is neither 0 or 1
        """
        # sa_known is from arbitrary ground motion component
        filename2 = None
        context = {}
        if num_components == 1:

            filename1 = np.append(
                self.metadata['Filename_1'], self.metadata['Filename_2'],
                axis=0)
            eq_id = np.append(
                self.metadata['EQID'], self.metadata['EQID'], axis=0)

            rsn = np.append(
                self.metadata[unique_key], self.metadata[unique_key], axis=0)

            for key, val in context_limits.items():
                if val is None or key not in self.metadata.keys() \
                        or all(_val is None for _val in val):
                    continue

                context[key] = np.append(
                    self.metadata[key], self.metadata[key], axis=0)

        elif num_components == 2:

            component_definition = component_definition.lower()
            filename1 = self.metadata['Filename_1']
            filename2 = self.metadata['Filename_2']
            eq_id = self.metadata['EQID']
            rsn = self.metadata[unique_key]

            for key, val in context_limits.items():
                if val is None or key not in self.metadata.keys() \
                        or all(_val is None for _val in val):
                    continue

                context[key] = self.metadata[key]

        else:
            raise ValueError(
                'Selection can only be performed for one or two components at '
                'the moment, exiting...')

        im_known = self._get_imi_database(
            imi, num_components, component_definition)

        return im_known, rsn, context, filename1, filename2, eq_id

    @staticmethod
    def _create_mask_for_strings(arr, search_string):
        """Create a boolean mask where True indicates the element contains
        any of the search terms (case-insensitive).

        Parameters
        ----------
        arr : np.ndarray
            numpy array of strings and/or numbers
        search_string : string
            String with terms separated by semicolon, or period
        """
        search_terms = [
            term.strip().lower()
            for term in re.split('[;.]', search_string)
            if term.strip()
        ]

        # convert array to lowercase strings for comparison
        arr_lower = np.char.lower(arr.astype(str))

        # Create the mask by checking if any search
        # term is in each element
        mask = np.zeros(len(arr), dtype=bool)
        for term in search_terms:
            found = np.char.find(arr_lower, term) == 0
            mask = mask | found

        return mask

    def _limit_context(self, context, context_limits, im_known, rsn):
        """Create a list of RSNs of ground motion records to disregard
        during selection based on causal context limits imposed

        Parameters
        ----------
        context : dict
            Causal context
        context_limits : dict
            Causal context limits
        im_known : dict
            IMis and respective values for each ground motion
        rsn : np.ndarray
            Ground motion unqiue identifiers

        Returns
        -------
        List
            Ground motion with unqiue identifiers to be ignored
        """
        not_allowed = []
        mask = np.zeros(len(rsn), dtype=bool)
        for im_vals in im_known.values():
            not_allowed.extend(np.unique(np.where(im_vals <= 0)[0]).tolist())

        for name, val in context.items():
            _limit = context_limits[name]

            if len(_limit) == 0:
                continue

            if name == "mechanism":
                _mech_values = np.array(
                    [_mech['value'] for _mech in _limit], dtype=np.int64)
                mask = np.isin(val.astype(np.int64), _mech_values)
            elif name == "EQ_name":
                # Ignore Earthquakes that match the key value provided
                mask = self._create_mask_for_strings(
                    val, _limit
                )
                mask = ~mask
            else:
                if _limit[0] == _limit[1] == "":
                    continue

                if _limit[0] == "" or _limit[0] is None:
                    _limit[0] = 0

                if _limit[1] == "" or _limit[1] is None:
                    _limit[1] = 1e5

                _limit = np.array(_limit, dtype=float)
                mask = (val >= min(_limit)) * (val <= max(_limit))

            temp = np.where(~mask)[0]
            not_allowed.extend(temp)

        return not_allowed

    def _filter_database(self, imi, num_records, context_limits,
                         num_components, component_definition):
        """Searches the database and does the filtering

        Parameters
        -------
        imi : dict
            IMi types and corresponding periods where applicable
        num_records : int
            Number of records to be selected
        context_limits : dict
            Limits on causal context
        num_components : int
            Number of components of ground motion to consider
        component_definition : str
            Component definition

        Returns
        -------
        dict
            IMi types and corresponding values queried from the metadata
        dict
            Causal context from metadata
        np.ndarray
            Filename of record in 1st primary direction
        np.ndarray
            Filename of record in 2nd primary direction
        np.ndarray
            Unique identifiers of ground motion records (RSNs)
        np.ndarray
            Earthquake identifier, an earthquake might have multiple
            ground motion recordings
        np.ndarray
            Allowed record indices, unique indices of records allowed
            for the selection

        Raises
        ------
        ValueError
            Unexpected Sa definition, exiting... Wrong spectrum definition
        ValueError
            Wrong number of components. Selection can only be performed
            for one or two components at the moment, exiting...
        ValueError
            NaNs found in input response spectra
        ValueError
            There are not enough records which satisfy, the given record
            selection criteria...Please broaden your selection criteria...
        """

        unique_key = "RSN"

        im_known, rsn, context, filename1, filename2, eq_id = \
            self._analyze_database(
                unique_key, num_components, component_definition,
                context_limits, imi)

        # Limiting the records to be considered using the
        # `not_allowed' variable
        # IM values cannot be negative or zero, remove those
        not_allowed = self._limit_context(
            context, context_limits, im_known, rsn)

        # Initialize indices for all available records
        all_indexes = set(range(len(rsn)))

        # get the unique values
        not_allowed = set(not_allowed)

        # Allowed set of indices
        allowed = np.array(list(all_indexes - not_allowed))

        # Use only allowed records
        for key, val in im_known.items():
            if len(val.shape) > 1:
                im_known[key] = val[allowed, :]
            else:
                im_known[key] = val[allowed]

        for key, val in context.items():
            context[key] = val[allowed]

        eq_id = eq_id[allowed]
        filename1 = filename1[allowed]
        rsn = rsn[allowed]

        if filename2 is not None:
            filename2 = filename2[allowed]

        # Arrange the available spectra in a usable format and check
        # for invalid input
        # Match periods (known periods and periods for error computations)
        im_known = self._parse_imi_of_interest(im_known, imi)

        if num_records > len(eq_id):
            raise ValueError('There are not enough records which satisfy',
                             'the given record selection criteria...',
                             'Please use broaden your selection criteria...')

        return im_known, context, filename1, filename2, rsn, eq_id, allowed

    def _parse_imi_of_interest(self, im_known: dict, imi: dict):
        """Select the IMi matching at periods of interest only

        Parameters
        ----------
        im_known : dict
            IMi and respective IM values from metadata
        imi : dict
            IMi and respective periods of interest

        Returns
        -------
        dict
            Reduced IMi and respective IM values from metadata

        Raises
        ------
        ValueError
            NaNs found in input response spectra in metadata file
        """

        for im, vals in im_known.items():
            # Loop over each IM type
            # If vals has two dimensions, then it is period-dependent
            # otherwise, period-independent, so skip
            if len(vals.shape) == 1:
                continue

            period_key = "Periods_Sa_avg" if im.startswith("Sa_avg") else \
                f"Periods_{im}"

            im_idx = []
            for period in imi[im]:
                if period == 0.0:
                    period = min(self.metadata[period_key])

                period = np.round(period, 5)
                self.metadata[period_key] = np.round(
                    self.metadata[period_key], 5)

                im_idx.append(
                    np.where(self.metadata[period_key] == period)[0][0])

            im_known[im] = vals[:, im_idx]

            # Check for invalid input
            if np.any(np.isnan(im_known[im])):
                raise ValueError('NaNs found in input response spectra')

        return im_known
