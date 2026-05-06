# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026 Djura | Risk - Data - Engineering S.r.l.
from typing import List
from pathlib import Path
from datetime import datetime
import json
import inspect
import warnings
import bisect
import re
import shutil

import numpy as np
from scipy.stats import lognorm, kstest, norm, binom
from scipy.optimize import curve_fit, minimize

from ..utilities import (     # noqa: F401 (re-exported)
    to_json_serializable, export_results, read_json)


def random_multivariate_normal(
        mu: np.ndarray, cov: np.ndarray,
        num_samples: int, sampling_option: str) -> np.ndarray:
    """Used to generate multivariate correlated normal samples

    References
    ----------
    Yang, T. Y., Moehle, J., Stojadinovic, B., & Der Kiureghian, A. (2009).
    Seismic Performance Evaluation of Facilities: Methodology and
    Implementation. In Journal of Structural Engineering
    (Vol. 135, Issue 10, pp. 1146-1154).
    American Society of Civil Engineers (ASCE).
    https://doi.org/10.1061/(asce)0733-9445(2009)135:10(1146)

    Parameters
    ----------
    mu : numpy.ndarray (1-D)
        Mean value vector
    cov : numpy.ndarray (2-D)
        Covariance matrix
    num_samples : int
        number of samples
    sampling_option : str
        Monte Carlo Sampling: 'MCS'
        Latin Hypercube Sampling: 'LHS'

    Returns
    -------
    numpy.ndarray (num_samples x num_dimensions)
        Array which contains randomly generated numbers between 0 and 1
    """

    from scipy.stats import norm

    num_dimensions = len(mu)
    if mu.size == mu.shape[0]:
        mu = mu.reshape(-1, 1)
    my = mu @ np.ones([1, num_samples])
    eigen_values, eigen_vectors = np.linalg.eigh(cov)
    # In case there is a negative value
    eigen_values = np.clip(eigen_values, 0, None)
    # The lower-triangular decomposition of the correlation matrix
    ly = eigen_vectors
    # Standard deviations
    dy = np.diag(eigen_values ** 0.5)
    # Generate uniformly distributed between 0 and 1
    u = random_uniform(num_dimensions, num_samples, sampling_option)
    # Compute standard random numbers
    u = norm(loc=0, scale=1).ppf(u)
    # Create realization matrix (Eqn. 4) - @ is the matrix multiplication
    z = (ly @ dy @ u.T + my).T

    return z


def random_uniform(
    num_dimensions: int, num_samples: int, sampling_type: str
) -> np.ndarray:
    """Used to perform sampling based on Monte Carlo Simulation or
    Latin Hypercube Sampling

    References
    ----------
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.LatinHypercube.html#scipy.stats.qmc.LatinHypercube

    Parameters
    ----------
    num_dimensions : int
        number of dimensions
    num_samples : int
        number of samples
    sampling_type : str
        type of sampling.
        Monte Carlo Sampling: 'MCS'
        Latin Hypercube Sampling: 'LHS'

    Returns
    -------
    numpy.ndarray (num_samples x num_dimensions)
        Array which contains randomly generated numbers between 0 and 1
    """
    from scipy.stats.qmc import LatinHypercube

    # Not really required, but will ensure different realizations each time
    seed = int(datetime.today().strftime("%H%M%S"))
    if sampling_type.lower() == 'mcs':
        # Do Monte Carlo Sampling without any grid
        np.random.seed(seed)
        sample = np.random.uniform(size=[num_dimensions, num_samples]).T
    elif sampling_type.lower() == 'lhs':
        # A Latin hypercube sample generates n points in [0, 1)^d.
        # Each univariate marginal distribution is stratified, placing exactly
        # one point in each possible grid.
        sampler = LatinHypercube(d=num_dimensions, seed=seed)
        sample = sampler.random(n=num_samples)

    return sample


def get_list_id(elements, key, val, name):
    generator = (ele for ele in elements if ele.get(key) == val)

    try:
        element = next(generator)
        return element
    except StopIteration:
        raise ValueError(
            f"{name} with {key} {val} not found!\n"
            "Double-check input keys 'gmms' and 'ruptures'")


def find_right_index(arr: List, value: float):
    """Get index where the value can be inserted
    into the array while maintaining increasing order
    using bisect_right

    Parameters
    ----------
    arr : List
        Sorted array
    value : float
        Value to be inserted

    Returns
    -------
    int
        Index
    """
    return bisect.bisect_right(arr, value)


def get_periods_ims(arr: List):
    im_types = [s.split('(', 1)[0].strip() if '(' in s else s for s in arr]

    pattern = r"\((\d+(?:\.\d+)?)\)?"

    periods = []
    for i, im in enumerate(arr):
        search = re.search(pattern, im)

        if im_types[i] == "PGA":
            periods.append(0)
            continue

        if search:
            periods.append(float(re.search(pattern, im).group(1)))
        else:
            periods.append(None)

    return im_types, periods


def get_period_im(name: str):
    # pattern = r"\((\d+\.\d+)\)?"
    pattern = r"\((\d+(\.\d+)?)\)?(?:\D*)?"

    if '(' in name:
        im_type = name.split('(', 1)[0].strip()
    else:
        im_type = name

    if re.search(pattern, name):
        period = float(re.search(pattern, name).group(1))
    else:
        period = None

    return im_type, period


def select_func_args(function, data: dict):
    sig = inspect.signature(function)

    func_params = [param.name for param in sig.parameters.values()
                   if param.name != 'self']

    return {k: v for k, v in data.items()
            if k in func_params}


def select_function(module, function_name):
    if hasattr(module, function_name) and \
            callable(getattr(module, function_name)):
        return getattr(module, function_name)
    else:
        return None


def interpolate_2d(
        x, y, data, x_int, y_int, bounds_error: bool = False,
        fill_value: float = None, message: str = ""):
    from scipy.interpolate import RegularGridInterpolator

    interp = RegularGridInterpolator(
        (x, y), data, bounds_error=bounds_error, fill_value=fill_value)

    if not (min(x) <= x_int <= max(x)):
        warnings.warn(
            f"{message} value not within interpolation range, "
            "extrapolating...")

    if not (min(y) <= y_int <= max(y)):
        warnings.warn(
            f"{message} value not within interpolation range, "
            "extrapolating...")

    val = interp((x_int, y_int))

    if val > 1:
        return 1
    elif val < -1:
        return -1

    return interp((x_int, y_int))


def inspect_file_for_classes(module):
    classes = inspect.getmembers(module, inspect.isclass)

    # defined_classes = [
    #     cls for cls in classes if inspect.getmodule(cls[1]) == module]
    class_names = [cls[0] for cls in classes]

    return class_names


def remove_path(directory: Path):
    """Removes the directory if it exists

    Parameters
    ----------
    directory : Path
        Directory to be removed
    """
    if directory.is_dir():
        shutil.rmtree(directory)


def create_path(directory: Path):
    """Create a folder if it does not exist

    Parameters
    ----------
    directory : Path
        Directory to be created
    """
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError:
        print("Error: Creating directory. ", directory)


def list_to_ndarray(data, keys):
    for key, val in data.items():
        if key in keys:
            data[key] = np.asarray(val)
    return data


def find_nearest(array: List, value: float) -> List[int]:
    """Find index of nearest value in array

    Parameters
    ----------
    array : List
    value : float

    Returns
    -------
    List[int]
        Index of nearest value
    """
    value = np.asarray(value)
    array = np.asarray(array)
    idx = np.abs(array - value[:, np.newaxis]).argmin(axis=1)
    return idx


def proc_oq_hazard_curve(
    poes: list[float],
    path_hazard_results: str | Path,
    json_file: str | Path = 'hazard.json',
    haz_file_start: str = 'hazard_curve-mean'
) -> None:
    """
    Process OpenQuake hazard curve results and store them in a JSON file.

    This function processes the hazard curve files generated by OpenQuake
    after performing Probabilistic Seismic Hazard Analysis (PSHA). For each
    hazard curve file, it interpolates the intensity measure levels (IMLs)
    corresponding to specified probabilities of exceedance (poes). The
    processed data, including the site location, investigation time, hazard
    curves, and interpolated IMLs, is then saved into a JSON file.

    Parameters
    ----------
    poes : list[float]
        List of probabilities of exceedance in a given investigation time
        (`investigation_time`) for which corresponding intensity measure
        levels (IMLs) will be obtained via interpolation.
    path_hazard_results : str | Path
        Path to the directory containing the OpenQuake hazard curve results.
    json_file : str | Path, optional
        Path and name of the output JSON file where the processed data will
        be saved. By default, it is saved as 'hazard.json'.
    haz_file_start : str, optional
        Prefix for the hazard curve files to process. Only files that start
        with this prefix will be processed. By default, this is set to
        'hazard_curve-mean'.

    Returns
    -------
    None

    Notes
    -----
    The output JSON file will contain the following keys:

    - `investigation_time`: The investigation time used in the PSHA
    (extracted from the hazard files).
    - `lat`: List of latitudes of the investigated sites.
    - `lon`: List of longitudes of the investigated sites.
    - `im`: List of intensity measure types (IMs) extracted from the hazard
    curve file names.
    - `cond_poes`: The provided list of probabilities of exceedance for which
    IMLs are calculated.
    - `cond_imls`: A dictionary containing interpolated IMLs corresponding to
    each IM for the given poes.
    - `hazard_curves`: A dictionary containing hazard curve data for each IM,
    including the original probabilities of exceedance and IMLs.

    Example
    -------
    To process hazard curve files in a directory and save them in a JSON file,
    run:

    >>> proc_oq_hazard_curve([0.1, 0.5], 'path/to/results', 'outputs.json')

    This will process hazard curve files from the directory `path/to/results`
    and save the results in `outputs.json`.
    """
    from scipy.interpolate import interp1d
    from pandas import read_csv

    # Convert paths to Path objects
    path_hazard_results = Path(path_hazard_results)
    json_file = Path(json_file)

    # Initialise dictionary to store all outputs
    output_data = {
        "investigation_time": None,  # Will be set when reading files
        "lat": [],
        "lon": [],
        "im": [],
        "cond_poes": poes,
        "cond_imls": {},
        "hazard_curves": {}
    }

    # Read through each file in the outputs folder
    for file in path_hazard_results.iterdir():
        if file.name.startswith(haz_file_start):

            # Strip the IM out of the file name
            items = (file.stem.split('-')[2]).split('_')
            im_type = "_".join(items[:-1])  # Join later for Sa_Avg

            # Load the results in as a dataframe
            df = read_csv(file, skiprows=1)

            # Get the column headers
            iml = list(df.columns.values)[3:]  # List of headers
            iml = [float(i[4:]) for i in iml]  # Strip out the actual IM values

            with file.open("r") as f:
                temp1 = f.readline().split(',')
                temp2 = list(filter(None, temp1))
                inv_t = float(list(filter(
                    lambda x: 'investigation_time=' in x, temp2
                ))[0].replace(" investigation_time=", ""))

                # Save inv_t once (assuming it is consistent across files)
                output_data["investigation_time"] = inv_t

            # For each of the sites investigated
            for site in np.arange(len(df)):

                # Append each site's info to the output dictionary
                output_data["lat"].append([df.lat[site]][0])
                output_data["lon"].append([df.lon[site]][0])
                output_data["im"].append(im_type)

                # Get the array of poe in inv_t and corresponding imls
                tmp1 = np.array(df.iloc[site, 3:].values)
                tmp2 = np.array(iml)
                # get rid of any infinite or nan value
                infs = np.isinf(tmp1)
                tmp1 = tmp1[~infs]
                tmp2 = tmp2[~infs]
                nans = np.isnan(tmp1)
                tmp1 = tmp1[~nans]
                tmp2 = tmp2[~nans]

                # Save hazard curve data in the dictionary
                output_data["hazard_curves"][f"{im_type}"] = {
                    "poe": tmp1.tolist(),
                    "iml": tmp2.tolist(),
                }

    # Get intensity measure levels corresponding to poes and store in
    # dictionary
    for idx, im in enumerate(output_data["hazard_curves"]):
        poe = output_data["hazard_curves"][im]["poe"]
        iml = output_data["hazard_curves"][im]["iml"]
        iml_interp = interp1d(poe, iml, kind='linear')(poes)
        output_data["cond_imls"][output_data["im"][idx]] = iml_interp.tolist()

    # Save the output dictionary as a JSON file
    if json_file:
        with open(json_file, 'w') as file:
            json.dump(output_data, file, indent=4)

    return output_data


def proc_oq_disaggregation_exc(
    path_disagg_results: str | Path,
    json_file: str | Path = 'disaggregation.json',
    disagg_file_start: str = 'Mag_Dist'
) -> dict:
    """
    Process disaggregation results from OpenQuake and store them in a JSON
    file.

    This function reads disaggregation data from OpenQuake results, including
    magnitudes (M) and distances (R) for different probabilities of exceedance
    (poes) and intensity measure types (IMTs). It calculates key metrics such
    as mean and modal magnitudes and distances, and stores the processed data
    in a JSON file for easy access and further analysis.

    Parameters
    ----------
    path_disagg_results : str | Path
        The directory containing the disaggregation result files produced by
        OpenQuake.
    json_file : str | Path, optional
        The output JSON file where the processed disaggregation data will be
        stored.
        Default is 'disaggregation.json'.
    disagg_file_start : str, optional
        Prefix of the disaggregation files to process. Files that begin with
        this prefix and do not contain 'Mag_Dist_Eps' will be processed.
        Default is 'Mag_Dist'.

    Returns
    -------
    None

    Notes
    -----
    The function processes each disaggregation result file to extract the
    following:

    - `location`: Latitude and longitude of the site.
    - `investigation_time`: The investigation time used in the disaggregation.
    - `imt_disagg`: A dictionary where each intensity measure type (IMT)
    contains:
        - `poes`: List of probabilities of exceedance (poes) used in the
        disaggregation.
        - `return_periods`: Corresponding return periods for each poe.
        - `mean_mags`: List of mean magnitudes for each poe.
        - `mean_dists`: List of mean distances for each poe.
        - `mod_mags`: List of modal magnitudes (magnitude with the highest
        hazard contribution).
        - `mod_dists`: List of modal distances (distance with the highest
        hazard contribution).
        - `mag_dist_hazard_contributions`: A dictionary of hazard
        contributions for each poe,
          containing magnitudes, distances, and their respective hazard
          contributions.

    Example
    -------
    To process disaggregation results stored in the directory 'results/disagg'
    and save them in 'disagg_output.json', you can run:

    >>> proc_oq_disaggregation('results/disagg', 'disagg_output.json')

    This will generate a JSON file with all the processed disaggregation data.
    """
    from pandas import read_csv, DataFrame

    # Convert paths to Path objects
    path_disagg_results = Path(path_disagg_results)

    # Initialize dictionary to store results
    disagg_data = {
        "location": {"lat": None, "lon": None},
        "investigation_time": None,
        "imt_disagg": {}
    }

    for file in path_disagg_results.iterdir():
        if file.name.startswith(disagg_file_start) and \
           'eps' not in file.name.lower():
            # Load the dataframe
            df = read_csv(file, skiprows=1)

            # Extract hazard key (column starting with 'rlz' or 'mean')
            hz_key = next(key for key in df.keys()
                          if key.startswith('rlz') or key == 'mean')

            # Extract unique values for poes and imt
            poes = np.unique(df['poe']).tolist()
            poes.sort(reverse=True)
            ims = np.unique(df['imt'])

            # Extract salient information from the first line of the file
            with file.open("r") as f:
                first_line = f.readline().split(',')
                lon = float(next(filter(lambda x: 'lon=' in x, first_line)
                                 ).replace(" lon=", ""))
                lat = float(next(filter(lambda x: 'lat=' in x, first_line)
                                 ).replace(" lat=", "").replace("\"\n", ""))
                inv_t = float(next(filter(
                    lambda x: 'investigation_time=' in x, first_line
                )).replace(" investigation_time=", ""))

                # Set lat, lon, and investigation time in the dictionary (once)
                disagg_data["location"]["lat"] = lat
                disagg_data["location"]["lon"] = lon
                disagg_data["investigation_time"] = inv_t

            # Loop through each intensity measure (imt)
            for imt in ims:
                disagg_data["imt_disagg"][imt] = {
                    "poes": poes,
                    "return_periods": [],
                    "mean_mags": [],
                    "mean_dists": [],
                    "mod_mags": [],
                    "mod_dists": [],
                    "mag_dist_hazard_contributions": {},
                }

                # Loop through each probability of exceedance (poe)
                for poe in poes:
                    return_period = round(-inv_t / np.log(1 - poe))
                    disagg_data["imt_disagg"][imt]["return_periods"].append(
                        return_period)

                    # Filter data for current poe and imt
                    mag_data = df['mag'][(
                        df['poe'] == poe) & (df['imt'] == imt)]
                    dist_data = df['dist'][
                        (df['poe'] == poe) & (df['imt'] == imt)]
                    hz_cont_data = df[hz_key][
                        (df['poe'] == poe) & (df['imt'] == imt)]
                    # Normalize hazard contribution
                    hz_cont_data_norm = hz_cont_data / hz_cont_data.sum()

                    # Create a DataFrame to hold the magnitude, distance,
                    # and hazard contribution
                    data = DataFrame({
                        "mag": mag_data,
                        "dist": dist_data,
                        "hz_cont": hz_cont_data_norm
                    })

                    # Compute modal (highest hazard contribution) values
                    mode = data.sort_values(by='hz_cont', ascending=False
                                            ).iloc[0]
                    mode_mag = mode['mag']
                    mode_dist = mode['dist']

                    # Compute mean values
                    mean_mag = np.sum(data['mag'] * data['hz_cont'])
                    mean_dist = np.sum(data['dist'] * data['hz_cont'])

                    disagg_data["imt_disagg"][imt]["mean_mags"].append(
                        mean_mag)
                    disagg_data["imt_disagg"][imt]["mean_dists"].append(
                        mean_dist)
                    disagg_data["imt_disagg"][imt]["mod_mags"].append(
                        mode_mag)
                    disagg_data["imt_disagg"][imt]["mod_dists"].append(
                        mode_dist)

                    # Store magnitude, distance, and hazard contribution in
                    # the dictionary
                    disagg_data["imt_disagg"][imt][
                        "mag_dist_hazard_contributions"][f"poe_{poe}"] = {
                        "mag": mag_data.tolist(),
                        "dist": dist_data.tolist(),
                        "hz_cont": hz_cont_data_norm.tolist(),
                        "gamma": hz_cont_data.tolist()
                    }
    if json_file:
        json_file = Path(json_file)
        # Save the output dictionary as a JSON file
        with open(json_file, 'w') as file:
            json.dump(disagg_data, file, indent=4)

    return disagg_data


def proc_oq_disaggregation_occ(
    poes: List[float],
    path_disagg_results: str | Path,
    json_file: str | Path = 'disaggregation.json',
    disagg_file_start: str = 'Mag_Dist',
    tol: float = 0.05
) -> None:

    disagg = proc_oq_disaggregation_exc(
        path_disagg_results, None, disagg_file_start
    )

    imts = list(disagg["imt_disagg"].keys())

    for imt in imts:
        data = disagg["imt_disagg"][imt]
        for target in poes:
            closest = [v for v in data["poes"] if v < target]
            if not closest:
                raise ValueError(
                    f"There is no smaller PoE than {target} "
                    "in disaggregation."
                )
            else:
                closest = max(closest)
                if (target - closest) / target > tol:
                    raise ValueError(
                        f"Add a close PoE to {target} to disaggregation PoEs"
                    )
            gamma_i = np.array(
                data["mag_dist_hazard_contributions"][f"poe_{target}"]["gamma"]
            )
            gamma_i_1 = np.array(
                data["mag_dist_hazard_contributions"][f"poe_{closest}"][
                    "gamma"
                ]
            )
            prob_im_m_r = (gamma_i_1 - gamma_i) / (
                np.sum(gamma_i_1) - np.sum(gamma_i)
            )
            # not required but let's make sure
            prob_im_m_r /= np.sum(prob_im_m_r)
            disagg["imt_disagg"][imt]["mag_dist_hazard_contributions"][
                f"poe_{target}"
            ]["prob_occur"] = list(prob_im_m_r)

    if json_file:
        # Save the output dictionary as a JSON file
        with open(json_file, 'w') as file:
            json.dump(disagg, file, indent=4)

    return disagg


def proc_oq_disaggregation(
    path_disagg_results: str | Path,
    poes: List[float] = None,
    json_file: str | Path = "disaggregation.json",
    disagg_file_start: str = "Mag_Dist",
    tol: float = 0.05,
) -> dict:

    if poes:
        disagg = proc_oq_disaggregation_occ(
            poes, path_disagg_results, json_file, disagg_file_start, tol
        )
    else:
        disagg = proc_oq_disaggregation_exc(
            path_disagg_results, json_file, disagg_file_start
        )

    imts = list(disagg["imt_disagg"].keys())
    for imt in imts:
        data = disagg["imt_disagg"][imt]["mag_dist_hazard_contributions"]
        poe_keys = list(data.keys())
        for poe in poe_keys:
            keys_to_filter = ["mag", "dist", "hz_cont", "gamma"]
            if "prob_occur" in data[poe]:
                valid_indices = [i for i, hz in
                                 enumerate(data[poe]["prob_occur"])
                                 if hz != 0]
                keys_to_filter.append("prob_occur")
            else:
                valid_indices = [i for i, hz in enumerate(data[poe]["hz_cont"])
                                 if hz != 0]
            # Filter all relevant keys using valid indices
            for key in keys_to_filter:
                data[poe][key] = [data[poe][key][i] for i in valid_indices]

        disagg["imt_disagg"][imt]["mag_dist_hazard_contributions"] = data

    return disagg


def get_rs_imi_intensities(
    selection_dir: Path,
    poes: List[float],
    imi: str
):
    imls = []
    for poe in poes:
        with open(selection_dir / f"records_{poe}.json") as f:
            records = json.load(f)

        records = records['selected_scaled_best']
        if '(' in imi:
            im_type = imi.split('(')[0]
            period = float(imi.split('(')[1].split(')')[0])
            period_idx = records['IMi'][im_type].index(period)
            imi_idx = records['im_idxs'][im_type][period_idx]
        else:
            imi_idx = records['im_idxs'][imi]
        imls.append(np.asarray(records['Scaled_IMs'])[:, imi_idx].flatten())

    imls = np.asarray(imls)
    return imls


def prepare_rs_for_hzc(
    selection_dir: Path,
    poes: List[float],
    imts: List[str],
):
    """Prepare input Record selection intensity values for
    hazard consistency checks

    Parameters
    ----------
    selection_dir : Path
        Directory of selected record json outputs following record selector of
        Djura
    poes : List[float]
        List of POEs of interest
    imts : List[str]
        List of intensity measure types of interest

    Returns
    -------
    Dict[numpy.ndarray]
        IM values of selected records
    """
    rs = {}
    for imi in imts:
        rs[imi] = get_rs_imi_intensities(
            selection_dir, poes, imi
        )
    return rs


def compute_ks_error(scaled_imi, mu_imi, sigma_imi, im_weights, error_weights):
    dev_total = 0.0
    for i in range(scaled_imi.shape[1]):
        # Work in exp(IMI) space if needed (i.e., lognormal)
        data = np.exp(scaled_imi[:, i])
        mu = np.exp(mu_imi[i])
        sigma = sigma_imi[i]

        # Theoretical lognormal CDF
        s = sigma
        scale = mu

        def cdf(x):
            return lognorm.cdf(x, s=s, scale=scale)

        # KS test (D-statistic only)
        D_statistic = kstest(data, cdf)[0]

        # Weighted error accumulation
        dev_total += im_weights[i] * D_statistic

    return dev_total


def mlefit(param1: float, param2: float, total_count: int, count: int,
           data) -> float:
    """Maximum likelihood method
    Performs a lognormal cumulative distribution function fit to the data
    points based on maximum likelihood method

    Parameters
    ----------
    param1 : float
        Median of the function, parameter of a statistical model to be found
    param2 : float
        Standard deviation of the function, parameter of a statistical model
        to be found
    total_count : int
        Number of data points
    count : int
        Number of failures
    data: Union[List, np.ndarray]
        The function, data points

    Returns
    -------
    float
        Negative Log likelihood to be minimized
    """
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    try:
        p = norm.cdf(np.log(data), loc=np.log(param1), scale=param2)

        likelihood = binom.pmf(count, total_count, p)
        likelihood[likelihood == 0] = 1e-290
        loglik = -sum(np.log10(likelihood))

        warnings.resetwarnings()

        return loglik
    except OverflowError:
        warnings.resetwarnings()
        return 1e+8
    except Exception:
        warnings.resetwarnings()
        return 1e+8


def neg_log_likelihood(params, data):
    p1, p2 = params
    if p2 <= 0:
        return np.inf
    likelihood = lognorm.logpdf(
        data, s=p2, scale=p1
    )
    loglik = -np.sum(likelihood)

    return loglik


def fit_cdf_to_data(
    x_data, y_data, distribution="lognormal", method="mle"
):
    """Fit a parametric CDF to empirical CDF data

    Parameters
    ----------
    x : array-like
        X-axis values (support of the distribution)
    y : array-like
        Y-axis values (CDF values between 0 and 1)
    distribution : str, optional
        'lognormal', 'normal', 'weibull', or 'exponential',
        by default 'lognormal'
    method : str, optional
        'least_squares' or 'mle' (maximum likelihood estimation)
        by default 'mle'
    """
    warnings_list = []

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        x_data = np.asarray(x_data)
        y_data = np.asarray(y_data)

        # Remove any invalid data points
        valid_idx = (x_data > 0) & (y_data >= 0) & (y_data <= 1)
        x_data = x_data[valid_idx]
        y_data = y_data[valid_idx]

        if distribution.lower() == "lognormal":
            if method.lower() == "least_squares":
                def lognormal_cdf(x, mu, sigma):
                    return lognorm.cdf(x, s=sigma, scale=np.exp(mu))

                log_x_median = np.log(np.median(x_data))
                log_x_std = np.std(np.log(x_data))
                p0 = [log_x_median, log_x_std]

                try:
                    params, pcov = curve_fit(
                        lognormal_cdf, x_data, y_data, p0=p0,
                        bounds=([-np.inf, 0.001], [np.inf, 10])
                    )
                    mu_fit, sigma_fit = params
                    perr = np.sqrt(np.diag(pcov))
                except Exception as e:
                    print(f"Curve fit failed: {e}")
                    return None

            elif method.lower() == 'mle':
                data = np.interp(np.random.rand(1000),
                                 y_data, x_data)

                x0 = [
                    np.log(np.mean(data)), np.std(np.log(data))
                ]

                result = minimize(
                    neg_log_likelihood, x0, args=(data, )
                )

                sigma_fit = result.x[1]
                mu_fit = result.x[0]
                perr = [np.nan, np.nan]

            else:
                raise ValueError(
                    "Incorrect method, must be 'mle' or 'least_squares'")

            y_fit = lognorm.cdf(x_data, s=sigma_fit, scale=np.exp(mu_fit))
            result = {
                'distribution': 'lognormal',
                'mu': mu_fit,
                'sigma': sigma_fit,
                'mu_stderr': perr[0] if method == 'least_squares' else np.nan,
                'sigma_stderr': perr[1] if method == 'least_squares'
                else np.nan,
                'y_fitted': y_fit,
                'x_data': x_data,
                'y_data': y_data
            }

        elif distribution.lower() == "normal":
            def normal_cdf(x, mu, sigma):
                return norm.cdf(x, loc=mu, scale=sigma)

            p0 = [np.mean(x_data), np.std(x_data)]
            params, pcov = curve_fit(normal_cdf, x_data, y_data, p0=p0)
            mu_fit, sigma_fit = params
            y_fit = norm.cdf(x_data, loc=mu_fit, scale=sigma_fit)

            result = {
                'distribution': 'normal',
                'mu': mu_fit,
                'sigma': sigma_fit,
                'y_fitted': y_fit,
                'x_data': x_data,
                'y_data': y_data
            }

        else:
            raise ValueError(
                "Incorrect distribution type, must be 'lognormal' or 'normal'")

        # Goodness-of-fit metrics
        residuals = y_data - y_fit
        result['rmse'] = np.sqrt(np.mean(residuals**2))
        result['mae'] = np.mean(np.abs(residuals))
        result['max_error'] = np.max(np.abs(residuals))

        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_data - np.mean(y_data))**2)
        result['r_squared'] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Store warnings in dictionary
        for warn in w:
            warnings_list.append({
                "message": str(warn.message),
                "category": warn.category.__name__,
                "filename": warn.filename,
                "lineno": warn.lineno
            })

    result["warnings"] = warnings_list

    return result
