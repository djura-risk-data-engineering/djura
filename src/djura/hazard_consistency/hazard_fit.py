from typing import List
import numpy as np
from scipy.optimize import leastsq
from scipy.interpolate import interp1d

from ..record_selection.utilities import find_nearest


def analytical_mafe(s: float, k0: float, k1: float, k2: float = 0.) -> float:
    """Compute mean annual frequency of exceedance (MAFE) of an
    intensity measure (IM) value based on SAC/FEMA-compatible coefficients

    Parameters
    ----------
    s : float
        Intensity measure level
    k0 : float
        SAC/FEMA-compatible coefficient
    k1 : float
        SAC/FEMA-compatible coefficient
    k2 : float
        SAC/FEMA-compatible coefficient, by default 0.

    Returns
    -------
    float
        Mean annual frequency of exceeding an IM value (MAFE)
    """
    mafe = k0 * np.exp(-k2 * np.power(np.log(s), 2) - k1 * np.log(s))
    return mafe


def solve_for_im(
    s: float, mafe: float, k0: float, k1: float, k2: float = 0.,
) -> float:
    # to avoid zeros in log
    s = np.array(s) + 1e-8

    mafe_computed = analytical_mafe(s, k0, k1, k2)

    return abs(mafe_computed - mafe)


def error_function(x, s, a):
    return np.log(a) - np.log(x[0] * np.exp(-x[2] * np.power(np.log(s), 2)
                                            - x[1] * np.log(s)))


class HazardFit:

    def __init__(self, im_range: List, s: List, mafe: List) -> None:
        """Initialize Hazard modeller

        Parameters
        ----------
        im_range : List
            Intensity measure (IM) range where fitting is carried out,
            avoid using 0.0g
        s : List
            IM range of seismic hazard
        mafe : List
            Mean annual frequency of exceeding (MAFE) an IM value
        """
        self.im_range = np.array(im_range)
        self.s = s
        self.mafe = mafe

    def _into_json_serializable(self, mafe_fit, coef):
        """Hazard data into a json serializable variable

        Parameters
        ----------
        s : List
            Range of intensity measures
        mafe : List
            Range of MAFE
        mafe_fit : List
            Range of fitted MAFE
        coef : List[float]
            SAC/FEMA-compatible coefficients, [k0, k1, k2]

        Returns
        -------
        dict
        """
        info = {
            "s": list(self.s),
            "mafe": list(self.mafe),
            "s_fit": list(self.im_range),
            "mafe_fit": list(mafe_fit),
            "coef": list(coef),
        }
        return info

    def analytical_sac_fema(self, return_periods: List[float] = None):
        """SAC/FEMA-compatible Hazard fitting function

        Parameters
        ----------
        return_periods : List[float]
            List of 3 integer values where the fitting will be prioritized,
            by default selects return periods at indices [0, 3, 9]

        Returns
        -------
        dict:
            s : List
                Original IM range
            mafe : List
                Original MAFE
            s_fit : List
                Fitted IM range
            mafe_fit : List
                Fitted MAFE
            coef : List[float]
                SAC/FEMA-compatible coefficients, [k0, k1, k2]
            return_periods : List[float]
        """
        if return_periods is None:
            # Assign default values
            max_idx = 9 if len(self.mafe) > 9 else len(self.mafe) - 1
            indices_to_fit_at = [0, 3, max_idx]
            return_periods = 1 / self.mafe[indices_to_fit_at]

        else:
            # Select MAFE based on proximity
            return_periods = np.array(return_periods)
            mafes = 1 / return_periods
            indices_to_fit_at = find_nearest(self.mafe, mafes)
            return_periods = 1 / self.mafe[indices_to_fit_at]

        if len(return_periods) != 3:
            raise ValueError("You must provide three return periods!")

        # Fitting the hazard curves
        coef = np.zeros(3)

        # select iterator depending on where we want to have a better fit
        r = np.zeros((3, 3))
        a = np.zeros(3)
        cnt = 0
        for i in indices_to_fit_at:
            r_temp = np.array([1])
            for j in range(1, 3):
                r_temp = np.append(r_temp, -np.power(np.log(self.s[i]), j))

            r[cnt] = r_temp
            a[cnt] = self.mafe[i]
            del r_temp
            cnt += 1

        temp1 = np.log(a)
        temp2 = np.linalg.inv(r).dot(temp1)
        temp2 = temp2.tolist()

        coef[0] = np.exp(temp2[0])
        coef[1] = temp2[1]
        coef[2] = temp2[2]

        mafe_fit = analytical_mafe(self.im_range, *coef)

        data = self._into_json_serializable(mafe_fit, coef)
        data['return_periods'] = list(return_periods)
        data['method'] = 'analytical-sac-fema'

        return data

    def least_squares(self):
        """Hazard fitting function using least squares method

        Returns
        -------
        dict:
            s : List
                Original IM range
            mafe : List
                Original MAFE
            s_fit : List
                Fitted IM range
            mafe_fit : List
                Fitted MAFE
            coef : List[float]
                SAC/FEMA-compatible coefficients, [k0, k1, k2]
        """
        x0 = np.array([0.1, 0.1, 0.1])

        self.mafe[self.mafe == 0] = 1e-12
        p = leastsq(error_function, x0, args=(
            self.s, self.mafe), factor=1)[0]

        mafe_fit = analytical_mafe(self.im_range, *p)
        data = self._into_json_serializable(mafe_fit, p)
        data['method'] = 'least-squares'

        return data

    def power_law(self, dbe: float = 465., mce: float = 10000.):
        """Performs fitting on a loglinear power law constrained at
        two return periods

        Parameters
        -------
        dbe : float
            First conditioning return period, by default 465
        mce : float
            Second conditioning return period, by default 10000

        Returns
        -------
        dict:
            s : List
                Original IM range
            mafe : List
                Original MAFE
            s_fit : List
                Fitted IM range
            mafe_fit : List
                Fitted MAFE
            coef : List[float]
                SAC/FEMA-compatible coefficients for a first-order law,
                [k0, k1]
        """
        # get constraining intensity levels
        mafe_dbe = 1 / dbe
        mafe_mce = 1 / mce

        interpolation = interp1d(self.mafe, self.s)
        s_dbe = interpolation(mafe_dbe)
        s_mce = interpolation(mafe_mce)

        # Get the fitting coefficients
        k = np.log(mafe_dbe / mafe_mce) / np.log(s_mce / s_dbe)
        k0 = mafe_dbe * s_dbe ** k

        # Fitted MAFE
        coef = [k0, k]
        mafe_fit = analytical_mafe(self.im_range, *coef)

        data = self._into_json_serializable(mafe_fit, coef)
        data['method'] = 'power-law'

        return data

    def analytical(self):
        """Performs fitting following the approach of Bradley et al., 2008

        Returns
        -------
        dict:
            s : List
                Original IM range
            mafe : List
                Original MAFE
            s_fit : List
                Fitted IM range
            mafe_fit : List
                Fitted MAFE
            coef : List[float]
                Fitting coefficients [H_asy, s_asy, alpha]
        """

        def func(x, s, a):
            mafe_asy = x[0]
            s_asy = x[1]
            alpha = x[2]
            return np.log(a) - np.log(mafe_asy * np.exp(alpha * (
                np.log(s / s_asy)) ** -1))

        x0 = np.array([100, 100, 50])

        p = leastsq(func, x0, args=(
            self.s, self.mafe), factor=100)[0]
        mafe_fit = p[0] * np.exp(p[2] * (np.log(self.im_range / p[1])) ** -1)

        data = self._into_json_serializable(mafe_fit, p)
        data['method'] = 'analytical'

        return data
