from typing import List
from pathlib import Path
import json
import numpy as np
from scipy.optimize import fsolve

from .hazard_fit import analytical_mafe, HazardFit
from .models import HazardModelSchema


class HazardModel:
    def __init__(self, method: str = 'analytical-sac-fema') -> None:
        """Initialize Hazard module

        Parameters
        ----------
        method : str
            Method to fit the hazard data, by default 'analytical-sac-fema'
                'analytical-sac-fema' - analytical approach (recommended)
                'least-squares' - least squares fitting method
                'power-law' - loglinear power law constrained at
                two intensity levels
                'analytical' - analytical approach, not SAC/FEMA-compatible
        """
        if method == "":
            method = "analytical-sac-fema"

        self.method = method.lower()

    def _remove_zeros(self, x, y) -> np.array:
        """Removes trailing zeros to avoid bias in fitting

        Parameters
        ----------
        x : List
        y : List

        Returns
        -------
        ndarray, ndarray
        """
        x = np.array(x)
        y = np.array(y)

        x = x[y > 1.00000000e-12]
        y = y[y > 1.00000000e-12]
        return x, y

    def read_hazard(self, filename: Path):
        """Reads provided hazard data

        Parameters
        -------
        filename : Path, "*.txt", "*.json"

        Returns
        -------
        data : dict
            s: list
                intensity measure values in [g]
            poe: list
                Probability of exceedance (PoE)
        """
        extension = filename.suffix

        if extension == ".txt":
            s, poes = np.transpose(np.loadtxt(filename))
            data = {'s': list(s), 'poe': list(poes)}

        elif extension == ".json":
            data = json.load(open(filename))

        else:
            raise ValueError("Extension of hazard file not supported or file "
                             "format is wrong! Supported options: "
                             ".pickle, .json")

        return data

    def fit_hazard_model(
            self, s: List, mafe: List, iterator: List[int] = None,
            dbe: float = 475., mce: float = 10000.) -> HazardModelSchema:
        """Runs the fitting function

        Parameters
        -------
            s : List
                intensity measure values in [g]
            mafe : List
                Mean annual frequency of exceeding an IM value (MAFE)
            iterator : List[int]
                List of 3 integer values where the fitting will be prioritized
                Necessary only for method="analytical-sac-fema"
            dbe : float
                First return period
                Only for haz_fit='power-law'
            mce : float
                Second return period
                Only for haz_fit='power-law'

        Returns
        -------
        HazardModelSchema
            s : List
                Original IM range
            mafe : List
                Original MAFE
            s_fit : List
                Fitted IM range
            mafe_fit : List
                Fitted MAFE
            coef : List[float]
                if method is 'analytical':
                    Fitting coefficients [H_asy, s_asy, alpha]
                if method is 'power-law':
                    SAC/FEMA-compatible coefficients, [k0, k1]
                else:
                    SAC/FEMA-compatible coefficients, [k0, k1, k2]
            method : str
                Fitting method name
        """
        # Range of IM values, where fitting will be performed
        im_range = np.linspace(min(s), max(s), 1000)

        # Initialize model
        model = HazardFit(im_range, s, mafe)

        # Get rid of trailing zeros
        s, mafe = self._remove_zeros(s, mafe)

        if self.method == 'analytical-sac-fema':
            out = model.analytical_sac_fema(iterator)
        elif self.method == 'least-squares':
            out = model.least_squares()
        elif self.method == 'power-law':
            out = model.power_law(dbe, mce)
        elif self.method == 'analytical':
            out = model.analytical()
        else:
            raise ValueError('Wrong hazard fitting model!')

        return out

    @staticmethod
    def _to_ndarray(data):
        if isinstance(data, list):
            return np.asarray(data)
        return data

    def _return_period_to_poe(self, return_period, investigation_time):
        return 1 - np.exp(-investigation_time
                          / self._to_ndarray(return_period))

    def _return_period_to_apoe(self, return_period):
        return 1 - np.exp(-1 / self._to_ndarray(return_period))

    def _mafe_to_apoe(self, mafe):
        return 1 - np.exp(-self._to_ndarray(mafe))

    def _apoe_to_mafe(self, apoe):
        return -np.log(1 - self._to_ndarray(apoe))

    def _poe_to_mafe(self, poe, investigation_time):
        return -np.log(1 - self._to_ndarray(poe)) / investigation_time

    def _return_period_to_mafe(self, return_period):
        return 1 / self._to_ndarray(return_period)

    def _mafe_to_return_period(self, mafe):
        return 1 / self._to_ndarray(mafe)

    def _mafe_to_poe(self, mafe, investigation_time):
        return 1 - np.exp(-self._to_ndarray(mafe) * investigation_time)

    def get_return_period(
            self, poe: float = None, apoe: float = None,
            mafe: float = None, investigation_time: float = 50.) -> float:
        """Get Return Period

        Parameters
        ----------
        poe : float, optional
            Probability of exceedance (POE), by default None
        apoe : float, optional
            Annual probability of exceedance (APOE), by default None
        mafe : float, optional
            Mean annual frequency of exceeding (MAFE) an IM value,
            by default None
        investigation_time : float, optional
            Investigation time in years, by default 50

        Returns
        -------
        float
            Return period

        Raises
        ------
        ValueError
            Must provide one of the following input arguments: poe, apoe, mafe
        """
        if mafe is not None:
            return self._mafe_to_return_period(mafe)

        if apoe is not None:
            mafe = self._apoe_to_mafe(apoe)
            return self._mafe_to_return_period(mafe)

        if poe is not None:
            mafe = self._poe_to_mafe(poe, investigation_time)
            return self._mafe_to_return_period(mafe)

        if poe is None and apoe is None and mafe is None:
            raise ValueError("Must provide one of the following "
                             "input arguments: poe, apoe, mafe")

    def get_poe(
            self, return_period: float = None, apoe: float = None,
            mafe: float = None, investigation_time: float = 50.) -> float:
        """Get probability of exceedance (POE)

        Parameters
        ----------
        return_period : float, optional
            Return period, by default None
        apoe : float, optional
            Annual probability of exceedance (APOE), by default None
        mafe : float, optional
            Mean annual frequency of exceeding (MAFE) an IM value,
            by default None
        investigation_time : float, optional
            Investigation time in years, by default 50

        Returns
        -------
        float
            Probability of exceedance (POE)

        Raises
        ------
        ValueError
            Must provide one of the following input arguments:
            return_period, apoe, mafe
        """
        if return_period is not None:
            return self._return_period_to_poe(return_period,
                                              investigation_time)

        if apoe is not None:
            mafe = self._apoe_to_mafe(apoe)
            return self._mafe_to_poe(mafe, investigation_time)

        if mafe is not None:
            return self._mafe_to_poe(mafe, investigation_time)

        if return_period is None and apoe is None and mafe is None:
            raise ValueError(
                "Must provide one of the following input arguments:"
                " return_period, apoe, mafe")

    def get_apoe(self, return_period: float = None, poe: float = None,
                 mafe: float = None, investigation_time: float = 50.) -> float:
        """Get annual probability of exceedance (APOE)

        Parameters
        ----------
        return_period : float, optional
            Return period, by default None
        poe : float, optional
            Probability of exceedance (POE), by default None
        mafe : float, optional
            Mean annual frequency of exceeding (MAFE) an IM value,
            by default None
        investigation_time : float, optional
            Investigation time in years, by default 50

        Returns
        -------
        float
            Annual probability of exceedance (APOE)

        Raises
        ------
        ValueError
            Must provide one of the following input arguments:
            return_period, poe, mafe
        """
        if return_period is not None:
            return self._return_period_to_apoe(return_period)

        if poe is not None:
            mafe = self._poe_to_mafe(poe, investigation_time)
            return self._mafe_to_apoe(mafe)

        if mafe is not None:
            return self._mafe_to_apoe(mafe)

        if return_period is None and poe is None and mafe is None:
            raise ValueError(
                "Must provide one of the following input arguments: "
                "return_period, poe, mafe")

    def get_mafe(
            self, return_period: float = None, poe: float = None,
            apoe: float = None, investigation_time: float = 50.) -> float:
        """Get mean annual frequency (rate) of exceeding (MAFE) an IM value

        Parameters
        ----------
        return_period : float, optional
            Return period, by default None
        poe : float, optional
            Probability of exceedance (POE), by default None
        apoe : float, optional
            Annual probability of exceedance (APOE), by default None
        investigation_time : float, optional
            Investigation time in years, by default 50

        Returns
        -------
        float
            Mean annual frequency of exceeding (MAFE) an IM value

        Raises
        ------
        ValueError
            Must provide one of the following input arguments:
            return_period, apoe, poe
        """
        if return_period is not None:
            return self._return_period_to_mafe(return_period)

        if apoe is not None:
            return self._apoe_to_mafe(apoe)

        if poe is not None:
            return self._poe_to_mafe(poe, investigation_time)

        if return_period is None and apoe is None and poe is None:
            raise ValueError(
                "Must provide one of the following input arguments:"
                " return_period, apoe, poe")

    def get_mafe_limit_state(
            self, k0: float, k1: float, k2: float = 0.0,
            s: float = None, return_period: float = None, beta: float = 0.0
    ) -> float:
        """Derive mean annual frequency of exceeding a limit state

        Parameters
        ----------
        k0 : float
            SAC/FEMA-compatible coefficient
        k1 : float
            SAC/FEMA-compatible coefficient
        k2 : float
            SAC/FEMA-compatible coefficient, by default 0.
        s : float, optional
            Intensity measure level, by default None
        return_period : float, optional
            Return period level, by default None
        beta: float, optional
            Aleatory uncertainty, by default 0.0

        Returns
        -------
        float
            Mean annual frequency of exceedance of a limit state

        Raises
        -------
        ValueError
            If both 's' and 'return_period' are None
        """
        if k2 is None:
            k2 = 0.0

        if s is None and return_period is None:
            raise ValueError("You must provide 's' or 'return_period'")

        # Compute MAFE of IM
        if return_period is not None:
            mafe = 1 / return_period
        else:
            mafe = analytical_mafe(s, k0, k1, k2)

        # Compute mean annual frequency of exceedance of a limit state
        p = 1 / (1 + 2 * k2 * beta**2)

        mafe_ls = np.sqrt(p) * k0**(1 - p) * mafe**p * \
            np.exp(1 / 2 * p * k1**2 * beta**2)

        return mafe_ls

    def get_im(
            self, mafe: float, k0: float, k1: float, k2: float = 0.) -> float:
        """Get intensity measure (IM) value

        Parameters
        ----------
        mafe : float
            Mean annual frequency of exceeding (MAFE) an IM value
        k0 : float
            SAC/FEMA-compatible coefficient
        k1 : float
            SAC/FEMA-compatible coefficient
        k2 : float
            SAC/FEMA-compatible coefficient, by default 0.

        Returns
        -------
        float
            intensity measure (IM)
        """
        if k2 is None:
            k2 = 0.0

        def func(x):
            return mafe - analytical_mafe(x, k0, k1, k2)

        s = fsolve(func, x0=0.1)
        return float(s)

    def get_im_ls(
        self, mafe_ls: float, k0: float, k1: float, k2: float = 0.,
        beta: float = 0.,
    ) -> float:
        """Get intensity measure (IM) value for Limit state

        Parameters
        ----------
        mafe : float
            Mean annual frequency of exceeding (MAFE) an IM value
        k0 : float
            SAC/FEMA-compatible coefficient
        k1 : float
            SAC/FEMA-compatible coefficient
        k2 : float
            SAC/FEMA-compatible coefficient, by default 0.

        Returns
        -------
        float
            intensity measure (IM)
        """
        if k2 is None:
            k2 = 0.0

        def func(x):
            return mafe_ls - self.get_mafe_limit_state(
                k0, k1, k2, s=x, beta=beta)

        s = fsolve(func, x0=0.1)
        return float(s)

    def get_beta_ls(
        self, mafe_ls: float, k0: float, k1: float, k2: float = 0.,
        mu: float = 0.,
    ) -> float:
        """Get intensity measure (IM) value for Limit state

        Parameters
        ----------
        mafe : float
            Mean annual frequency of exceeding (MAFE) an IM value
        k0 : float
            SAC/FEMA-compatible coefficient
        k1 : float
            SAC/FEMA-compatible coefficient
        k2 : float
            SAC/FEMA-compatible coefficient, by default 0.

        Returns
        -------
        float
            intensity measure (IM)
        """
        if k2 is None:
            k2 = 0.0

        def func(x):
            return mafe_ls - self.get_mafe_limit_state(
                k0, k1, k2, s=mu, beta=x)

        s = fsolve(func, x0=0.1)
        return float(s)
