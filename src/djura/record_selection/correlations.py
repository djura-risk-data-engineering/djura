from typing import List
from numpy import zeros

from . import correlation_models


class Correlations:
    def __init__(self) -> None:
        """Performs calculations of correlations among various IM types
        """
        pass

    def get_correlation(self, period_i: float, period_j: float,
                        correlation_model: str) -> float:
        """Compute the inter-period correlation for any two Sa(T) values

        Parameters
        ----------
        period_i : float
            first period
        period_j : float
            second period
        correlation_model : str
            Correlation model name, supported 'baker_jayaram', 'akkar'

        Returns
        -------
        float
            Predicted correlation coefficient

        Raises
        ------
        ValueError
            Not a valid correlation function if wront GCIM.corr_func is
            provided
        """
        correlation_function_handles = {
            'baker_jayaram': correlation_models.baker_jayaram,
            'akkar': correlation_models.akkar,
        }

        # Check for existing correlation function
        if correlation_model not in correlation_function_handles:
            raise ValueError('Not a valid correlation function')
        else:
            rho = correlation_function_handles[correlation_model](
                period_i, period_j)

        return rho

    def get_correlation_matrix(self, periods: List, model: str):
        """Get correlation matrix

        Parameters
        ----------
        periods : List
            Periods
        model : str
            Name of correlation model

        Returns
        -------
        np.ndarray
            Correlation matrix
        """
        corr = zeros((len(periods), len(periods)))

        for i, period_i in enumerate(periods):
            for j, period_j in enumerate(periods):

                corr[i, j] = self.get_correlation(period_i, period_j, model)

        return corr
