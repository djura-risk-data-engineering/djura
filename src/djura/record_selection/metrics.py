import warnings
import numpy as np
from scipy import integrate
from scipy.stats import lognorm

_trapezoid = np.trapezoid if np.lib.NumpyVersion(np.__version__) >= "2.0.0" \
    else np.trapz


def hellinger_distance(mu1, sigma1, mu2, sigma2, method="quadrature"):
    """
    Compute Hellinger distance between two probability distributions

    Parameters
    ----------
    mu1, sigma1 : float
        Parameters of first probability distribution
        (location and scale of underlying normal)
    mu2, sigma2 : float
        Parameters of second probability distribution
    method : str
        'quadrature' for numerical interation or
        'sampling' for discrete approximation
        'closed-form' for a closed form computation using
        medians and dispersions
        by default, 'quadrature'

    Returns
    ----------
    float : Hellinger distance (between 0 and 1)
    """
    messages = {"warnings": [], "errors": []}
    h = np.nan

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        try:
            if method.lower() == "quadrature":
                def integrand(x):
                    f1 = lognorm.pdf(x, s=sigma1, scale=np.exp(mu1))
                    f2 = lognorm.pdf(x, s=sigma2, scale=np.exp(mu2))

                    return (np.sqrt(f1) - np.sqrt(f2)) ** 2

                result, _ = integrate.quad(integrand, 0, np.inf)
                h = np.sqrt(0.5 * result)

            elif method.lower() == "sampling":
                x_max = max(np.exp(mu1 + 5 * sigma1), np.exp(mu2 + 5 * sigma2))
                x = np.logspace(-6, np.log10(x_max), 10000)

                f1 = lognorm.pdf(x, s=sigma1, scale=np.exp(mu1))
                f2 = lognorm.pdf(x, s=sigma2, scale=np.exp(mu2))

                integrand = (np.sqrt(f1) - np.sqrt(f2)) ** 2
                result = _trapezoid(integrand, x)
                h = np.sqrt(0.5 * result)

            elif method.lower() == "closed-form":

                h = np.sqrt(1 - np.sqrt(2 * sigma1
                            * sigma2 / (sigma1**2 + sigma2**2))
                            * np.exp(-((mu1 - mu2)**2
                                       / (4 * (sigma1**2 + sigma2**2)))))

            else:
                raise ValueError(
                    "Wrong method, must be 'quadrature' or 'sampling'"
                    " or 'closed-form'")

        except Exception as e:
            messages["errors"].append({
                "type": type(e).__name__,
                "message": str(e)
            })

        for warn in w:
            messages["warnings"].append({
                "message": str(warn.message),
                "category": warn.category.__name__,
                "filename": warn.filename,
                "lineno": warn.lineno
            })

        return h, messages
