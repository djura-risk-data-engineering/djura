from pathlib import Path
import json
import pytest
import numpy as np
from djura.record_selection.utilities import fit_cdf_to_data
from djura.record_selection.metrics import hellinger_distance

path = Path(__file__).resolve().parent


class TestCDFFits:

    @pytest.mark.parametrize(
        "method", [
            "least_squares",
            "mle"
        ]
    )
    def test_cdf(self, method):
        data = json.load(open(path / "assets/data/cdf_fit.json"))

        x = data["iml_range"]
        y = data["probs"]

        mu, s = data["median"], data["beta"]

        fit = fit_cdf_to_data(
            x, y, distribution="lognormal",
            method=method
        )

        mu_fit, s_fit = fit["mu"], fit["sigma"]

        h = hellinger_distance(mu, s, np.exp(mu_fit), s_fit)

        print(f"Median: {mu:.3f} and dispersion: {s:.3f} of"
              " the original curve")
        print(f"Median: {np.exp(mu_fit):.3f} and dispersion: {s_fit:.3f}"
              " of the fitted curve")
        print("Hellinger distance, ", method, np.round(h[0], 3))
