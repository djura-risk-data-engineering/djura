"""
A GCIM analysis needs a correlation model for every pair of the intensity
measures considered, so the set of IMs which may be analysed together is
restricted to those which are mutually supported.
"""

import pytest

from djura.record_selection._gcim import _GCIM
from djura.record_selection.constants import (
    CORRELATION_MODELS, SUPPORTED_IMS, get_compatible_ims,
    is_correlation_supported)


class TestIsCorrelationSupported:

    def test_an_im_is_always_correlated_with_itself(self):
        for im in SUPPORTED_IMS:
            assert is_correlation_supported(im, im)

    def test_lookup_is_order_independent(self):
        assert is_correlation_supported("ASI", "PGA")
        assert is_correlation_supported("PGA", "ASI")

    @pytest.mark.parametrize(
        "im_i, im_j, out", [
            ("IA", "ASI", True),
            ("IA", "SI", True),
            ("IA", "DSI", True),
            ("IA", "CAV", True),
            ("IA", "FIV3", False),
            ("IA", "Sa_avg2", False),
        ]
    )
    def test_pairs(self, im_i, im_j, out):
        assert is_correlation_supported(im_i, im_j) is out


class TestCompatibleIms:

    def test_includes_itself(self):
        assert "IA" in get_compatible_ims("IA")

    def test_arias_intensity(self):
        """The Bradley (2015) correlations cover the amplitude, duration and
        cumulative IMs, but not the next-generation ones
        """
        compatible = get_compatible_ims("IA")

        assert {"SA", "PGA", "PGV", "ASI", "SI", "DSI", "CAV",
                "Ds575", "Ds595"} <= compatible
        assert not {"FIV3", "Sa_avg2", "Sa_avg3"} & compatible

    def test_is_symmetric(self):
        for im in SUPPORTED_IMS:
            for other in get_compatible_ims(im):
                assert im in get_compatible_ims(other)

    def test_unsupported_im(self):
        with pytest.raises(ValueError) as exc:
            get_compatible_ims("NotAnIM")

        assert "is not supported" in str(exc.value)


class TestValidateCorrelationPairs:

    @pytest.mark.parametrize(
        "imi, im_star", [
            # Bradley (2010), complete since the 2015 IA correlations
            (["SA", "SI", "ASI", "IA", "Ds595"], "SA"),
            # Bradley (2012), the closed nine-IM set
            (["SA", "PGA", "PGV", "ASI", "SI", "DSI", "CAV",
              "Ds575", "Ds595"], "SA"),
            # The same set extended with IA
            (["SA", "PGA", "PGV", "ASI", "SI", "DSI", "CAV",
              "Ds575", "Ds595", "IA"], "SA"),
        ]
    )
    def test_supported_sets_pass(self, imi, im_star):
        _GCIM({})._validate_correlation_pairs(
            {im: [] for im in imi}, {"type": im_star})

    def test_unsupported_pair_raises(self):
        with pytest.raises(ValueError) as exc:
            _GCIM({})._validate_correlation_pairs(
                {"IA": [], "FIV3": [1.0]}, {"type": "SA"})

        message = str(exc.value)
        assert "FIV3-IA" in message
        # The message must say what may be used instead
        assert "FIV3 may only be combined with" in message
        assert "IA may only be combined with" in message

    def test_conditioning_im_is_included_in_the_check(self):
        """IMi alone is fine here; the clash is with IM*"""
        _GCIM({})._validate_correlation_pairs({"FIV3": [1.0]}, {"type": "SA"})

        with pytest.raises(ValueError) as exc:
            _GCIM({})._validate_correlation_pairs(
                {"FIV3": [1.0]}, {"type": "IA"})

        assert "FIV3-IA" in str(exc.value)

    def test_every_unsupported_pair_is_reported_at_once(self):
        with pytest.raises(ValueError) as exc:
            _GCIM({})._validate_correlation_pairs(
                {"IA": [], "FIV3": [1.0], "Sa_avg2": [1.0]})

        message = str(exc.value)
        assert "FIV3-IA" in message
        assert "IA-Sa_avg2" in message

    def test_unconditional_omits_the_conditioning_im(self):
        _GCIM({})._validate_correlation_pairs({"SA": [1.0], "IA": []})


class TestSelectCorrelationModel:
    """The correlation model used for each IM pair may be chosen, so that a
    published study can be reproduced with the equations it adopted
    """

    def test_first_registered_model_is_the_default(self):
        gcim = _GCIM({})

        assert gcim._select_correlation_model("SA-SA") == \
            CORRELATION_MODELS["SA-SA"][0]

    def test_requested_model_overrides_the_default(self):
        gcim = _GCIM({})

        assert gcim._select_correlation_model(
            "SA-SA", {"SA-SA": "baker_jayaram"}) == "baker_jayaram"

    def test_pair_is_matched_in_either_order(self):
        gcim = _GCIM({})

        assert gcim._select_correlation_model(
            "SA-SA", {"SA-SA": "akkar"}) == "akkar"
        assert gcim._select_correlation_model(
            "IA-SA", {"SA-IA": "baker2007_ia_sa"}) == "baker2007_ia_sa"

    def test_unnamed_pairs_keep_the_default(self):
        gcim = _GCIM({})

        assert gcim._select_correlation_model(
            "IA-SA", {"SA-SA": "baker_jayaram"}) == \
            CORRELATION_MODELS["IA-SA"][0]

    def test_chosen_model_is_recorded(self):
        gcim = _GCIM({})

        gcim._select_correlation_model("SA-SA", {"SA-SA": "baker_jayaram"})
        gcim._select_correlation_model("IA-SA")

        assert gcim.correlation_models_used == {
            "SA-SA": "baker_jayaram",
            "IA-SA": CORRELATION_MODELS["IA-SA"][0],
        }


class TestValidateCorrelationModels:

    def test_no_request_is_valid(self):
        _GCIM({})._validate_correlation_models(None)
        _GCIM({})._validate_correlation_models({})

    def test_supported_request_is_valid(self):
        _GCIM({})._validate_correlation_models({"SA-SA": "baker_jayaram"})

    def test_malformed_pair(self):
        with pytest.raises(ValueError) as exc:
            _GCIM({})._validate_correlation_models({"SA": "baker_jayaram"})

        assert "is not of the form" in str(exc.value)

    def test_unsupported_pair(self):
        with pytest.raises(ValueError) as exc:
            _GCIM({})._validate_correlation_models({"IA-FIV3": "aso2024"})

        assert "not a supported intensity measure pair" in str(exc.value)

    def test_model_not_available_for_the_pair(self):
        with pytest.raises(ValueError) as exc:
            _GCIM({})._validate_correlation_models(
                {"SA-SA": "bradley2011_pga"})

        assert "is not available for" in str(exc.value)
