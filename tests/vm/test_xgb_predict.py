import pytest

from djura.edp_im.XGBPredict import XGBPredict


class TestXGBPredict:
    @pytest.mark.parametrize("im_type, collapse, expected", [
        ("sa", True, "R"),
        ("SAAVG", False, "ro_2"),
        ("SAAVG", True, "ro_3"),
        ("", True, ValueError),
    ])
    def test_arguments(self, im_type, collapse, expected):
        if im_type != "":
            model = XGBPredict(im_type, collapse)
            assert model.parameter == expected
        else:
            with pytest.raises(expected):
                XGBPredict(im_type, collapse)
