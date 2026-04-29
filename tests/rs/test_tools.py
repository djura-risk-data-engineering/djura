import pytest
from pathlib import Path

from src.read_record import read_esm, read_nga
from src.gsim.models import AristeidouEtAl2024

path = Path(__file__).resolve().parent


@pytest.mark.parametrize(
    "content", [
        (None),
        (True),
    ]
)
def test_read_nga(content):
    filename = path / "assets/records/nga_gm.AT2"

    if content:
        with open(filename, 'r') as file:
            content = file.readlines()

    t, acc, desc = read_nga(filename, content)

    assert len(acc) == len(t) == 8000
    assert t[1] - t[0] == pytest.approx(0.005, abs=1e-4)
    assert isinstance(desc, str)


@pytest.mark.parametrize(
    "content", [
        (None),
        (True),
    ]
)
def test_read_esm(content):
    filename = path / "assets/records/esm_gm.asc"

    if content:
        with open(filename, 'r') as file:
            content = file.readlines()

    t, acc, desc = read_esm(filename, content)

    assert len(acc) == len(t) == 5590
    assert t[1] - t[0] == pytest.approx(0.01, abs=1e-4)
    assert isinstance(desc, list)


class TestAristeidou:
    SUGGESTED_LIMITS = {
        'magnitude': [4.5, 7.9],
        'Rjb': [0.0, 299.44],
        'Rrup': [0.07, 299.59],
        'D_hyp': [2.3, 18.65],
        'Vs30': [106.83, 1269.78],
        'mechanism': [0, 4],
        'Z2pt5': [0.0, 7780],
        'Rx': [-297.13, 292.39],
        'Ztor': [0, 16.23]
    }

    def test_suggested_limits(self):
        limits = AristeidouEtAl2024().get_suggested_parameter_limits()

        assert limits == self.SUGGESTED_LIMITS

    def test_supported_ims(self):
        ims = AristeidouEtAl2024().get_supported_ims()

        assert len(ims) == 169
