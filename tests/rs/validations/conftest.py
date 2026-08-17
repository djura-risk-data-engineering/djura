"""
Options shared by the validation cases.

The figures of the articles being replicated can be redrawn from the results
of the validation runs, which is useful when a comparison is being judged by
eye rather than by a tolerance. Plotting is off by default, requires
matplotlib, which is not a dependency of the package, and is requested with

    pytest tests/rs/validations --plot

The figures are written to 'tests/rs/validations/_plots' unless another
directory is given with '--plot-dir'.
"""

from pathlib import Path

import pytest


def pytest_addoption(parser):
    """Register the plotting options"""
    group = parser.getgroup("validations")
    group.addoption(
        "--plot",
        action="store_true",
        default=False,
        help="redraw the figures of the articles being replicated")
    group.addoption(
        "--plot-dir",
        action="store",
        default=None,
        help="directory to write the figures to, by default '_plots' "
             "alongside the validation cases")


@pytest.fixture(scope="session")
def plot_dir(request):
    """Directory to write figures to, skipping when plotting is not requested

    Returns
    -------
    pathlib.Path
        An existing directory
    """
    if not request.config.getoption("--plot"):
        pytest.skip("figures are drawn only when --plot is given")

    pytest.importorskip(
        "matplotlib", reason="matplotlib is required to draw the figures")

    directory = request.config.getoption("--plot-dir")
    directory = Path(directory) if directory is not None \
        else Path(__file__).resolve().parent / "_plots"
    directory.mkdir(parents=True, exist_ok=True)

    return directory
