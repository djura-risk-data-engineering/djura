"""
Command-line options of the validation cases under 'tests/rs/validations'.

Registered here rather than beside the cases, since pytest reads options only
from conftest files it loads at startup, which a run from the repository root
does not reach that deep.
"""


def pytest_addoption(parser):
    """Register the plotting and reporting options"""
    group = parser.getgroup("validations")
    group.addoption(
        "--plot",
        action="store_true",
        default=False,
        help="redraw the figures of the articles being replicated")
    group.addoption(
        "--report",
        action="store_true",
        default=False,
        help="tabulate the computed values against the published ones")
    group.addoption(
        "--plot-dir",
        action="store",
        default=None,
        help="directory to write the figures and tables to, by default "
             "'_plots' alongside the validation cases")
