from typing import List, Union

from pandas import DataFrame, read_csv
import numpy as np
from pathlib import Path

from .intensity_measure import IntensityMeasure


class ResponseSpectrumFromGM:
    # Periods
    periods = np.arange(0, 4.01, 0.01)

    def __init__(self, damping: float, output_format: str = "dict"):
        """Initialize

        Parameters
        ----------
        damping : float
            Damping ratio
        output_format : str
            Output format, by default "dict"
        """
        self.damping = damping
        self.output_format = output_format.lower()

    def derive_response_spectrum_batch(
        self, gm_dir_path: Path, dt_filepath: Path,
        gm_filepath: Union[Path, List[Path]], periods: List = None
    ) -> None:
        """Derives response spectrum for 1 or more ground motion records
        and stores into self.rs

        Parameters
        ----------
        gm_dir_path : Path
            Path to the folder containing ground motion files
        dt_filepath : Path
            Path to a file containing time steps of each
            ground motion of interest
        gm_filepath : Union[Path, List[Path]]
            Path to a file containing filenames of each
            ground motion of interest
        periods : List, optional
            Periods used to compute the accelerations, if left None,
            uses a range between 0 and 4 seconds
        """

        if isinstance(gm_filepath, List):
            gm_files = []

            for file in gm_filepath:
                gm_files += list(read_csv(file, header=None)[0])

        else:
            gm_files = list(read_csv(gm_filepath, header=None)[0])

        dts = np.array(read_csv(dt_filepath, header=None)[0])

        rs = {}
        for i in range(len(dts)):
            acc = np.array(read_csv(
                gm_dir_path / gm_files[i], header=None)[0])
            dt = dts[i]

            _, sa = self.derive_response_spectrum(acc, dt, periods)

            rs[gm_files[i].replace('.txt', '')] = sa

        if self.output_format == "dict":
            return rs

        rs = DataFrame.from_dict(rs)

        if periods is None:
            periods = self.periods

        rs['T1'] = periods

        return rs

    def derive_response_spectrum(
            self, accelerations: List, dt: float,
            periods: List = None) -> tuple[List, any]:
        """ Derives response spectrum for a single acceleration time history

        Parameters
        ----------
        accelerations : List
            Accelerations time history
        dt : float
            Time step
        periods : List, optional
            Periods used to compute the accelerations, if left None, uses a
            range between 0 and 4 seconds

        Returns
        -------
        tuple[List, any]
            Periods in [s]
            Spectral accelerations, Union[List, float]
        """
        if periods is None:
            periods = np.arange(0, 4.01, 0.01)
        else:
            periods = np.array(periods)

        im = IntensityMeasure()

        sa = im.get_sat(periods, accelerations, dt, self.damping)

        periods = list(periods)

        return periods, list(sa)
