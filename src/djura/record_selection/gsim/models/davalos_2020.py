# Portions of this file are adapted from the OpenQuake Engine
# (https://github.com/gem/oq-engine), Copyright (C) GEM Foundation,
# licensed under AGPL-3.0-or-later. See ../NOTICE.md (or ../../NOTICE.md
# for files under models/) for full attribution.

import numpy as np
import pandas as pd
from scipy import interpolate

from ..base import GMPE
from ..imt import FIV3, IMT
from ..contexts import Context
from .. import const
from ...utilities import get_period_im


class DavalosEtAl2020(GMPE):

    #: Supported tectonic region type is subduction interface
    DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.ACTIVE_SHALLOW_CRUST

    #: Supported intensity measure types
    DEFINED_FOR_INTENSITY_MEASURE_TYPES = {FIV3}

    #: Supported intensity measure components
    DEFINED_FOR_INTENSITY_MEASURE_COMPONENT = const.IMC.GEOMETRIC_MEAN

    #: Supported standard deviation types
    DEFINED_FOR_STANDARD_DEVIATION_TYPES = {
        const.StdDev.TOTAL, const.StdDev.INTER_EVENT, const.StdDev.INTRA_EVENT}

    #: Required rupture parameters
    REQUIRES_RUPTURE_PARAMETERS = {'mag'}

    #: Required distance measures
    REQUIRES_DISTANCES = {'rjb'}

    def get_mean_and_stddevs(self, ctx: Context, imt: IMT):
        """Provides the ground motion prediction equation for the
        FIV3

        Parameters
        ----------
        ----------
        ctx : Context
            Instance of Context class which contains the site parameters and,
            rupture and site-to-source distance parameters of the scenario.
        imt : IMT
            Instance of IMT class which describes the intensity measure type.

        Returns
        -------
        numpy.ndarray and float
            Means and stadard deviations

        Reference
        -------
        Dávalos, H., & Miranda, E. (2019). Filtered incremental velocity:
        A novel approach in intensity measures for seismic collapse estimation.
        Earthquake Engineering &amp; Structural Dynamics, 48(12), 1384–1405.
        https://doi.org/10.1002/eqe.3205

        Dávalos, H., Heresi, P., & Miranda, E. (2020). A ground motion
        prediction equation for filtered incremental velocity, FIV3.
        Soil Dynamics and Earthquake Engineering, 139, 106346.
        https://doi.org/10.1016/j.soildyn.2020.106346
        """

        imt = str(imt)
        _, period = get_period_im(imt)
        C = self.COEFFS

        # Interpolate if period is not included in the tabulated values
        if period not in C.index:
            f = interpolate.interp1d(
                np.array(C.index), C.values, axis=0,
                fill_value='extrapolate')
            new_col = pd.DataFrame(f(period).reshape(
                1, -1), index=[period], columns=self.coeff_names)

            C = pd.concat(
                [C, new_col], sort=False).sort_index()

        mag = np.array(ctx.mag).reshape(-1, 1)
        rjb = np.array(ctx.rjb).reshape(-1, 1)

        if mag < C['Mh'][period]:
            f_e = C['e1'][period] + C['e2'][period] * \
                (mag - C['Mh'][period]) + C['e3'][period] * \
                (mag - C['Mh'][period])**2
        else:
            f_e = C['e1'][period] + C['e4'][period] * (mag - C['Mh'][period])

        r_ref = 1
        r = np.sqrt(rjb**2 + C['h'][period]**2)

        f_p = (C['c1'][period] + C['c2'][period] * mag) * np.log(r / r_ref)

        mean_ln = f_e + f_p

        intra_std = C['phi'][period]
        inter_std = C['tau'][period]
        total_std = C['sigma'][period]

        return mean_ln, intra_std, inter_std, total_std

    COEFFS = np.array([[0.1, 4.6192, 1.0157, -0.1090, -0.0572, 6.300, -1.5617,
                        0.1189, 3.9605, 0.41, 0.34, 0.53],
                       [0.2, 5.2434, 1.0565, -0.0928, -0.0626, 6.300,
                           - 1.5528, 0.1185, 3.8128, 0.41, 0.34, 0.53],
                       [0.3, 5.6017, 1.1567, -0.0443, -0.0595, 6.300,
                           - 1.5410, 0.1175, 3.7151, 0.42, 0.35, 0.54],
                       [0.4, 5.7771, 1.2179, -0.0380, -0.0394, 6.300,
                           - 1.5037, 0.1125, 3.6179, 0.42, 0.35, 0.55],
                       [0.5, 5.9229, 1.3419, 0.0139, -0.0154, 6.300,
                           - 1.4658, 0.1073, 3.5361, 0.43, 0.35, 0.56],
                       [0.6, 5.9970, 1.4537, 0.0738, 0.0217, 6.300,
                           - 1.4060, 0.0990, 3.4720, 0.44, 0.36, 0.56],
                       [0.7, 6.0326, 1.5616, 0.1354, 0.0703, 6.298,
                           - 1.3324, 0.0886, 3.4158, 0.44, 0.37, 0.57],
                       [0.8, 6.0513, 1.6668, 0.1976, 0.1171, 6.289,
                           - 1.2743, 0.0802, 3.4131, 0.45, 0.37, 0.58],
                       [0.9, 6.0524, 1.8293, 0.2985, 0.1787, 6.279,
                           - 1.1952, 0.0688, 3.4128, 0.45, 0.37, 0.59],
                       [1.0, 6.0509, 1.9090, 0.3300, 0.2293, 6.271,
                           - 1.1444, 0.0612, 3.3830, 0.46, 0.38, 0.59],
                       [1.1, 6.0530, 1.9982, 0.3787, 0.2723, 6.262,
                           - 1.1061, 0.0552, 3.4091, 0.46, 0.38, 0.59],
                       [1.2, 6.0523, 2.0209, 0.3603, 0.3244, 6.254,
                           - 1.0682, 0.0491, 3.4535, 0.46, 0.38, 0.60],
                       [1.3, 6.0484, 2.0069, 0.3419, 0.3650, 6.249,
                           - 1.0416, 0.0447, 3.4692, 0.46, 0.38, 0.60],
                       [1.4, 6.0294, 2.0443, 0.3568, 0.4077, 6.238,
                           - 1.0069, 0.0393, 3.5118, 0.47, 0.39, 0.60],
                       [1.5, 6.0031, 2.0428, 0.3418, 0.4573, 6.225,
                           - 0.9642, 0.0328, 3.5573, 0.47, 0.39, 0.61],
                       [1.6, 5.9671, 2.0360, 0.2964, 0.5066, 6.210,
                           - 0.9243, 0.0269, 3.5819, 0.47, 0.38, 0.61],
                       [1.7, 5.9418, 2.0302, 0.2703, 0.5512, 6.200,
                           - 0.8833, 0.0206, 3.6928, 0.47, 0.38, 0.61],
                       [1.8, 5.9339, 2.0139, 0.2529, 0.5729, 6.200,
                           - 0.8673, 0.0180, 3.7186, 0.47, 0.39, 0.61],
                       [1.9, 5.9393, 1.9945, 0.2185, 0.5820, 6.200,
                           - 0.8745, 0.0188, 3.7758, 0.47, 0.39, 0.61],
                       [2.0, 5.9509, 2.0040, 0.2263, 0.5944, 6.200,
                           - 0.8993, 0.0217, 3.8490, 0.47, 0.39, 0.61],
                       [2.1, 5.9744, 2.0701, 0.2981, 0.5798, 6.200,
                           - 0.9196, 0.0245, 3.8949, 0.47, 0.38, 0.61],
                       [2.2, 5.9769, 2.0888, 0.2937, 0.5753, 6.200,
                           - 0.9402, 0.0274, 3.9146, 0.47, 0.38, 0.60],
                       [2.3, 6.0022, 2.1712, 0.3696, 0.5557, 6.200,
                           - 0.9481, 0.0284, 3.9192, 0.47, 0.37, 0.60],
                       [2.4, 5.9884, 2.1313, 0.3285, 0.5735, 6.200,
                           - 0.9264, 0.0254, 3.9077, 0.47, 0.37, 0.60],
                       [2.5, 5.9971, 2.1818, 0.3689, 0.5659, 6.200,
                           - 0.9307, 0.0262, 3.8965, 0.47, 0.37, 0.60],
                       [2.6, 5.9685, 2.0729, 0.2527, 0.5971, 6.200,
                           - 0.9190, 0.0246, 3.9325, 0.47, 0.37, 0.60],
                       [2.7, 5.9850, 2.1692, 0.3453, 0.5776, 6.200,
                           - 0.9184, 0.0249, 3.9493, 0.47, 0.37, 0.60],
                       [2.8, 5.9663, 2.2062, 0.3815, 0.6026, 6.200,
                           - 0.8816, 0.0196, 3.9589, 0.47, 0.36, 0.59],
                       [2.9, 5.9402, 2.2186, 0.3988, 0.6237, 6.200,
                           - 0.8672, 0.0177, 3.9860, 0.47, 0.35, 0.58],
                       [3.0, 5.9444, 2.2662, 0.4480, 0.6109, 6.200, -0.8711,
                        0.0185, 3.9792, 0.47, 0.37, 0.60]])

    coeff_names = ['e1', 'e2', 'e3', 'e4', 'Mh',
                   'c1', 'c2', 'h', 'phi', 'tau', 'sigma']
    COEFFS = pd.DataFrame(data=COEFFS[:, 1:],
                          index=COEFFS[:, 0],
                          columns=coeff_names)
