# Portions of this file are adapted from the OpenQuake Engine
# (https://github.com/gem/oq-engine), Copyright (C) GEM Foundation,
# licensed under AGPL-3.0-or-later. See ../NOTICE.md (or ../../NOTICE.md
# for files under models/) for full attribution.

'''
class GenericGmpeAvgSA(GMPE):
    """
    Implements a modified GMPE class that can be used to compute average
    ground motion over several spectral ordinates from an arbitrary GMPE.
    The mean and standard deviation are computed according to:
    Kohrangi M., Reddy Kotha S. and Bazzurro P., 2018, Ground-motion models
    for average spectral acceleration in a period range: direct and indirect
    methods, Bull. Earthquake. Eng., 16, pp. 45–65.
    Note that only the Total Standard Deviation is supported.

    :param string gmpe_name:
        The name of a GMPE class used for the calculation.

    :param list avg_periods:
        List of averaging periods (must be a subset of the periods allowed
        in the selected GMPE)

    :param string corr_func:
        Handle of the function to compute correlation coefficients between
        different spectral acceleration ordinates. Valid options are:
        'baker_jayaram', 'akkar', 'eshm20', 'none'. Default is none.
    """

    # Parameters
    REQUIRES_SITES_PARAMETERS = set()
    REQUIRES_DISTANCES = set()
    REQUIRES_RUPTURE_PARAMETERS = set()
    DEFINED_FOR_INTENSITY_MEASURE_COMPONENT = ''
    DEFINED_FOR_INTENSITY_MEASURE_TYPES = {AVGSA}
    DEFINED_FOR_STANDARD_DEVIATION_TYPES = {const.StdDev.TOTAL}
    DEFINED_FOR_TECTONIC_REGION_TYPE = ''

    def __init__(self, gmpe_name, avg_periods, corr_func='none', **kwargs):

        super().__init__(gmpe_name=gmpe_name, avg_periods=avg_periods,
                         corr_func=corr_func, **kwargs)
        self.gmpe = registry[gmpe_name](**kwargs)
        # Combine the parameters of the GMPE provided at the construction
        # level with the ones assigned to the average GMPE.
        for key in dir(self):
            if key.startswith('REQUIRES_'):
                setattr(self, key, getattr(self.gmpe, key))
            if key.startswith('DEFINED_'):
                if not key.endswith('FOR_INTENSITY_MEASURE_TYPES'):
                    setattr(self, key, getattr(self.gmpe, key))

        # Ensure that it is always recogised that the AvgSA GMPE is defined
        # only for total standard deviation even if the called GMPE is
        # defined for inter- and intra-event standard deviations too
        self.DEFINED_FOR_STANDARD_DEVIATION_TYPES = {const.StdDev.TOTAL}
        self.avg_periods = avg_periods
        self.tnum = len(self.avg_periods)

        # Check for existing correlation function
        if corr_func not in CORRELATION_FUNCTION_HANDLES:
            raise ValueError('Not a valid correlation function')
        else:
            self.corr_func = \
                CORRELATION_FUNCTION_HANDLES[corr_func](avg_periods)

        # Check if this GMPE has the necessary requirements
        # TO-DO

    def compute(self, ctx: np.recarray, imt: IMT):
        """
        :param imts: must be a single IMT of kind AvgSA
        """
        sas = [SA(period) for period in self.avg_periods]
        out = contexts.get_mean_stds(self.gmpe, ctx, sas)

        stddvs_avgsa = 0.
        for i1 in range(self.tnum):
            mean[:] += out[0, i1]
            for i2 in range(self.tnum):
                rho = self.corr_func(i1, i2)
                stddvs_avgsa += rho * out[1, i1] * out[1, i2]

        mean[:] /= self.tnum
        sig[:] = np.sqrt(stddvs_avgsa) / self.tnum

'''
