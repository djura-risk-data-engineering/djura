# Portions of this file are adapted from the OpenQuake Engine
# (https://github.com/gem/oq-engine), Copyright (C) GEM Foundation,
# licensed under AGPL-3.0-or-later. See ../NOTICE.md (or ../../NOTICE.md
# for files under models/) for full attribution.

import re
import collections
import numpy

FREQUENCY_PATTERN = '^(EAS|FAS|DRVT|AvgSA)\\((\\d+\\.*\\d*)\\)'


def positivefloat(val):
    """
    Raise a ValueError if val <= 0
    """
    if val <= 0:
        raise ValueError(val)


def imt2tup(string):
    """
    >>> imt2tup('PGA')
    ('PGA',)
    >>> imt2tup('SA(1.0)')
    ('SA(1.0)', 1.0)
    >>> imt2tup('SA(1)')
    ('SA(1.0)', 1.0)
    """
    s = string.strip()
    name, *rest = s.split('(')

    if name not in globals():
        raise KeyError(name)
    elif len(name) > 12:
        raise NameError('IMT class name longer than 12 chars: %s' % name)
    elif not rest:
        if name == 'SA':
            raise ValueError('Missing period in SA')
        # no parenthesis, PGA is considered the same as PGA()
        return (s,)
    period = float(rest[0][:-1])
    return ('SA(%s)' % period, period)


def from_string(imt, _damping=5.0):
    """
    Convert an IMT string into an hazardlib object.

    :param str imt:
        Intensity Measure Type.
    """
    m = re.match(FREQUENCY_PATTERN, imt)
    if m:  # passed float interpreted as frequency
        if m.group(1) == 'EAS':
            im = EAS(float(m.group(2)))
        elif m.group(1) == 'FAS':
            im = FAS(float(m.group(2)))
        elif m.group(1) == 'DRVT':
            im = DRVT(float(m.group(2)))
        elif m.group(1) == 'AvgSA':
            im = AVGSA(float(m.group(2)))
        return im
    elif re.match(r'[ \+\d\.]+', imt):  # passed float interpreted as period
        return SA(float(imt))
    return IMT(*imt2tup(imt))


def sort_by_imt(imtls):
    """
    :param imtls: a dictionary keyed by IMT string
    :returns: a new dictionary with the keys sorted by period

    >>> sort_by_imt({'SA(10.0)': 1, 'SA(2.0)': 2})
    {'SA(2.0)': 2, 'SA(10.0)': 1}
    """
    imts = sorted(imtls, key=lambda imt: from_string(imt).period)
    return {imt: imtls[imt] for imt in imts}


def repr(self):
    if self.period and self.damping != 5.0:
        return 'SA(%s, %s)' % (self.period, self.damping)
    return self.string


IMT = collections.namedtuple('IMT', 'string period damping')
IMT.__new__.__defaults__ = (0., 5.0)
IMT.__lt__ = lambda self, other: self[1] < other[1]
IMT.__gt__ = lambda self, other: self[1] > other[1]
IMT.__le__ = lambda self, other: self[1] <= other[1]
IMT.__ge__ = lambda self, other: self[1] >= other[1]
IMT.__repr__ = repr
IMT.frequency = property(lambda self: 1. / self.period)


def PGA():
    """
    Peak ground acceleration during an earthquake measured in units
    of ``g``, times of gravitational acceleration.
    """
    return IMT('PGA')


def PGV():
    """
    Peak ground velocity during an earthquake measured in units of ``cm/sec``.
    """
    return IMT('PGV')


def PGD():
    """
    Peak ground displacement during an earthquake measured in units of ``cm``.
    """
    return IMT('PGD')


def EAS(frequency):
    """
    Effective Amplitude Spectrum in terms of a frequency (in Hz).
    """
    period = 1. / frequency
    return IMT('EAS(%.6f)' % frequency, period, 5.0)


def FAS(frequency):
    """
    Fourier Amplitude Spectrum in terms of a frequency (in Hz).
    """
    period = 1. / frequency
    return IMT('FAS(%.6f)' % frequency, period, 5.0)


def DRVT(frequency):
    """
    Duration as defined in Bora et al. (2019)
    """
    period = 1. / frequency
    return IMT('DRVT(%.6f)' % frequency, period, 5.0)


def SA(period, damping=5.0):
    """
    Spectral acceleration, defined as the maximum acceleration of a damped,
    single-degree-of-freedom harmonic oscillator. Units are ``g``, times
    of gravitational acceleration.
    """
    period = float(period)
    return IMT('SA(%s)' % period, period, damping)


def IA():
    """
    Arias intensity. Determines the intensity of shaking by measuring
    the acceleration of transient seismic waves. Units are ``m/s``.
    """
    return IMT('IA')


def CAV():
    """
    Cumulative Absolute Velocity. Defins the integral of the absolute
    acceleration time series. Units are "g-sec"
    """
    return IMT('CAV')


def RSD():
    """
    Relative significant duration, 5-95% of :class:`Arias intensity<IA>`,
    in seconds.
    """
    return IMT('RSD')


def RSD595():
    """
    Alias for RSD
    """
    return IMT('RSD595')


def RSD575():
    """
    Relative significant duration, 5-75% of :class:`Arias intensity<IA>`,
    in seconds.
    """
    return IMT('RSD575')


def RSD2080():
    """
    Relative significant duration, 20-80% of :class:`Arias intensity<IA>`,
    in seconds.
    """
    return IMT('RSD2080')


def Ds595():
    return RSD595()


def Ds575():
    return RSD575()


def AVGSA(period=None, damping=5.0):
    """
    Dummy spectral acceleration to compute average ground motion over
    several spectral ordinates. Depending on the choice of AvgSA GMPE, this
    can operate as a scalar value or as a vector quantity.
    """
    return IMT('AvgSA(%s)' % period, period, damping)\
        if period else IMT('AvgSA')


def FIV3(period, damping=5.0):
    period = float(period)
    return IMT('FIV3(%s)' % period, period, damping)


def Sa_avg2(period, damping=5.0):
    period = float(period)
    return IMT('Sa_avg2(%s)' % period, period, damping)


def Sa_avg3(period, damping=5.0):
    period = float(period)
    return IMT('Sa_avg3(%s)' % period, period, damping)


def MMI():
    """
    Modified Mercalli intensity, a Roman numeral describing the severity
    of an earthquake in terms of its effects on the earth's surface
    and on humans and their structures.
    """
    return IMT('MMI')


def JMA():
    """
    Modified Mercalli intensity, a Roman numeral describing the severity
    of an earthquake in terms of its effects on the earth's surface
    and on humans and their structures.
    """
    return IMT('JMA')


# Volcanic IMTs

def ASH():
    """
    Level of the ash fall in millimeters
    """
    return IMT('ASH')


# secondary perils

def Disp():
    """
    Displacement
    """
    return IMT('Disp')


def DispProb():
    """
    Displacement probability
    """
    return IMT('RSD595')


def LiqProb():
    """
    Liquefaction probability
    """
    return IMT('LiqProb')


def PGDMax(vert_settlement, lat_spread):
    """
    Maximum between vert_settlement and lat_spread
    """
    return numpy.maximum(vert_settlement, lat_spread)


def LSD():
    """
    Liquefaction-induced lateral spread displacements measured in
    units of ``m``.
    """
    return IMT('LSD')


def PGDGeomMean(vert_settlement, lat_spread):
    """
    Geometric mean between vert_settlement and lat_spread
    """
    return numpy.sqrt(vert_settlement * lat_spread)
