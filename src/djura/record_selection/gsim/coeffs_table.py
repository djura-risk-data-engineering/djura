# Portions of this file are adapted from the OpenQuake Engine
# (https://github.com/gem/oq-engine), Copyright (C) GEM Foundation,
# licensed under AGPL-3.0-or-later. See ../NOTICE.md (or ../../NOTICE.md
# for files under models/) for full attribution.


import re
from math import log
from scipy.interpolate import interp1d
import numpy as np
from ..utilities import RecordBuilder
from .imt import from_string

SA_LIKE_PREFIXES = ['SA', 'EA', 'FA', 'DR']


class CoeffsTable(object):
    @classmethod
    def fromdict(cls, ddic, logratio=True, opt=0):
        """
        :param ddic: a dictionary of dictionaries
        :param logratio: flag (default True)
        :param opt: int (default 0)
        """
        firstdic = ddic[next(iter(ddic))]
        self = object.__new__(cls)
        self.rb = RecordBuilder(**firstdic)
        self._coeffs = {imt: self.rb(**dic) for imt, dic in ddic.items()}
        self.logratio = logratio
        self.opt = opt
        return self

    def __init__(self, table, **kwargs):
        self._coeffs = {}  # cache
        self.opt = kwargs.pop('opt', 0)
        self.logratio = kwargs.pop('logratio', True)
        sa_damping = kwargs.pop('sa_damping', None)
        if kwargs:
            raise TypeError('CoeffsTable got unexpected kwargs: %r' % kwargs)
        self.rb = self._setup_table_from_str(table, sa_damping)
        if self.opt == 1:
            keys = list(self._coeffs.keys())
            num_coeff = len(self._coeffs[keys[0]])
            self.cmtx = np.zeros((len(self._coeffs.keys()), num_coeff))
            periods = np.array([i.period for i in keys])
            idxs = np.argsort(periods)
            tmp = []
            for i, idx in enumerate(idxs):
                key = keys[i]
                tmp.append(np.array(self._coeffs[key].tolist()))
            tmp = np.array(tmp)
            self.cmtx = tmp[idxs, :]
            self.periods = periods[idxs]

    def _setup_table_from_str(self, table, sa_damping):
        """
        Builds the input tables from a string definition
        """
        lines = table.strip().splitlines()
        header = lines.pop(0).split()
        if not header[0].upper() == "IMT":
            raise ValueError('first column in a table must be IMT')
        dt = RecordBuilder(**{name: 0. for name in header[1:]})
        for line in lines:
            row = line.split()
            imt_name_or_period = row[0].upper()
            if imt_name_or_period == 'SA':  # protect against stupid mistakes
                raise ValueError('specify period as float value '
                                 'to declare SA IMT')
            imt = from_string(imt_name_or_period, sa_damping)
            self._coeffs[imt] = dt(*row[1:])
        return dt

    @property
    def sa_coeffs(self):
        return {imt: self._coeffs[imt] for imt in self._coeffs
                if imt.string[:2] in SA_LIKE_PREFIXES}

    @property
    def non_sa_coeffs(self):
        return {imt: self._coeffs[imt] for imt in self._coeffs
                if imt.string[:2] not in SA_LIKE_PREFIXES}

    def get_coeffs(self, coeff_list):
        """
        :param coeff_list:
            A list with the names of the coefficients
        """
        coeffs = []
        pof = []
        for imt in self._coeffs:
            if re.search('^(SA|EAS|FAS|DRVT)', imt.string):
                tmp = np.array(self._coeffs[imt])
                coeffs.append([tmp[i] for i in coeff_list])
                if re.search('^(SA)', imt.string):
                    pof.append(imt.period)
                elif re.search('^(EAS|FAS|DRVT)', imt.string):
                    pof.append(imt.frequency)
                else:
                    raise ValueError('Unknown IMT: {:s}'.format(imt.string))
        pof = np.array(pof)
        coeffs = np.array(coeffs)
        idx = np.argsort(pof)
        pof = pof[idx]
        coeffs = coeffs[idx, :]
        return pof, coeffs

    def __getitem__(self, imt):
        try:  # see if already in cache
            return self._coeffs[imt]
        except KeyError:  # populate the cache
            pass

        if self.opt == 0:
            max_below = min_above = None
            for unscaled_imt in list(self.sa_coeffs):
                if unscaled_imt.damping != getattr(imt, 'damping', None):
                    pass
                elif unscaled_imt.period > imt.period:
                    if (min_above is None
                            or unscaled_imt.period < min_above.period):
                        min_above = unscaled_imt
                elif unscaled_imt.period < imt.period:
                    if (max_below is None
                            or unscaled_imt.period > max_below.period):
                        max_below = unscaled_imt
            if max_below is None or min_above is None:
                raise KeyError(imt)
            if self.logratio:  # regular case
                # ratio tends to 1 when target period tends to a minimum
                # known period above and to 0 if target period is close
                # to maximum period below.
                ratio = ((log(imt.period) - log(max_below.period))
                         / (log(min_above.period)
                            - log(max_below.period)))
            else:  # in the ACME project
                ratio = ((imt.period - max_below.period)
                         / (min_above.period - max_below.period))
            below = self.sa_coeffs[max_below]
            above = self.sa_coeffs[min_above]
            lst = [(above[n] - below[n]) * ratio + below[n]
                   for n in self.rb.names]
            self._coeffs[imt] = c = self.rb(*lst)

        elif self.opt == 1:
            if imt.period < self.periods[0] or imt.period > self.periods[-1]:
                raise KeyError(imt)
            fit = interp1d(np.log10(self.periods), self.cmtx,
                           axis=0, kind='cubic')
            vals = fit(np.log10(imt.period))
            self._coeffs[imt] = c = self.rb(*vals)
        return c

    def __repr__(self):
        return '<%s %s>' % (self.__class__.__name__, ' '.join(self.rb.names))
